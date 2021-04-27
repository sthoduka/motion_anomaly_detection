import os
import glob
import argparse
import numpy as np

from urdfpy import URDF
import pyrender
import PIL
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('trial_data_folder', type=str, help="Path to trial data folder")
    parser.add_argument('robot_urdf_path', type=str, help="path to robot URDF file")
    parser.add_argument('-p', '--preview', action="store_true", help="Don't save files, only view rendered images with viewer")
    args = parser.parse_args()
    trial_data_folder = args.trial_data_folder

    robot = URDF.load(args.robot_urdf_path)
    scene, mesh_nodes = robot.create_scene()
    # Load intrinsic camera calibration matrix here
    # The included file contains the intrinsic camera calibration matrix
    # of the ASUS xtion Pro on the HSR robot which was used
    # to collect this dataset: https://zenodo.org/record/4578539
    with open('rgb_camera_calibration.txt', 'r') as fp:
        cam_calibration = yaml.safe_load(fp)
    fx = cam_calibration['K'][0]
    cx = cam_calibration['K'][2]
    fy = cam_calibration['K'][4]
    cy = cam_calibration['K'][5]
    width = cam_calibration['width']
    height = cam_calibration['height']

    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.1, zfar=100.0)

    # This has no effect on the final result and is only for initialization.
    # It can be taken from the first entry in the 'camera_matrix.npy' file, which contains
    # the camera poses (w.r.t base_link)
    initial_cam_pose = np.array([[ 1.73560879e-05,  4.79423570e-01, -8.77583637e-01,  5.31304827e-02],
                                 [-1.00000000e+00,  8.32091554e-06, -1.52314198e-05,  2.20009221e-02],
                                 [-2.34767761e-12,  8.77583637e-01,  4.79423570e-01,  1.02904540e+00],
                                 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

    camera_node = scene.add(camera, pose=initial_cam_pose)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
    light_node = pyrender.Node(light=light, matrix=np.eye(4))
    scene.add_node(light_node, parent_node=camera_node)
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)

    camera_poses = np.load(os.path.join(trial_data_folder, 'camera_matrix.npy'))
    joint_positions = np.load(os.path.join(trial_data_folder, 'joint_state_positions.npy'))
    joint_names = np.load(os.path.join(trial_data_folder, 'joint_names.npy'))
    joint_names = [n.decode() for n in joint_names]

    img_folder = os.path.join(trial_data_folder, 'rendered_body')
    if not os.path.exists(img_folder):
        os.mkdir(img_folder)

    # get name of first frame in 'rgb' folder
    first_frame = sorted(glob.glob(os.path.join(trial_data_folder, 'rgb') + '/*.jpg'))[0]
    first_frame = os.path.basename(first_frame)
    # file name has format 'frame_%04d.jpg', so extract the 4 digit frame number
    frame_num = int(first_frame[6:-4])

    if args.preview:
        viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True)

    for mat, joint_pos in zip(camera_poses, joint_positions):
        cfg = dict()
        for name, position in zip(joint_names, joint_pos):
            cfg[name] = position
        if args.preview:
            viewer.render_lock.acquire()
            scene.set_pose(camera_node, pose=mat)
            robot.set_scene(scene, mesh_nodes, cfg=cfg)
            viewer.render_lock.release()
        else:
            scene.set_pose(camera_node, pose=mat)
            robot.set_scene(scene, mesh_nodes, cfg=cfg)
            color, depth = renderer.render(scene)
            color_img = PIL.Image.fromarray(color)
            filename = 'frame_%04d.png' % frame_num
            print(filename)
            color_img.save(os.path.join(img_folder, filename))
            frame_num += 1

if __name__ == '__main__':
    main()
