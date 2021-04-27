import numpy as np
import os
import yaml
import argparse

def get_pixel_motion_caused_by_camera_motion(root_folder):
    '''
    Computes the expected pixel motion caused by known camera motion using epipolar geometry.
    The camera motion is measured using proprioceptive sensors on the robot (encoders mostly),
    which is read here from the 'camera_matrix.npy' file, representing the extrinsic camera
    calibration matrices at each time step.
    '''
    with open('rgb_camera_calibration.txt', 'r') as fp:
        cam_calibration = yaml.safe_load(fp)
    fx = cam_calibration['K'][0]
    cx = cam_calibration['K'][2]
    fy = cam_calibration['K'][4]
    cy = cam_calibration['K'][5]
    width = cam_calibration['width']
    height = cam_calibration['height']

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    camera_poses = np.load(os.path.join(root_folder, 'camera_matrix.npy'))
    camera_poses = camera_poses[::3]

    # Here we simplify the problem by making two assumptions:
    # 1. assume only camera translations are present (i.e. no rotations)
    # 2. assume a fixed depth in the scene
    # Because of these assumptions, all pixels are assumed to have translated by the same amount.
    # Hence we do not need to compute the similarity transform using correspondence pairs.
    # We only compute the pixel translation using Eq. 9.6 in Hartley and Zimmerman

    fixed_depth = 0.5
    # Get camera positions only
    x = camera_poses[:, 0, -1]
    y = camera_poses[:, 1, -1]
    z = camera_poses[:, 2, -1]
    # get relative translation
    xdiff = np.diff(x)
    ydiff = np.diff(y)
    zdiff = np.diff(z)

    expected_pixel_motion = []
    for xtrans, ytrans, ztrans in zip(xdiff, ydiff, zdiff):
        # move from base frame to camera frame
        t = np.array([ytrans, ztrans, xtrans])
        # (x' - x) = K*t/Z
        pixel_motion = K.dot(t) / fixed_depth
        pixel_motion = pixel_motion[:2] # homogeneous coordinates
        expected_pixel_motion.append(pixel_motion.tolist())
    expected_pixel_motion = np.array(expected_pixel_motion)
    return expected_pixel_motion

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('trial_data_folder', type=str, help="Path to trial data folder")
    args = parser.parse_args()
    root_folder = args.trial_data_folder

    expected_pixel_motion = get_pixel_motion_caused_by_camera_motion(root_folder)
    np.save(os.path.join(root_folder, 'expected_pixel_motion.npy'), expected_pixel_motion)


if __name__ == '__main__':
    main()
