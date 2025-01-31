import os
import random
import glob
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torchvision
from torchvision import transforms
import torch
from PIL import Image, ImageOps


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        img.convert('L')
    return img

def make_dataset_flow_images(directory, flow_type='normal'):
    if not os.path.exists(directory):
        print('Path %s does not exist ' % directory)
        exit(0)

    ds_optical_flow_x = []
    ds_optical_flow_y = []
    ds_body_optical_flow_x = []
    ds_body_optical_flow_y = []
    ds_rendered_body = []

    ds_frames_per_video = []
    ds_annotation = []

    directory = os.path.expanduser(directory)
    trials = sorted(glob.glob(os.path.join(directory, '*')))
    trial_id = 0
    for trial in trials:
        body_flow_folder = 'body_flow'
        if flow_type == 'normal' or flow_type == 'normal_masked':
            flow_folder = 'flow'
        elif flow_type == 'registered' or flow_type =='registered_masked':
            flow_folder = 'flow_registered'

        flow_x_files = sorted(glob.glob(os.path.join(trial, flow_folder, 'framex*.jpg')))
        flow_y_files = sorted(glob.glob(os.path.join(trial, flow_folder, 'framey*.jpg')))
        body_flow_x_files = sorted(glob.glob(os.path.join(trial, body_flow_folder, 'framex*.jpg')))
        body_flow_y_files = sorted(glob.glob(os.path.join(trial, body_flow_folder, 'framey*.jpg')))
        annotation = np.zeros(len(flow_x_files))

        rgb_files = sorted(glob.glob(os.path.join(trial, 'rgb', '*.jpg')))

        json_file = os.path.join(trial, 'annotation.json')
        with open(json_file) as jdat:
            data = json.load(jdat)
            if 'Anomalies' in data.keys():
                for anomaly in data["Anomalies"]:
                    for idx in range(anomaly[0], anomaly[1]+1):
                        frame_id = int(rgb_files[idx][-8:-4])
                        flow_file = os.path.join(trial, flow_folder, 'framex_%04d.jpg' % frame_id)
                        try:
                            flow_idx = flow_x_files.index(flow_file)
                            annotation[flow_idx] = 1
                        except:
                            continue


        ds_optical_flow_x.extend(flow_x_files)
        ds_optical_flow_y.extend(flow_y_files)
        ds_body_optical_flow_x.extend(body_flow_x_files)
        ds_body_optical_flow_y.extend(body_flow_y_files)
        ds_frames_per_video.append(len(flow_x_files))
        ds_annotation.extend(annotation)
        for flow in flow_x_files:
            frame_id = int(flow[-8:-4])
            rendered_body = os.path.join(trial, 'rendered_body', 'frame_%04d.png'%frame_id)
            ds_rendered_body.append(rendered_body)
        trial_id += 1
    data = {}
    data['rendered_body'] = np.array(ds_rendered_body)
    data['ofx'] = np.array(ds_optical_flow_x)
    data['ofy'] = np.array(ds_optical_flow_y)
    data['ofx_body'] = np.array(ds_body_optical_flow_x)
    data['ofy_body'] = np.array(ds_body_optical_flow_y)
    data['num_frames_per_video'] = np.array(ds_frames_per_video)
    data['annotation'] = np.array(ds_annotation)

    return data

class OpticalFlowPair(torch.utils.data.Dataset):
    def __init__(self, root, flow_type='normal', frame_offset_start=5, frame_offset=9, transform=None, train=True):
        self.root = root
        self.flow_type = flow_type
        samples = make_dataset_flow_images(self.root, flow_type=flow_type)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            raise RuntimeError(msg)

        self.img_loader = pil_loader
        self.samples = samples

        self.frame_offset = frame_offset
        self.frame_offset_start = frame_offset_start

        self.frames_per_video = np.cumsum(self.samples['num_frames_per_video'])

        self.transform = transform

        if self.flow_type == 'normal_masked' or self.flow_type == 'registered_masked':
            cv2.setNumThreads(0)

        self.time_index = self.frame_offset
        self.train = train


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        annotation = self.samples['annotation'][index]

        # 1. Load target optical flow image and corresponding body flow image
        ofx = self.samples['ofx'][index]
        ofy = self.samples['ofy'][index]
        of_img = self.stack_channels(ofx, ofy)

        ofx_body = self.samples['ofx_body'][index]
        ofy_body = self.samples['ofy_body'][index]
        of_body_img = self.stack_channels(ofx_body, ofy_body)


        if self.flow_type == 'normal_masked' or self.flow_type == 'registered_masked':
            body_img = self.samples['rendered_body'][index]
            body_img = self.img_loader(body_img)
            body_img = ImageOps.grayscale(body_img)
            body_img = np.asarray(body_img)
            mask = self.create_mask(body_img)
            of_img[mask == 0] = 128

        of_img = torch.from_numpy(of_img)
        of_body_img = torch.from_numpy(of_body_img)

        if self.transform is not None:
            of_img = self.transform(of_img)
            of_body_img = self.transform(of_body_img)
            of_img = of_img - 0.5
            of_body_img = of_body_img - 0.5

        # 2. Determine index of input optical flow image to load
        if self.train:
            random_frame_offset = random.randint(self.frame_offset_start, self.frame_offset)
        else:
            random_frame_offset = self.time_index

        for i in range(random_frame_offset):
            prev_index = index - (random_frame_offset - i)
            curr_video_id = np.argmax(self.frames_per_video > index)
            prev_video_id = np.argmax(self.frames_per_video > prev_index)
            if curr_video_id == prev_video_id:
                break
        if curr_video_id != prev_video_id or prev_index < 0:
            prev_index = index

        # 3. Load input optical flow image and corresponding body flow image
        ofx_prev = self.samples['ofx'][prev_index]
        ofy_prev = self.samples['ofy'][prev_index]
        of_img_prev = self.stack_channels(ofx_prev, ofy_prev)

        ofx_body_prev = self.samples['ofx_body'][prev_index]
        ofy_body_prev = self.samples['ofy_body'][prev_index]
        of_body_img_prev = self.stack_channels(ofx_body_prev, ofy_body_prev)


        if self.flow_type == 'normal_masked' or self.flow_type == 'registered_masked':
            of_img_prev[mask == 0] = 128

        of_img_prev = torch.from_numpy(of_img_prev)
        of_body_img_prev = torch.from_numpy(of_body_img_prev)

        if self.transform is not None:
            of_img_prev = self.transform(of_img_prev)
            of_body_img_prev = self.transform(of_body_img_prev)
            of_img_prev = of_img_prev - 0.5
            of_body_img_prev = of_body_img_prev - 0.5

        return of_img_prev, of_body_img_prev, of_img, of_body_img, annotation

    def __len__(self):
        return len(self.samples['ofx'])

    def create_mask(self, bodyimg):
        ret, img = cv2.threshold(bodyimg, 250, 255, 0)
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, 0, (0,255,0), 3)
        img = img[np.newaxis, :, :]
        img = np.repeat(img, 2, axis=0)
        return img

    def stack_channels(self, path1, path2):
        img1 = self.img_loader(path1)
        img2 = self.img_loader(path2)
        img1 = np.asarray(img1)
        img2 = np.asarray(img2)
        img = np.stack((img1, img2), axis=0)
        return img


    def set_time_index(self, idx):
        self.time_index = idx



def main():
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
        ]
    )
    dataset = OpticalFlowPair('/media/santosh/TOSHIBA_EXT/phd/lucy/submitted/validation', flow_type='normal_masked', transform=transform)
    img, img_body, img_prev, img_body_prev, annotation = dataset[20]
    img_body = img_body + 0.5
    img = img + 0.5
    img_prev = img_prev + 0.5
    img_body_prev = img_body_prev + 0.5
    out = np.concatenate((img[1], img_body[1], img_prev[1], img_body_prev[1]), axis=1)
    plt.imshow(out, cmap='gray')
    plt.show()

if __name__ == '__main__':
    main()
