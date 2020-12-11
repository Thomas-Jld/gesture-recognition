import argparse

import sys
import os
import cv2
import numpy as np
import torch
import json
import math
import mediapipe as mp


sys.path.append(os.path.join(os.path.dirname(__file__), "pose-estimation"))
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose
from val import normalize, pad_width



class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img





def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256):
    """
    Scale the image and estimate the probabilty of each point being a keypoint (heatmap)
    """
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


def get_data(net, hands, image_provider, name, send = False, cpu = False):
    height_size = 256
    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    delay = 33

    data = {'name': name, 'frames': []}

    """
    Estimate the pose and find the person in the middle
    """

    for img in image_provider:
        img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
        ori = img.copy()

        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        current_poses = []

        distMin = 310
        midPose = None

        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])

            dist = abs(pose.keypoints[0][0] - img.shape[0]/2)
            if dist < distMin:
                distMin = dist
                midPose = pose
            current_poses.append(pose)
        
        blank_image = np.zeros((img.shape[0],img.shape[1],3), np.uint8)

        if midPose != None:
            midPose.draw(blank_image)

        ori.flags.writeable = False

        results = hands.process(ori)

        if results.multi_hand_landmarks:
            for f_hands in results.multi_hand_landmarks:
                for landmark in f_hands.landmark:
                    cv2.circle(blank_image, (int(landmark.x*blank_image.shape[0]), int(landmark.y*blank_image.shape[1])), 3, [255,255,255], -1)
    
        # cv2.imshow('Trainning frame', blank_image)
        data['frames'].append(blank_image)
        # key = cv2.waitKey(delay)
        # if key == 27:  # esc
        #     return
        # elif key == 112:  # 'p'
        #     if delay == 33:
        #         delay = 0
        #     else:
        #         delay = 33
                    
    return data


def init(cpu = False):
    net = PoseEstimationWithMobileNet()

    checkpoint_path = "pose-estimation/checkpoint_iter_370000.pth"
    checkpoint = torch.load(checkpoint_path, map_location='cpu') #load the existing model
    load_state(net, checkpoint)

    net = net.eval()
    if not cpu:
        net = net.cuda()
    
    hands = mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.2)

    return net, hands


def generate_data(path: str) -> dict:
    frames = VideoReader(path)
    net, hands = init()
    name = path.split('.')[0]

    return get_data(net, hands, frames, name)

if __name__ == "__main__":
    generate_data("0")