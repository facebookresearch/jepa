import json 
import os
import cv2
import numpy as np

from src.datasets.utils.video.functional import *
from torch.utils.data import Dataset


actions = "Pass, Drive, Header, High Pass, Out, Cross, Throw In, Shot, Ball Player Block, Player Successful Tackle, Free Kick, Goal".split(", ")
action_to_id = {action.upper(): action_id for action_id, action in enumerate(actions)}

class FramesDataset(Dataset):
    def __init__(self, root_dir, frame_window_size=64, frame_step=8, frame_dim=(224, 224), n_channels=3, shuffle=True, transforms=None):
        self.frame_dim = frame_dim
        self.frame_window_size = frame_window_size
        self.frame_step = frame_step
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.transforms = transforms
        self.frames_info = []

        # Load annotations and create a global index for frames
        for root, dirs, files in os.walk(root_dir):
            for match_folder in dirs:
                match_folder_path = os.path.join(root, match_folder)
                video_files = [f for f in os.listdir(match_folder_path) if f.endswith('.mp4')]
                if len(video_files) != 2:
                    continue
                labels_file = 'Labels-ball.json'
                if labels_file not in os.listdir(match_folder_path):
                    continue
                with open(match_folder_path + '/' + labels_file, 'r') as f:
                    data = json.load(f)
                annotations = data['annotations']
                for ann in annotations:
                    video_path = os.path.join(root_dir, data['UrlLocal'], '720p.mp4'.format(ann['gameTime'][0]))
                    frame_count = get_frame_count(video_path)
                    start_frame = int(ann['position'])
                    if int(start_frame/40) + self.frame_window_size < frame_count:
                        self.frames_info.append((video_path, int(start_frame/40), action_to_id[ann['label']]))
        if shuffle:
            np.random.shuffle(self.frames_info)

    def __len__(self):
        return int(np.floor(len(self.frames_info)))
                   
    def __getitem__(self, index):
        action_frames_info = self.frames_info[index] 
        raw_frames, target = self.data_generation(action_frames_info)
        raw_frames = self.transforms(raw_frames)
        return np.array(raw_frames), target


    def data_generation(self, action_frames_info):
        video_path, start_frame, y = action_frames_info
        cap = cv2.VideoCapture(video_path)
        frames = []
        for frame_index in range(0, self.frame_window_size, self.frame_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + frame_index)
            ret, frame = cap.read()
            frame = cv2.resize(frame, self.frame_dim)
            frames.append(frame.astype(np.float32) / 255.)
            if not ret:
                break  # Reached the end of the video
        cap.release()

        return np.array(frames), y