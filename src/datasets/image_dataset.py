# Copyright (c) NeoCybernetica, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

from logging import getLogger

import torch
import torchvision
from datetime import datetime
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler, Sampler


_GLOBAL_SEED = 0
logger = getLogger()


class ImageFolder(torchvision.datasets.ImageFolder):

    def __init__(
        self,
        root,
        image_folder="imagenet_full_size/061417/",
        transform=None,
        train=True,
    ):
        """
        ImageFolder
        :param root: root network directory for ImageFolder data
        :param image_folder: path to images inside root network directory
        :param train: whether to load train data (or validation)
        """

        suffix = "train/" if train else "val/"
        data_path = os.path.join(root, image_folder, suffix)
        logger.info(f"data-path {data_path}")
        super(ImageFolder, self).__init__(root=data_path, transform=transform)
        logger.info("Initialized ImageFolder")


def make_imagedataset(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    training=True,
    copy_data=False,
    drop_last=True,
    persistent_workers=False,
    subset_file=None,
):
    dataset = ImageFolder(
        root=root_path, image_folder=image_folder, transform=transform, train=training
    )
    logger.info("ImageFolder dataset created")
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset, num_replicas=world_size, rank=rank
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
    logger.info("ImageFolder unsupervised data loader created")

    return dataset, data_loader, dist_sampler


import os
import pandas as pd
import torch
from PIL import Image


class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_paths,  # List of directories containing timestamped image folders
        transform=None,
        shared_transform=None,
    ):
        self.data_paths = data_paths
        self.transform = transform
        self.shared_transform = shared_transform

        # Load Image Paths and Labels
        self.samples = []
        for data_path in self.data_paths:
            timestamped_folders = [
                f
                for f in os.listdir(data_path)
                if os.path.isdir(os.path.join(data_path, f))
            ]
            for folder in timestamped_folders:
                folder_path = os.path.join(data_path, folder)
                image_files = sorted(
                    os.listdir(folder_path)
                )  # Sort for sequential order
                self.samples.extend(
                    [(folder_path, image_file) for image_file in image_files]
                )  # Store (folder_path, image_filename) tuples

        if len(self.samples) == 0:
            raise RuntimeError(f"Found 0 image files in the data_paths: {data_paths}")

    def __getitem__(self, index):
        folder_path, image_filename = self.samples[index]
        image_path = os.path.join(folder_path, image_filename)

        # Load Image
        image = Image.open(image_path)

        # Apply Transforms
        if self.shared_transform is not None:
            image = self.shared_transform(image)
        if self.transform is not None:
            image = self.transform(image)

        # Load Action Data
        action_filepath = os.path.join(folder_path, "action_data.csv")
        action_df = pd.read_csv(action_filepath)

        # Extract Timestamp from Image Filename
        image_timestamp = self.extract_timestamp_from_filename(image_filename)
        action_label = self.get_action_label_for_timestamp(action_df, image_timestamp)

        return (
            image,
            action_label,
        )  # Return the image and its corresponding action label

    def extract_timestamp_from_filename(self, filename):
        timestamp_str = os.path.splitext(filename)[0].split("_")[
            0
        ]  # Get '20240516_175159'
        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        return timestamp

    def get_action_labels_for_clip(self, action_df, image_timestamps):
        action_labels = []
        for timestamp in image_timestamps:
            # Find closest action timestamps before and after the image timestamp
            before_idx = action_df["timestamp"].searchsorted(timestamp) - 1
            after_idx = before_idx + 1

            # Handle edge cases (first or last image)
            before_idx = max(0, before_idx)
            after_idx = min(len(action_df) - 1, after_idx)

            # Get action labels and timestamps
            action_before = action_df.iloc[before_idx]["action_name"]
            action_after = action_df.iloc[after_idx]["action_name"]
            timestamp_before = action_df.iloc[before_idx]["timestamp"]
            timestamp_after = action_df.iloc[after_idx]["timestamp"]

            # Linear Interpolation (if needed, can be removed for simple nearest neighbor)
            weight_after = (timestamp - timestamp_before) / (
                timestamp_after - timestamp_before
            )
            if weight_after < 0.5:  # Closer to the previous action
                action_label = action_before
            else:  # Closer to the next action
                action_label = action_after

            action_labels.append(action_label)

        return action_labels



class SequentialImageSampler(Sampler):
    def __init__(self, image_dataset, num_replicas=None, rank=None):
        super().__init__(image_dataset)
        self.image_dataset = image_dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.grouped_images = self.group_images_by_folder()

    def group_images_by_folder(self):
        # Group image paths by folder, sorting by timestamp within each folder
        grouped_images = {}
        for folder_path, image_filename in self.image_dataset.samples:
            grouped_images.setdefault(folder_path, []).append(image_filename)
        for folder_path in grouped_images:
            grouped_images[folder_path] = sorted(grouped_images[folder_path], key=self.image_dataset.extract_timestamp_from_filename)
        return grouped_images

    def __iter__(self):
        # Determine which folders this worker should handle
        worker_folders = [
            folder
            for i, folder in enumerate(sorted(self.grouped_images.keys()))
            if i % self.num_replicas == self.rank
        ]

        # Yield image indices in sequential order for each assigned folder
        for folder_path in worker_folders:
            for image_filename in self.grouped_images[folder_path]:
                yield self.image_dataset.samples.index((folder_path, image_filename))

    def __len__(self):
        # Total number of samples across all workers
        total_samples = sum(len(images) for images in self.grouped_images.values())
        # Number of samples for this worker
        num_samples_per_worker = total_samples // self.num_replicas
        # Add any remaining samples to the last worker
        if self.rank == self.num_replicas - 1:
            num_samples_per_worker += total_samples % self.num_replicas
        return num_samples_per_worker


def make_egovehicle_imagedataset(
    data_paths,
    batch_size,
    transform=None,
    shared_transform=None,
    rank=0,
    world_size=1,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
):
    dataset = ImageDataset(
        data_paths=data_paths,
        transform=transform,
        shared_transform=shared_transform,
    )

    logger.info("ImageDataset created")

    # Ensure that each worker gets a subset of folders while maintaining sequential order
    sampler = SequentialImageSampler(dataset, num_replicas=world_size, rank=rank)
    
    # Wrap the sampler with DistributedSampler for shuffling at the folder level
    dist_sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    # DataLoader should use both samplers
    data_loader = DataLoader(
        dataset,
        batch_sampler=dist_sampler, # Using batch_sampler instead of sampler
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )

    logger.info("ImageDataset data loader created")

    return dataset, data_loader, dist_sampler  
