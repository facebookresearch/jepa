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
        csv_file_path,  # List of directories containing timestamped image folders
        transform=None,
        shared_transform=None,
    ):        
        self.transform = transform
        self.shared_transform = shared_transform

        # Load Image Paths and Labels from CSV
        df = pd.read_csv(csv_file_path, header=None, delimiter=" ")
        self.samples = []  # List to store (image_path, action_label) tuples

        for _, row in df.iterrows():
            folder_path = row[0]
            action_filepath = os.path.join(folder_path, "action_data.csv")
            if os.path.exists(action_filepath):
                try:
                    action_df = pd.read_csv(action_filepath)
                except pd.errors.EmptyDataError:
                    logger.warning(
                        f"Skipping folder '{folder_path}' due to empty action_data.csv"
                    )
                    continue
                self.samples.extend(list(action_df[["image_path", "maneuver"]].values))  # Store image paths and action labels

        if not self.samples:
            raise RuntimeError(
                f"Found 0 image files with corresponding action data in the CSV: {csv_file_path}"
            )

    def __getitem__(self, index):
        image_path, action_label = self.samples[index]

        # Load Image
        image = Image.open(image_path)

        # Apply Transforms
        if self.shared_transform is not None:
            image = self.shared_transform(image)
        if self.transform is not None:
            image = self.transform(image)

        return image, action_label  # Return the image and its corresponding action label



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
    csv_file_path,
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
        csv_file_path=csv_file_path,
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
