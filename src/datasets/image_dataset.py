# Copyright (c) NeoCybernetica, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import PIL
from collections import defaultdict


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
    def __init__(self, data_dir, transform=None, shared_transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.shared_transform = shared_transform

        # Load data from drive folders
        self.samples = []
        self.drive_data = {}

        try:
            drive_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
            for drive_folder in drive_folders:
                drive_path = os.path.join(data_dir, drive_folder)
                csv_file = os.path.join(drive_path, "drive_data.csv")

                if not os.path.exists(csv_file):
                    logger.warning(f"Skipping drive folder '{drive_folder}' due to missing drive_data.csv file.")
                    continue

                try:
                    drive_df = pd.read_csv(csv_file)
                    self.drive_data[drive_folder] = drive_df
                    drive_samples = [(os.path.join(drive_path, row['path_to_image']), row['maneuverID']) for _, row in drive_df.iterrows()]
                    self.samples.extend(drive_samples)
                except (pd.errors.EmptyDataError, KeyError) as e:
                    logger.warning(f"Skipping drive folder '{drive_folder}' due to error: {str(e)}")

            if len(self.samples) == 0:
                raise RuntimeError(f"No valid drive folders found in the dataset directory: {data_dir}")

        except OSError as e:
            raise RuntimeError(f"Error accessing dataset directory: {data_dir}. Exception: {str(e)}")
        
    def __getitem__(self, index):
        try:
            image_path, maneuver_id = self.samples[index]

            # Load image
            try:
                image = Image.open(image_path).convert("RGB")  # Convert to RGB here
            except (IOError, PIL.UnidentifiedImageError) as e:
                logger.warning(f"Error loading image: {image_path}. Exception: {str(e)}")
                raise e

            # Apply transforms
            if self.shared_transform is not None:
                image = self.shared_transform(image)
            if self.transform is not None:
                image = self.transform(image)

            return image, maneuver_id

        except IndexError as e:
            raise IndexError(f"Index {index} is out of bounds for the dataset.")

    def __len__(self):
        if not self.samples:
            raise RuntimeError("Dataset is empty. No valid samples found.")
        return len(self.samples)

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

def collate_fn(batch):
    images, maneuvers = zip(*batch)
    
    # Stack images into a single tensor
    images = torch.stack(images, dim=0)
    
    # Convert maneuvers to a tensor
    maneuvers = torch.tensor([m for m in maneuvers])
    
    return images, maneuvers

class SequentialDriveSampler(Sampler):
    def __init__(self, image_dataset):
        self.image_dataset = image_dataset
        self.drive_indices = self._get_drive_indices()

    def _get_drive_indices(self):
        drive_indices = defaultdict(list)
        for idx, (image_path, _) in enumerate(self.image_dataset.samples):
            drive_folder = os.path.basename(os.path.dirname(image_path))
            drive_indices[drive_folder].append(idx)
        return drive_indices

    def __iter__(self):
        for drive_folder, indices in self.drive_indices.items():
            yield from indices

    def __len__(self):
        return len(self.image_dataset)
    
def make_egovehicle_imagedataset(
    data_dir,
    batch_size,
    transform=None,
    shared_transform=None,
    mask_collator=None,
    num_workers=10,
    rank=0,
    world_size=1,
    pin_mem=True,
    drop_last=True,
):
    dataset = ImageDataset(
        data_dir=data_dir,
        transform=transform,
        shared_transform=shared_transform,
    )

    logger.info("ImageDataset created")

    # sampler = SequentialDriveSampler(dataset)
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
    
    # data_loader = DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     sampler=dist_sampler,
    #     collate_fn=mask_collator,
    #     num_workers=num_workers,
    #     pin_memory=pin_mem,
    #     drop_last=True,
    # )

    data_loader = DataLoader(
        dataset,
        collate_fn=mask_collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    logger.info("ImageDataset data loader created")

    return dataset, data_loader, dist_sampler
