# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import pathlib
import warnings

from logging import getLogger

import numpy as np
import pandas as pd

from decord import VideoReader, cpu

import torch

from src.datasets.utils.weighted_sampler import DistributedWeightedSampler

_GLOBAL_SEED = 0
logger = getLogger()


def make_sdfdataset(
    data_paths,
    batch_size,
    transform=None,
    rank=0,
    world_size=1,
    datasets_weights=None,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    log_dir=None,
):
    dataset = SdfDataset(
        data_paths=data_paths,
        datasets_weights=datasets_weights,
        transform=transform)

    logger.info('SdfDataset dataset created')
    if datasets_weights is not None:
        dist_sampler = DistributedWeightedSampler(
            dataset.sample_weights,
            num_replicas=world_size,
            rank=rank,
            shuffle=True)
    else:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=num_workers > 0)
    logger.info('SdfDataset unsupervised data loader created')

    return dataset, data_loader, dist_sampler


class SdfDataset(torch.utils.data.Dataset):
    """ Sdf classification dataset. """

    def __init__(
        self,
        data_paths,
        datasets_weights=None,
        transform=None,
    ):
        self.data_paths = data_paths
        self.datasets_weights = datasets_weights
        self.transform = transform

        # Load video paths and labels
        samples, labels = [], []
        self.num_samples_per_dataset = []
        for data_path in self.data_paths:

            if data_path[-4:] == '.csv':
                data = pd.read_csv(data_path, header=None, delimiter=" ")
                samples += list(data.values[:, 0])
                labels += list(data.values[:, 1])
                num_samples = len(data)
                self.num_samples_per_dataset.append(num_samples)
                print(f'Loaded {num_samples} samples from {data_path}')
            else:
                raise ValueError(f'Invalid data path (not .csv) {data_path=}')

        self.samples = samples
        self.labels = labels

    def __getitem__(self, index):
        sample = self.samples[index]
        
        sdf_obj = torch.from_numpy(np.load(sample)).unsqueeze(-1) # [T H W 1]

        # Label/annotations
        label = self.labels[index]

        # apply data augmentations
        if self.transform is not None:
            sdf_obj = self.transform(sdf_obj)

        return [sdf_obj], label

    def __len__(self):
        return len(self.samples)
