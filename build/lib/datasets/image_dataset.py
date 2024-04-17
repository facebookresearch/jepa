# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

from logging import getLogger

import torch
import torchvision

_GLOBAL_SEED = 0
logger = getLogger()


class ImageFolder(torchvision.datasets.ImageFolder):

    def __init__(
        self,
        root,
        image_folder='imagenet_full_size/061417/',
        transform=None,
        train=True,
    ):
        """
        ImageFolder
        :param root: root network directory for ImageFolder data
        :param image_folder: path to images inside root network directory
        :param train: whether to load train data (or validation)
        """

        suffix = 'train/' if train else 'val/'
        data_path = os.path.join(root, image_folder, suffix)
        logger.info(f'data-path {data_path}')
        super(ImageFolder, self).__init__(root=data_path, transform=transform)
        logger.info('Initialized ImageFolder')


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
    subset_file=None
):
    dataset = ImageFolder(
        root=root_path,
        image_folder=image_folder,
        transform=transform,
        train=training)
    logger.info('ImageFolder dataset created')
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=persistent_workers)
    logger.info('ImageFolder unsupervised data loader created')

    return dataset, data_loader, dist_sampler
