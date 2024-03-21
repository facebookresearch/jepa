# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger


_GLOBAL_SEED = 0
logger = getLogger()


def init_data(
    batch_size,
    transform=None,
    shared_transform=None,
    data='ImageNet',
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
    tokenize_txt=True,
    subset_file=None,
    clip_len=8,
    frame_sample_rate=2,
    duration=None,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_videos=False,
    filter_long_videos=int(1e9),
    decode_one_clip=True,
    datasets_weights=None,
    persistent_workers=False,
    repeat_wds=False,
    ipe=300,
    log_dir=None,
):

    if (data.lower() == 'imagenet') \
            or (data.lower() == 'inat21') \
            or (data.lower() == 'places205'):
        from src.datasets.image_dataset import make_imagedataset
        dataset, data_loader, dist_sampler = make_imagedataset(
            transform=transform,
            batch_size=batch_size,
            collator=collator,
            pin_mem=pin_mem,
            training=training,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            persistent_workers=persistent_workers,
            copy_data=copy_data,
            drop_last=drop_last,
            subset_file=subset_file)

    elif data.lower() == 'videodataset':
        from src.datasets.video_dataset import make_videodataset
        dataset, data_loader, dist_sampler = make_videodataset(
            data_paths=root_path,
            batch_size=batch_size,
            frames_per_clip=clip_len,
            frame_step=frame_sample_rate,
            duration=duration,
            num_clips=num_clips,
            random_clip_sampling=random_clip_sampling,
            allow_clip_overlap=allow_clip_overlap,
            filter_short_videos=filter_short_videos,
            filter_long_videos=filter_long_videos,
            shared_transform=shared_transform,
            transform=transform,
            datasets_weights=datasets_weights,
            collator=collator,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            drop_last=drop_last,
            log_dir=log_dir)

    return (data_loader, dist_sampler)
