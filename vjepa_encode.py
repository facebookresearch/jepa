""" Encode videos with VJEPA encoder. """

from __future__ import annotations

import logging
from pathlib import Path

import torch

import src.models.vision_transformer as vit
from src.datasets.data_manager import init_data
from evals.video_classification_frozen.utils import (
    make_transforms,
    ClipAggregation
)

INPUT_DATALOADER: Path = Path("/home/ubuntu/Desktop/lehack/datasets/something_something_v2/validation_dataloader.csv")
OUTPUT_DIR: Path = Path("/home/ubuntu/Desktop/lehack/v_jepa_embedding_inversion_attack/embeddings/validation")

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    # Initialise parameters

    # -- model parameters
    resolution = 224
    device = "cuda"
    pretrained_path = "/home/ubuntu/Desktop/lehack/models/vjepa/vitl16.pth.tar"
    model_name = "vit_large"
    patch_size = 16
    tubelet_size = 2
    pretrain_frames_per_clip = 16
    uniform_power = True
    checkpoint_key = "target_encoder"
    use_SiLU = False
    tight_SiLU = False
    use_sdpa = True

    # -- dataloader parameters
    dataset_type = "VideoDataset"
    data_path = [str(INPUT_DATALOADER)]
    eval_frames_per_clip = 16
    eval_frame_step = 1
    eval_num_segments = 1 
    eval_duration = None
    eval_num_views_per_segment = 1
    batch_size = 16

    # -- optimization parameters
    attend_across_segments = True

    # Initialise model

    # -- pretrained encoder (frozen)
    encoder = init_model(
        crop_size=resolution,
        device=device,
        pretrained=pretrained_path,
        model_name=model_name,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        frames_per_clip=pretrain_frames_per_clip,
        uniform_power=uniform_power,
        checkpoint_key=checkpoint_key,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
        use_sdpa=use_sdpa)

    encoder = ClipAggregation(
        encoder,
        tubelet_size=tubelet_size,
        attend_across_segments=attend_across_segments
    ).to(device)
    encoder.eval()
    # Freeze encoder
    for p in encoder.parameters():
        p.requires_grad = False

    # Initialise dataloader
    dataloader = make_dataloader(
        dataset_type=dataset_type,
        root_path=data_path,
        resolution=resolution,
        frames_per_clip=eval_frames_per_clip,
        frame_step=eval_frame_step,
        num_segments=eval_num_segments,
        eval_duration=eval_duration,
        num_views_per_segment=eval_num_views_per_segment,
        allow_segment_overlap=True,
        batch_size=batch_size,
        training=False)

    logger.info(f"Loaded dataloader from {INPUT_DATALOADER}")

    num_videos: int = 0
    for data in dataloader:
        # Load data and put on GPU
        clips = [[data[0][0][0].to(device, non_blocking=True)]]
        clip_indices = data[2][0].to(device, non_blocking=True)
        video_paths = data[3]
        batch_size = len(video_paths)

        # Forward pass
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                outputs = encoder(clips, clip_indices)

        # embeddings shape: [batch_size, num_tokens, embed_dim]
        embeddings = outputs[0]

        # Save each video embedding as sequence of tokens
        for i in range(batch_size):
            video_embedding: torch.Tensor = embeddings[i:i+1].to(dtype=torch.bfloat16).cpu()
            video_idx = Path(video_paths[i]).stem
            torch.save(video_embedding, OUTPUT_DIR / f"{video_idx}.pt")
            num_videos += 1

        del clips
        del outputs
        del embeddings
        torch.cuda.empty_cache()

    logger.info(f"Encoded a total of {num_videos} video(s)")

def init_model(
    device: str,
    pretrained: str,
    model_name: str,
    patch_size: int = 16,
    crop_size: int = 224,
    # Video specific parameters
    frames_per_clip: int = 16,
    tubelet_size: int = 2,
    use_sdpa: bool = False,
    use_SiLU: bool = False,
    tight_SiLU: bool = True,
    uniform_power:bool = False,
    checkpoint_key: str = 'target_encoder'
):
    encoder = vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=frames_per_clip,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
    )
    encoder = load_pretrained(encoder=encoder, pretrained=pretrained, checkpoint_key=checkpoint_key)
    encoder.to(device, dtype=torch.bfloat16)
    return encoder


def load_pretrained(
    encoder: vit.VisionTransformer,
    pretrained: str,
    checkpoint_key: str = 'target_encoder'
):
    logger.info(f'Loading pretrained model from {pretrained}')
    checkpoint = torch.load(pretrained, map_location='cpu')
    try:
        pretrained_dict = checkpoint[checkpoint_key]
    except Exception:
        pretrained_dict = checkpoint['encoder']

    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace('backbone.', ''): v for k, v in pretrained_dict.items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info(f'loaded pretrained model with msg: {msg}')
    logger.info(f'loaded pretrained encoder from epoch: {checkpoint["epoch"]}\n path: {pretrained}')
    del checkpoint
    return encoder


def make_dataloader(
    root_path: list[str],
    batch_size: int,
    world_size: int = 1,
    rank: int = 0,
    dataset_type: str = 'VideoDataset',
    resolution: int = 224,
    frames_per_clip: int = 16,
    frame_step: int = 4,
    num_segments: int = 8,
    eval_duration: float | None = None,
    num_views_per_segment: int = 1,
    allow_segment_overlap: bool = True,
    training: bool = False,
    num_workers: int = 12,
    subset_file: str | None = None
):
    # Make Video Transforms
    transform = make_transforms(
        training=training,
        num_views_per_clip=num_views_per_segment,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(0.75, 4/3),
        random_resize_scale=(0.08, 1.0),
        reprob=0.25,
        auto_augment=True,
        motion_shift=False,
        crop_size=resolution,
    )

    data_loader, _ = init_data(
        data=dataset_type,
        root_path=root_path,
        transform=transform,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        clip_len=frames_per_clip,
        frame_sample_rate=frame_step,
        duration=eval_duration,
        num_clips=num_segments,
        random_clip_sampling=False,
        allow_clip_overlap=allow_segment_overlap,
        num_workers=num_workers,
        copy_data=False,
        drop_last=False,
        subset_file=subset_file)
    return data_loader


if __name__ == "__main__":
    main()
