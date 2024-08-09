# Inspired form https://github.com/facebookresearch/jepa/issues/66

# from video_classification_frozen.utils import make_dataloader
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from app.vjepa.utils import (
    init_video_model,
)
from evals.video_classification_frozen.utils import make_transforms
from src.datasets.data_manager import init_data


def make_dataloader(
    root_path,
    batch_size,
    world_size,
    rank,
    dataset_type="VideoDataset",
    resolution=224,
    frames_per_clip=16,
    frame_step=4,
    num_segments=8,
    eval_duration=None,
    num_views_per_segment=1,
    allow_segment_overlap=True,
    training=False,
    num_workers=12,
    subset_file=None,
):
    # Make Video Transforms
    transform = make_transforms(
        training=training,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(0.75, 4 / 3),
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
        allow_clip_overlap=allow_segment_overlap,
        num_workers=num_workers,
        copy_data=False,
        drop_last=False,
        subset_file=subset_file,
    )
    return data_loader


def get_robust_pca(features: torch.Tensor, m: float = 2, remove_first_component=False):
    # features: (N, C)
    # m: a hyperparam controlling how many std dev outside for outliers
    assert len(features.shape) == 2, "features should be (N, C)"
    reduction_mat = torch.pca_lowrank(features, q=3, niter=20)[2]
    colors = features @ reduction_mat
    if remove_first_component:
        colors_min = colors.min(dim=0).values
        colors_max = colors.max(dim=0).values
        tmp_colors = (colors - colors_min) / (colors_max - colors_min)
        fg_mask = tmp_colors[..., 0] < 0.2
        reduction_mat = torch.pca_lowrank(features[fg_mask], q=3, niter=20)[2]
        colors = features @ reduction_mat
    else:
        fg_mask = torch.ones_like(colors[:, 0]).bool()
    d = torch.abs(colors[fg_mask] - torch.median(colors[fg_mask], dim=0).values)
    mdev = torch.median(d, dim=0).values
    s = d / mdev
    try:
        rins = colors[fg_mask][s[:, 0] < m, 0]
        gins = colors[fg_mask][s[:, 1] < m, 1]
        bins = colors[fg_mask][s[:, 2] < m, 2]
        rgb_min = torch.tensor([rins.min(), gins.min(), bins.min()])
        rgb_max = torch.tensor([rins.max(), gins.max(), bins.max()])
    except:
        rins = colors
        gins = colors
        bins = colors
        rgb_min = torch.tensor([rins.min(), gins.min(), bins.min()])
        rgb_max = torch.tensor([rins.max(), gins.max(), bins.max()])

    return reduction_mat, rgb_min.to(reduction_mat), rgb_max.to(reduction_mat)


def get_pca_map_whole_volume(
    feature_map: torch.Tensor,
    img_size,
    interpolation="bicubic",
    return_pca_stats=False,
    pca_stats=None,
    remove_first_component=False,
):
    """
    feature_map: (num_frames, h, w, C) is the feature map of a single image.
    """
    # print(feature_map.shape)
    if feature_map.shape[0] != 1:
        # make it (1, num_frames, h, w, C)
        feature_map = feature_map[None]
    if pca_stats is None:
        reduct_mat, color_min, color_max = get_robust_pca(
            feature_map.reshape(-1, feature_map.shape[-1]),
            remove_first_component=remove_first_component,
        )
    else:
        reduct_mat, color_min, color_max = pca_stats
    pca_color = feature_map @ reduct_mat
    pca_color = (pca_color - color_min) / (color_max - color_min)
    pca_color = pca_color.clamp(0, 1)
    resized_pca_colors = []
    for i in range(pca_color.shape[1]):
        resized_pca_color = F.interpolate(
            pca_color[:, i, :, :, :].permute(0, 3, 1, 2),
            size=img_size,
            mode=interpolation,
        ).permute(0, 2, 3, 1)
        resized_pca_colors.append(resized_pca_color.cpu().numpy().squeeze(0))
    pca_color = np.stack(resized_pca_colors, axis=0)
    if return_pca_stats:
        return pca_color, (reduct_mat, color_min, color_max)
    return pca_color


with open("configs/pretrain/vitl16.yaml", "r") as y_file:
    args = yaml.load(y_file, Loader=yaml.FullLoader)

# -- set device
device = torch.device("cpu")

# -- META
cfgs_meta = args.get("meta")
use_sdpa = cfgs_meta.get("use_sdpa", False)

# -- MODEL
cfgs_model = args.get("model")
model_name = cfgs_model.get("model_name")
pred_depth = cfgs_model.get("pred_depth")
pred_embed_dim = cfgs_model.get("pred_embed_dim")
uniform_power = cfgs_model.get("uniform_power", True)
use_mask_tokens = cfgs_model.get("use_mask_tokens", True)
zero_init_mask_tokens = cfgs_model.get("zero_init_mask_tokens", True)

# -- MASK
cfgs_mask = args.get("mask")

# -- DATA
cfgs_data = args.get("data")
dataset_type = cfgs_data.get("dataset_type", "videodataset")
mask_type = cfgs_data.get("mask_type", "multiblock3d")
dataset_paths = cfgs_data.get("datasets", [])
datasets_weights = cfgs_data.get("datasets_weights", None)
if datasets_weights is not None:
    assert len(datasets_weights) == len(
        dataset_paths
    ), "Must have one sampling weight specified for each dataset"
batch_size = 1
num_clips = cfgs_data.get("num_clips")
num_frames = cfgs_data.get("num_frames")
tubelet_size = cfgs_data.get("tubelet_size")
sampling_rate = cfgs_data.get("sampling_rate")
duration = cfgs_data.get("clip_duration", None)
crop_size = cfgs_data.get("crop_size", 224)
patch_size = cfgs_data.get("patch_size")
pin_mem = cfgs_data.get("pin_mem", False)
num_workers = cfgs_data.get("num_workers", 1)
filter_short_videos = cfgs_data.get("filter_short_videos", False)
decode_one_clip = cfgs_data.get("decode_one_clip", True)
log_resource_util_data = cfgs_data.get("log_resource_utilization", False)

eval_num_segments = 1
attend_across_segments = False
world_size = 1
rank = 0


train_data_path = [
    "/home/your-user/ML-datasets/video_datasets/unlabeled_videos.csv"
]

data_loader = make_dataloader(
    dataset_type=dataset_type,
    root_path=train_data_path,
    resolution=crop_size,
    frames_per_clip=num_frames,
    frame_step=sampling_rate,
    eval_duration=duration,
    num_segments=eval_num_segments if attend_across_segments else 1,
    num_views_per_segment=1,
    allow_segment_overlap=True,
    batch_size=batch_size,
    world_size=world_size,
    rank=rank,
    training=False,
)

for data in data_loader:
    clips, masks_enc, masks_pred = data
    break


clips[0][0].shape
min_val = clips[0][0][0].permute(1, 2, 3, 0)[0].numpy().min()
max_val = clips[0][0][0].permute(1, 2, 3, 0)[0].numpy().max()
img = (clips[0][0][0].permute(1, 2, 3, 0)[0].numpy() - min_val) / (max_val - min_val)
print(img.min(), img.max())
plt.imshow(img)


encoder, predictor = init_video_model(
    uniform_power=uniform_power,
    use_mask_tokens=use_mask_tokens,
    num_mask_tokens=len(cfgs_mask),
    zero_init_mask_tokens=zero_init_mask_tokens,
    device=device,
    patch_size=patch_size,
    num_frames=num_frames,
    tubelet_size=tubelet_size,
    model_name=model_name,
    crop_size=crop_size,
    pred_depth=pred_depth,
    pred_embed_dim=pred_embed_dim,
    use_sdpa=use_sdpa,
)


checkpoint = torch.load(
    "/home/your-user/ML-logging/nano-jepa-logging/jepa-latest.pth.tar",
    map_location="cpu",
)
# checkpoint = torch.load('vith16.pth.tar', map_location='cpu')
print(checkpoint.keys())
new_encoder_state_dict = {}
pretrained_dict = checkpoint["target_encoder"]
pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
# pretrained_dict = {k.replace('backbone.', ''): v for k, v in pretrained_dict.items()}
encoder.load_state_dict(pretrained_dict)


x = encoder(clips[0][0].to(device))

output_of_vjepa = x
print("output_of_vjepa:", x.shape)
print("input shape:", clips[0][0].shape)
B, N, D = x.shape
B, C, FRAMES, H, W = clips[0][0].shape
# Patch = (tubelet_size, patch_size, patch_size)
N_FRAMES = FRAMES // tubelet_size
N_H = H // patch_size
N_W = W // patch_size

print(
    f"Thus, N feature ({output_of_vjepa.shape[1]}) is calcuated from",
    H * W * FRAMES / tubelet_size / patch_size / patch_size,
)


image_size = (crop_size, crop_size)
volume_pca_map = get_pca_map_whole_volume(
    x.detach().reshape(batch_size, N_FRAMES, N_H, N_W, D),
    image_size,
    interpolation="bilinear",
    remove_first_component=False,
)
print(volume_pca_map.shape)


axes, fig = plt.subplots(2, 8, figsize=(40, 20))
for i in range(8):
    fig[0, i].imshow(volume_pca_map[i])

for clip_index in range(8):
    image = clips[0][0][0].permute(1, 2, 3, 0)[clip_index].numpy()
    image = (image - image.min()) / (image.max() - image.min())
    fig[1, clip_index].imshow(image)

plt.show()
