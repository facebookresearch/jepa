from torch.utils.data import DataLoader
from torchvision import transforms
import yaml
from src.datasets.image_dataset import ImageDataset, SequentialDriveSampler
from src.masks.random_tube import MaskCollatorWithActions as TubeMaskCollator
from src.masks.multiblock3d import MaskCollatorWithActions as MB3DMaskCollator
from src.utils.logging import (
    get_logger,
)

logger = get_logger(__name__)

data_dir = "/home/ncdev/Documents/darwin/data/raw"
filename = "/home/ncdev/Documents/darwin/jepa/configs/pretrain/vith16_384.yaml"

# Load configuration from YAML file
with open(filename, "r") as y_file:
    params = yaml.load(y_file, Loader=yaml.FullLoader)
logger.info("Loaded configuration parameters.")

def test_data_loader(data_dir, batch_size, mask_collator):
    # Define the necessary transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[:3] if x.size(0) > 3 else x),  # Convert to RGB if needed
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageDataset(data_dir=data_dir, transform=transform)
    sampler = SequentialDriveSampler(dataset)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=mask_collator,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    for batch_idx, (images, maneuvers, masks_enc, masks_dec) in enumerate(data_loader):
        print(f"Batch {batch_idx + 1}")
        print(f"Images shape: {images.shape}")
        print(f"Maneuvers shape: {maneuvers.shape}")
        print(f"Encoder Masks shape: {masks_enc[0].shape}")
        print(f"Decoder Masks shape: {masks_dec[0].shape}")
        print("---")
        
        if batch_idx == 4:
            break

cfgs_mask = params.get("mask")

# Test with TubeMaskCollator
print("Testing with TubeMaskCollator")
tube_mask_collator = TubeMaskCollator(
    crop_size=224,
    num_frames=16,
    patch_size=16,
    tubelet_size=2,
    cfgs_mask=cfgs_mask,
)
test_data_loader(data_dir=data_dir, batch_size=32, mask_collator=tube_mask_collator)

# Test with MB3DMaskCollator
print("Testing with MB3DMaskCollator")
mb3d_mask_collator = MB3DMaskCollator(
    crop_size=224,
    num_frames=16,
    patch_size=16,
    tubelet_size=2,
    cfgs_mask=cfgs_mask,
)
test_data_loader(data_dir=data_dir, batch_size=32, mask_collator=mb3d_mask_collator)