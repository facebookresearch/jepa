import matplotlib.pyplot as plt
import cv2

from football_frames_dataset import FramesDataset
from torch.utils.data import DataLoader 


if __name__== "__main__":
    val_dataset = FramesDataset('./spotting-ball-2024')
    raw_frames, label = val_dataset[0]
    print(raw_frames.shape, label)
    val_loader = DataLoader(val_dataset, batch_size=8)
    batch = next(iter(val_loader))
    print(batch[0].shape, batch[1].shape, len(batch))
    # Plot the images using matplotlib
    # fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    # axes = axes.ravel()

    # for i in range(8):
    #     axes[i].imshow(raw_frames[i])
    #     axes[i].axis('off')

    # plt.tight_layout()
    # plt.show()