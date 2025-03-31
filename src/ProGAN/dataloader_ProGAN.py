import os
import cv2
import torch
from torch.utils.data import Dataset  # Using Dataset for proper indexing
from torchvision.transforms.functional import pad, to_tensor
import torchvision.transforms as T

class ChunkDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, channels=3):
        super().__init__()
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.channels = channels

        self.image_files = sorted([f for f in os.listdir(hr_dir) if f.endswith(".png")])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        filename = self.image_files[index]
        lr_path = os.path.join(self.lr_dir, filename)
        hr_path = os.path.join(self.hr_dir, filename)

        # Load images with cv2
        lr_bgr = cv2.imread(lr_path)
        hr_bgr = cv2.imread(hr_path)
        if lr_bgr is None or hr_bgr is None:
            raise ValueError(f"Image {filename} could not be loaded.")

        # Convert from BGR to RGB then to tensor
        lr_rgb = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2RGB)
        hr_rgb = cv2.cvtColor(hr_bgr, cv2.COLOR_BGR2RGB)
        lr_image = to_tensor(lr_rgb)  # shape: (3, H_lr, W_lr)
        hr_image = to_tensor(hr_rgb)  # shape: (3, H_hr, W_hr)

        return lr_image, hr_image
