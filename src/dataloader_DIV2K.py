import os
import cv2
import torch
from torch.utils.data import Dataset  # Using Dataset for proper indexing
from torchvision.transforms.functional import pad, to_tensor
import torchvision.transforms as T

class PatchIterableDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, patch_size=512, channels=3):
        super().__init__()
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.channels = channels
        self.patch_size = patch_size
        self.lr_patch_size = patch_size // 4  # e.g., 128 if patch_size is 512

        self.image_files = sorted([f for f in os.listdir(hr_dir) if f.endswith(".png")])
        self.normalize = T.Normalize(mean=[0.5] * self.channels, std=[0.5] * self.channels)

    def __len__(self):
        # Each image yields exactly 12 patches.
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

        # Compute necessary padding to make dimensions multiples of patch sizes
        _, lr_h, lr_w = lr_image.shape
        _, hr_h, hr_w = hr_image.shape
        lr_pad_h = (self.lr_patch_size - (lr_h % self.lr_patch_size)) % self.lr_patch_size
        lr_pad_w = (self.lr_patch_size - (lr_w % self.lr_patch_size)) % self.lr_patch_size
        hr_pad_h = (self.patch_size - (hr_h % self.patch_size)) % self.patch_size
        hr_pad_w = (self.patch_size - (hr_w % self.patch_size)) % self.patch_size

        lr_padded = pad(lr_image, (0, 0, lr_pad_w, lr_pad_h), fill=0)
        hr_padded = pad(hr_image, (0, 0, hr_pad_w, hr_pad_h), fill=0)

        # Determine number of patches (we assume it to be exactly 12)
        h_patches = (lr_h + lr_pad_h) // self.lr_patch_size
        w_patches = (lr_w + lr_pad_w) // self.lr_patch_size

        if h_patches * w_patches != 12:
            raise ValueError(f"Image {filename} does not produce 12 patches (got {h_patches * w_patches}).")

        lr_patches = []
        hr_patches = []
        for i in range(h_patches):
            for j in range(w_patches):
                lr_patch = lr_padded[:,
                                       i*self.lr_patch_size:(i+1)*self.lr_patch_size,
                                       j*self.lr_patch_size:(j+1)*self.lr_patch_size]
                hr_patch = hr_padded[:,
                                       i*self.patch_size:(i+1)*self.patch_size,
                                       j*self.patch_size:(j+1)*self.patch_size]
                lr_patch = self.normalize(lr_patch)
                hr_patch = self.normalize(hr_patch)
                lr_patches.append(lr_patch)
                hr_patches.append(hr_patch)
                
        # Stack patches to form a tensor of shape (12, 3, 128, 128) and (12, 3, 512, 512)
        lr_patches = torch.stack(lr_patches)
        hr_patches = torch.stack(hr_patches)
        return lr_patches, hr_patches
