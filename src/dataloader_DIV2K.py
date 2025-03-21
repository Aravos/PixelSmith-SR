import os
import cv2
import torch
import random
from torch.utils.data import IterableDataset
from torchvision.transforms.functional import pad, to_tensor
import torchvision.transforms as T

class PatchIterableDataset(IterableDataset):
    def __init__(self, hr_dir, lr_dir, patch_size=512, shuffle=True, channels=3):
        """
        Streams patches from each image, one patch at a time.
        No entire image or entire dataset is stored in memory at once.
        """
        super().__init__()
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.channels = channels
        self.patch_size = patch_size
        self.lr_patch_size = patch_size // 4

        self.image_files = sorted([f for f in os.listdir(hr_dir) if f.endswith(".png")])

        self.normalize = T.Normalize(mean=[0.5] * self.channels, std=[0.5] * self.channels)

        if shuffle:
            random.shuffle(self.image_files)

        # Compute total number of patches across all images
        self.total_patches = sum(
            self._compute_patches_per_image(os.path.join(self.lr_dir, filename))
            for filename in self.image_files
        )

    def _compute_patches_per_image(self, lr_path):
        """Computes the number of patches for a given LR image."""
        lr_bgr = cv2.imread(lr_path)
        if lr_bgr is None:
            return 0  # Skip corrupted images
        
        lr_h, lr_w, _ = lr_bgr.shape  # Get image dimensions
        h_patches = (lr_h + self.lr_patch_size - 1) // self.lr_patch_size  # Ceil division
        w_patches = (lr_w + self.lr_patch_size - 1) // self.lr_patch_size
        return h_patches * w_patches

    def __len__(self):
        """Returns the total number of patches available."""
        return self.total_patches

    def __iter__(self):
        """
        Yields one (lr_patch, hr_patch) at a time.
        """
        for filename in self.image_files:
            lr_path = os.path.join(self.lr_dir, filename)
            hr_path = os.path.join(self.hr_dir, filename)

            # Load images
            lr_bgr = cv2.imread(lr_path)
            hr_bgr = cv2.imread(hr_path)
            if lr_bgr is None or hr_bgr is None:
                continue  # Skip corrupted images

            # Convert to RGB, then to tensor
            lr_rgb = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2RGB)
            hr_rgb = cv2.cvtColor(hr_bgr, cv2.COLOR_BGR2RGB)
            lr_image = to_tensor(lr_rgb)  # shape: (3, H_lr, W_lr)
            hr_image = to_tensor(hr_rgb)  # shape: (3, H_hr, W_hr)

            # Pad to be multiples of patch sizes
            _, lr_h, lr_w = lr_image.shape
            _, hr_h, hr_w = hr_image.shape
            lr_pad_h = (self.lr_patch_size - (lr_h % self.lr_patch_size)) % self.lr_patch_size
            lr_pad_w = (self.lr_patch_size - (lr_w % self.lr_patch_size)) % self.lr_patch_size
            hr_pad_h = (self.patch_size - (hr_h % self.patch_size)) % self.patch_size
            hr_pad_w = (self.patch_size - (hr_w % self.patch_size)) % self.patch_size

            lr_padded = pad(lr_image, (0, 0, lr_pad_w, lr_pad_h), fill=0)
            hr_padded = pad(hr_image, (0, 0, hr_pad_w, hr_pad_h), fill=0)

            # Extract patches, but yield them one at a time
            h_patches = (lr_h + lr_pad_h) // self.lr_patch_size
            w_patches = (lr_w + lr_pad_w) // self.lr_patch_size

            for i in range(h_patches):
                for j in range(w_patches):
                    # LR patch
                    lr_patch = lr_padded[:, 
                                         i*self.lr_patch_size:(i+1)*self.lr_patch_size,
                                         j*self.lr_patch_size:(j+1)*self.lr_patch_size]
                    # HR patch
                    hr_patch = hr_padded[:, 
                                         i*self.patch_size:(i+1)*self.patch_size,
                                         j*self.patch_size:(j+1)*self.patch_size]
                    
                    lr_patch = self.normalize(lr_patch)
                    hr_patch = self.normalize(hr_patch)
                    yield (lr_patch, hr_patch)
