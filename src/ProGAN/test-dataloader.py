import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from dataloader_ProGAN import ChunkDataset  # Ensure this file is in the same directory or update the import accordingly
import matplotlib.pyplot as plt

HR_DIR = "/home/aravos/Code/02-Upscale-Project/Image-Upscaler/dataset/Processed/HR_Chunks"
LR_DIR = "/home/aravos/Code/02-Upscale-Project/Image-Upscaler/dataset/Processed/\LR_Chunks"

BATCH_SIZE = 2
dataset = ChunkDataset(hr_dir=HR_DIR, lr_dir=LR_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Total samples in dataset: {len(dataset)}")


for batch in dataloader:
    lr_batch, hr_batch = batch
    
    # Convert each tensor in the batch to a PIL image
    lr_pil_list = [to_pil_image(lr) for lr in lr_batch]
    hr_pil_list = [to_pil_image(hr) for hr in hr_batch]
    
    fig, axs = plt.subplots(BATCH_SIZE, 2, figsize=(8, 4 * BATCH_SIZE))

    if BATCH_SIZE == 1:
        axs = [axs]
    
    for i in range(BATCH_SIZE):
        axs[i][0].imshow(lr_pil_list[i])
        axs[i][0].set_title("LR Image")
        axs[i][0].axis("off")
        
        axs[i][1].imshow(hr_pil_list[i])
        axs[i][1].set_title("HR Image")
        axs[i][1].axis("off")
    
    plt.tight_layout()
    plt.show()
    
    break