import os
import glob
import cv2
import numpy as np

# Paths
HR_DIR = "/home/aravos/Code/02-Upscale-Project/Image-Upscaler/dataset/DIV2K/DIV2K_train_HR"
LR_256_DIR = "/home/aravos/Code/02-Upscale-Project/Image-Upscaler/dataset/DIV2K/DIV2K_train_LR_blurred_2"

# Output directories
HR_OUT = "/home/aravos/Code/02-Upscale-Project/Image-Upscaler/dataset/Processed/HR_Chunks"
LR_256_OUT = "/home/aravos/Code/02-Upscale-Project/Image-Upscaler/dataset/Processed/LR_256"

os.makedirs(HR_OUT, exist_ok=True)
os.makedirs(LR_256_OUT, exist_ok=True)

# Desired chunk sizes
HR_CHUNK_SIZE = 256
LR_256_CHUNK_SIZE = 128

def reflect_pad(image, patch_size):
    """Apply reflect padding so that image dimensions become multiples of patch_size."""
    h, w, _ = image.shape
    pad_h = (patch_size - (h % patch_size)) % patch_size
    pad_w = (patch_size - (w % patch_size)) % patch_size
    padded = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    return padded

# Get sorted list of PNG files from HR and LR_256 directories
hr_images = sorted(glob.glob(os.path.join(HR_DIR, "*.png")))
lr256_images = sorted(glob.glob(os.path.join(LR_256_DIR, "*.png")))

print(f"Found {len(hr_images)} HR images and {len(lr256_images)} LR images.")

assert len(hr_images) == len(lr256_images), "Mismatch between HR and LR file counts!"

total_chunks = 0

for hr_path, lr256_path in zip(hr_images, lr256_images):
    base_name = os.path.splitext(os.path.basename(hr_path))[0]
    print(f"\nProcessing {base_name} ...")
    
    # Read images
    hr_img = cv2.imread(hr_path, cv2.IMREAD_COLOR)
    lr256_img = cv2.imread(lr256_path, cv2.IMREAD_COLOR)
    
    if hr_img is None or lr256_img is None:
        print(f"Warning: Could not load one of the images for {base_name}. Skipping.")
        continue

    # Optional check for matching filenames
    if os.path.basename(hr_path) != os.path.basename(lr256_path):
        print(f"Error: Filename mismatch for {base_name}. Skipping.")
        continue

    # Apply reflect padding
    hr_img = reflect_pad(hr_img, HR_CHUNK_SIZE)
    lr256_img = reflect_pad(lr256_img, LR_256_CHUNK_SIZE)

    # Get padded sizes
    h_hr, w_hr, _ = hr_img.shape
    h_lr256, w_lr256, _ = lr256_img.shape

    hr_chunks = []
    lr256_chunks = []
    
    # Extract patches from HR image and matching patches from LR image
    for y in range(0, h_hr, HR_CHUNK_SIZE):
        for x in range(0, w_hr, HR_CHUNK_SIZE):
            hr_chunk = hr_img[y:y+HR_CHUNK_SIZE, x:x+HR_CHUNK_SIZE]
            # Assuming the LR image is half the resolution of the HR image, scale coordinates by 1/2
            lr256_y = y // 2
            lr256_x = x // 2
            lr256_chunk = lr256_img[lr256_y:lr256_y+LR_256_CHUNK_SIZE, lr256_x:lr256_x+LR_256_CHUNK_SIZE]
            
            hr_chunks.append(hr_chunk)
            lr256_chunks.append(lr256_chunk)

    print(f"Extracted {len(hr_chunks)} chunks from {base_name}")
    total_chunks += len(hr_chunks)
    
    # Save each chunk pair with deterministic filenames (originalname_chunk{index}.png)
    for idx, (hr_chunk, lr256_chunk) in enumerate(zip(hr_chunks, lr256_chunks)):
        hr_save_path = os.path.join(HR_OUT, f"{base_name}_chunk{idx}.png")
        lr256_save_path = os.path.join(LR_256_OUT, f"{base_name}_chunk{idx}.png")
        cv2.imwrite(hr_save_path, hr_chunk)
        cv2.imwrite(lr256_save_path, lr256_chunk)
        print(f"Saved chunk: {base_name}_chunk{idx}.png")

print(f"\nProcessing complete. Total chunks extracted: {total_chunks}")
