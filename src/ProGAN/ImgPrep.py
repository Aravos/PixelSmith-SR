import os
import glob
import random
import cv2
import numpy as np

# Paths (update if needed)
HR_DIR = "/home/aravos/Code/02-Upscale-Project/Image-Upscaler/dataset/DIV2K/DIV2K_train_HR"
LR_DIR = "/home/aravos/Code/02-Upscale-Project/Image-Upscaler/dataset/DIV2K/DIV2K_train_LR_blurred"

# Output directories for chunks
HR_OUT = "/home/aravos/Code/02-Upscale-Project/Image-Upscaler/dataset/Processed/HR_Chunks"
LR_OUT = "/home/aravos/Code/02-Upscale-Project/Image-Upscaler/dataset/Processed/\LR_Chunks"

os.makedirs(HR_OUT)
os.makedirs(LR_OUT)

# Desired chunk sizes
HR_CHUNK_SIZE = 512
LR_CHUNK_SIZE = 128

def reflect_pad(image, patch_size):
    """Apply reflect padding so that the image dimensions become multiples of patch_size."""
    h, w, _ = image.shape
    pad_h = (patch_size - (h % patch_size)) % patch_size
    pad_w = (patch_size - (w % patch_size)) % patch_size
    padded = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    return padded

# Get sorted list of PNG files for HR and LR images
hr_images = sorted(glob.glob(os.path.join(HR_DIR, "*.png")))
lr_images = sorted(glob.glob(os.path.join(LR_DIR, "*.png")))

print(f"Found {len(hr_images)} HR images and {len(lr_images)} LR images.")

assert len(hr_images) == len(lr_images), "Mismatch between HR and LR file counts!"

# Process each image pair
total_chunks = 0
for hr_path, lr_path in zip(hr_images, lr_images):
    print(f"\nProcessing {os.path.basename(hr_path)} ...")
    # Read images
    hr_img = cv2.imread(hr_path, cv2.IMREAD_COLOR)
    lr_img = cv2.imread(lr_path, cv2.IMREAD_COLOR)

    if hr_img is None or lr_img is None:
        print(f"Warning: Could not load one of the images: {hr_path}, {lr_path}. Skipping.")
        continue

    # Ensure corresponding file names match
    if os.path.basename(hr_path) != os.path.basename(lr_path):
        print(f"Error: Filename mismatch between {hr_path} and {lr_path}")
        continue

    # Apply reflect padding
    hr_img = reflect_pad(hr_img, HR_CHUNK_SIZE)
    lr_img = reflect_pad(lr_img, LR_CHUNK_SIZE)

    # Get padded sizes
    h_hr, w_hr, _ = hr_img.shape
    h_lr, w_lr, _ = lr_img.shape

    chunk_map = []  # List to store (HR_chunk, LR_chunk, random_name)

    # Extract patches from HR image and matching LR image
    for y in range(0, h_hr, HR_CHUNK_SIZE):
        for x in range(0, w_hr, HR_CHUNK_SIZE):
            hr_chunk = hr_img[y:y+HR_CHUNK_SIZE, x:x+HR_CHUNK_SIZE]
            # LR image is 4x smaller, so scale coordinates
            lr_y = y // 4
            lr_x = x // 4
            lr_chunk = lr_img[lr_y:lr_y+LR_CHUNK_SIZE, lr_x:lr_x+LR_CHUNK_SIZE]

            random_name = f"{random.randint(100000, 999999)}.png"
            chunk_map.append((hr_chunk, lr_chunk, random_name))
    
    print(f"Extracted {len(chunk_map)} chunks from {os.path.basename(hr_path)}")
    total_chunks += len(chunk_map)

    # Shuffle chunk_map to randomize filenames
    random.shuffle(chunk_map)

    # Save each chunk pair
    for hr_chunk, lr_chunk, random_name in chunk_map:
        hr_save_path = os.path.join(HR_OUT, random_name)
        lr_save_path = os.path.join(LR_OUT, random_name)
        success_hr = cv2.imwrite(hr_save_path, hr_chunk)
        success_lr = cv2.imwrite(lr_save_path, lr_chunk)
        if not (success_hr and success_lr):
            print(f"Error saving chunk {random_name}")
        else:
            print(f"Saved chunk: {random_name}")

print(f"\nProcessing complete. Total chunks extracted: {total_chunks}")
print("Chunks saved successfully!")
