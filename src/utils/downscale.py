import os
import cv2
import numpy as np

# Input and output directories
DATASET_DIR = "02-Upscale-Project/Image-Upscaler/dataset/DIV2K"
HR_TRAIN_DIR = os.path.join(DATASET_DIR, "DIV2K_train_HR")
OUTPUT_TRAIN_DIR = os.path.join(DATASET_DIR, "DIV2K_train_LR_blurred_2")

# Ensure output directories exist
os.makedirs(OUTPUT_TRAIN_DIR, exist_ok=True)

# Downscaling function
def process_image(image_path, output_path, scale=2, blur_kernel=5):
    img = cv2.imread(image_path)
    h, w, c = img.shape
    blurred = cv2.GaussianBlur(img, (blur_kernel, blur_kernel), 0)
    downscaled = cv2.resize(blurred, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(output_path, downscaled)

for filename in os.listdir(HR_TRAIN_DIR):
    if filename.endswith(".png"):
        input_path = os.path.join(HR_TRAIN_DIR, filename)
        output_path = os.path.join(OUTPUT_TRAIN_DIR, filename)
        process_image(input_path, output_path)
        print(f"Processed: {filename}")

print("\n Done")
