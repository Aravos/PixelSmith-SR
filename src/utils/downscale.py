import os
import cv2
import numpy as np

# Input and output directories
DATASET_DIR = "dataset/DIV2K"
HR_TRAIN_DIR = os.path.join(DATASET_DIR, "DIV2K_train_HR")
HR_VALID_DIR = os.path.join(DATASET_DIR, "DIV2K_valid_HR")
OUTPUT_TRAIN_DIR = os.path.join(DATASET_DIR, "DIV2K_train_LR_blurred")
OUTPUT_VALID_DIR = os.path.join(DATASET_DIR, "DIV2K_valid_LR_blurred")

# Ensure output directories exist
os.makedirs(OUTPUT_TRAIN_DIR, exist_ok=True)
os.makedirs(OUTPUT_VALID_DIR, exist_ok=True)

# Downscaling function
def process_image(image_path, output_path, scale=4, blur_kernel=5):
    img = cv2.imread(image_path)
    h, w, c = img.shape
    blurred = cv2.GaussianBlur(img, (blur_kernel, blur_kernel), 0)
    downscaled = cv2.resize(blurred, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(output_path, downscaled)

for folder, output_folder in [(HR_TRAIN_DIR, OUTPUT_TRAIN_DIR), (HR_VALID_DIR, OUTPUT_VALID_DIR)]:
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            input_path = os.path.join(folder, filename)
            output_path = os.path.join(output_folder, filename)
            process_image(input_path, output_path)
            print(f"✅ Processed: {filename}")

print("\n🎉 Done! All images have been processed and saved.")
