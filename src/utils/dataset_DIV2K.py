import os
import cv2
import math
import shutil

REQUIRED_NUM_PATCHES = 12

BASE_DIR = "dataset/DIV2K"
TRAIN_HR_DIR = os.path.join(BASE_DIR, "DIV2K_train_HR")
TRAIN_LR_DIR = os.path.join(BASE_DIR, "DIV2K_train_LR_blurred")
VALID_HR_DIR = os.path.join(BASE_DIR, "DIV2K_valid_HR")
VALID_LR_DIR = os.path.join(BASE_DIR, "DIV2K_valid_LR_blurred")

# Excluded directories (one per folder)
EXCLUDED_TRAIN_HR = os.path.join(BASE_DIR, "excluded_train_hr")
EXCLUDED_TRAIN_LR = os.path.join(BASE_DIR, "excluded_train_lr")
EXCLUDED_VALID_HR = os.path.join(BASE_DIR, "excluded_valid_hr")
EXCLUDED_VALID_LR = os.path.join(BASE_DIR, "excluded_valid_lr")

# Create them if they don't exist
os.makedirs(EXCLUDED_TRAIN_HR, exist_ok=True)
os.makedirs(EXCLUDED_TRAIN_LR, exist_ok=True)
os.makedirs(EXCLUDED_VALID_HR, exist_ok=True)
os.makedirs(EXCLUDED_VALID_LR, exist_ok=True)

def num_patches_for_image(image_path, patch_size):
    """Reads image and computes how many patches it yields (no zero padding)."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    h, w, _ = img.shape
    h_patches = math.ceil(h / patch_size)
    w_patches = math.ceil(w / patch_size)
    return h_patches * w_patches

def move_if_not_12(dir_path, move_to, patch_size):
    """
    For all images in dir_path, if total patches != REQUIRED_NUM_PATCHES,
    move them to move_to.
    """
    for fname in os.listdir(dir_path):
        if not (fname.lower().endswith(".png") or fname.lower().endswith(".jpg")):
            continue  # skip non-image files

        full_path = os.path.join(dir_path, fname)
        n_patches = num_patches_for_image(full_path, patch_size)
        if n_patches is None:
            print(f"[WARNING] Could not read {full_path}. Moving to {move_to}.")
            shutil.move(full_path, os.path.join(move_to, fname))
            continue

        if n_patches != REQUIRED_NUM_PATCHES:
            print(f"Moving {fname} => yields {n_patches} patches (not {REQUIRED_NUM_PATCHES}).")
            shutil.move(full_path, os.path.join(move_to, fname))

# For HR images => patch_size=512
move_if_not_12(TRAIN_HR_DIR, EXCLUDED_TRAIN_HR, patch_size=512)
move_if_not_12(VALID_HR_DIR, EXCLUDED_VALID_HR, patch_size=512)

# For LR images => patch_size=128
move_if_not_12(TRAIN_LR_DIR, EXCLUDED_TRAIN_LR, patch_size=128)
move_if_not_12(VALID_LR_DIR, EXCLUDED_VALID_LR, patch_size=128)

print("Done! Images that don't yield exactly 12 patches have been moved.")
