import cv2
import torch
from math import log2
import os
import sys

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
except NameError:
    current_dir = os.path.abspath(os.getcwd())
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    print(f"Warning: __file__ not defined. Using CWD: {current_dir}")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added to sys.path: {project_root}")
else:
    print(f"Project root already in sys.path: {project_root}")

try:
    from src.ProGAN.dataloader_ProGAN import ChunkDataset
    print("Successfully imported ChunkDataset.")
except ImportError as e:
    print(f"Error: Could not import ChunkDataset: {e}")
    print("Check if 'src/ProGAN/dataloader_ProGAN.py' exists and project root is correct.")
    exit()

dataset_base = os.path.join(project_root, "dataset", "Processed")

HR_DIR = os.path.join(dataset_base, "HR_Chunks")
LR_128_DIR = os.path.join(dataset_base, "LR_128")
LR_256_DIR = os.path.join(dataset_base, "LR_256")

for dir_path in [HR_DIR, LR_128_DIR, LR_256_DIR]:
    if not os.path.isdir(dir_path):
        print(f"Warning: Dataset directory not found: {dir_path}")
    else:
         print(f"Found dataset directory: {dir_path}")

CHECKPOINT_STATE = os.path.join(project_root, "src", "ProGAN", "checkpoints", "training_state.pth")
LOG_PATH = os.path.join(project_root, "src", "ProGAN", "logs")

print(f"\nAbsolute Checkpoint Path: {CHECKPOINT_STATE}")
print(f"Absolute Log Path:        {LOG_PATH}")

START_IMG_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_MODEL = True
LOAD_MODEL = True
LEARNING_RATE = 1e-3
BATCH_SIZES = [64, 32, 4]
CHANNELS_IMG = 3
CRITIC_ITERATIONS = 3
LAMBDA_GP = 10
PROGRESSIVE_EPOCHS = [40, 25, 20]
NUM_WORKERS = 4

try:
    DATASET = [
        ChunkDataset(hr_dir=LR_128_DIR, lr_dir=LR_128_DIR),
        ChunkDataset(hr_dir=LR_256_DIR, lr_dir=LR_128_DIR),
        ChunkDataset(hr_dir=HR_DIR, lr_dir=LR_128_DIR)
    ]
    print("\nSuccessfully initialized DATASET list.")
except NameError:
     print("\nDATASET list not initialized because ChunkDataset failed to import.")
     DATASET = []
except Exception as e:
    print(f"\nError initializing DATASET list: {e}")
    print("Check dataset paths and ChunkDataset class.")
    DATASET = []

try:
    os.makedirs(LOG_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(CHECKPOINT_STATE), exist_ok=True)
    print(f"Ensured log directory exists: {LOG_PATH}")
    print(f"Ensured checkpoint directory exists: {os.path.dirname(CHECKPOINT_STATE)}")
except OSError as e:
    print(f"Error creating output directories: {e}")