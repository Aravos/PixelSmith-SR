import cv2
import torch
from math import log2
from dataloader_ProGAN import ChunkDataset

START_IMG_SIZE = 128
HR_DIR = "/home/aravos/Code/02-Upscale-Project/Image-Upscaler/dataset/Processed/HR_Chunks"
LR_128_DIR = "/home/aravos/Code/02-Upscale-Project/Image-Upscaler/dataset/Processed/LR_128"
LR_256_DIR = "/home/aravos/Code/02-Upscale-Project/Image-Upscaler/dataset/Processed/LR_256"
DATASET = [ChunkDataset(hr_dir=LR_128_DIR, lr_dir=LR_128_DIR),ChunkDataset(hr_dir=LR_256_DIR, lr_dir=LR_128_DIR),ChunkDataset(hr_dir=HR_DIR, lr_dir=LR_128_DIR)]
CHECKPOINT_GEN = "./02-Upscale-Project/Image-Upscaler/src/ProGAN/checkpoints/generator.pth"
CHECKPOINT_CRITIC = "./02-Upscale-Project/Image-Upscaler/src/ProGAN/checkpoints/critic.pth"
LOG_PATH = "./02-Upscale-Project/Image-Upscaler/src/ProGAN/logs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_MODEL = True
LOAD_MODEL = False
LEARNING_RATE = 1e-3
BATCH_SIZES = [64,16,4]
CHANNELS_IMG = 3
CRITIC_ITERATIONS = 3
LAMBDA_GP = 10
PROGRESSIVE_EPOCHS = [30] * len(BATCH_SIZES)
NUM_WORKERS = 4