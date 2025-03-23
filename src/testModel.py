import os
import math
import torch
import cv2
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import save_image
from tqdm import tqdm

from wgan_gp import Generator  # Import your generator definition
# from dataloader_DIV2K import PatchIterableDataset # Not needed for single-image testing

# ---------------------------
# Configuration
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Suppose your generator expects LR patches of 128×128, upscaled by ×4 → 512×512
LR_PATCH_SIZE = 128
UPSCALE_FACTOR = 4
HR_PATCH_SIZE = LR_PATCH_SIZE * UPSCALE_FACTOR  # 512

# Normalization used during training
mean = (0.5, 0.5, 0.5)
std  = (0.5, 0.5, 0.5)
normalize = T.Normalize(mean, std)

CHECKPOINT_DIR = "/home/aravos/Code/02-Upscale-Project/Image-Upscaler/checkpoints"
VALID_LR_DIR   = "/home/aravos/Code/02-Upscale-Project/Image-Upscaler/dataset/DIV2K/DIV2K_valid_LR_blurred"
VALID_HR_DIR   = "/home/aravos/Code/02-Upscale-Project/Image-Upscaler/dataset/DIV2K/DIV2K_valid_HR"
OUTPUT_DIR     = "test_outputs_fix"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# Load Generator
# ---------------------------
def load_latest_checkpoint(checkpoint_dir):
    ckpts = [f for f in os.listdir(checkpoint_dir) if f.startswith("gan_epoch_") and f.endswith(".pth")]
    if not ckpts:
        raise FileNotFoundError("No checkpoints found in directory.")

    latest_ckpt = max(ckpts, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    ckpt_path   = os.path.join(checkpoint_dir, latest_ckpt)
    print(f"Loading checkpoint: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    gen = Generator(channels_img=3, features_g=256).to(device)
    gen.load_state_dict(checkpoint["gen_state_dict"])
    gen.eval()
    return gen

generator = load_latest_checkpoint(CHECKPOINT_DIR)

# ---------------------------
# Upscale Function
# ---------------------------
def upscale_image_chunked(lr_path, generator):
    """
    1) Read LR image in BGR
    2) Convert BGR->RGB->Tensor, then Normalize(mean=0.5, std=0.5)
    3) Pad so dims are multiple of LR_PATCH_SIZE=128
    4) Chunk into (128×128), run generator => (512×512)
    5) Reassemble, remove padding
    6) Convert from [-1,1] -> [0,1] exactly once
    7) Return final upscaled PIL image
    """
    # 1) Read LR image
    bgr = cv2.imread(lr_path)
    if bgr is None:
        raise ValueError(f"Could not load {lr_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    # 2) Convert to tensor & Normalize
    to_tensor = T.ToTensor()
    lr_tensor = to_tensor(pil).unsqueeze(0).to(device)  # shape: [1,3,H,W]
    lr_tensor = normalize(lr_tensor.squeeze(0)).unsqueeze(0)  # apply mean/std => range ~[-1,1]

    # 3) Pad
    _, _, H, W = lr_tensor.shape
    pad_h = (LR_PATCH_SIZE - (H % LR_PATCH_SIZE)) % LR_PATCH_SIZE
    pad_w = (LR_PATCH_SIZE - (W % LR_PATCH_SIZE)) % LR_PATCH_SIZE
    lr_tensor = F.pad(lr_tensor, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, H_pad, W_pad = lr_tensor.shape

    # Prepare final SR canvas
    out_h = H_pad * UPSCALE_FACTOR
    out_w = W_pad * UPSCALE_FACTOR
    sr_canvas = torch.zeros((1, 3, out_h, out_w), device=device)

    # 4) Chunk & run generator
    for i in range(0, H_pad, LR_PATCH_SIZE):
        for j in range(0, W_pad, LR_PATCH_SIZE):
            lr_patch = lr_tensor[:, :, i:i+LR_PATCH_SIZE, j:j+LR_PATCH_SIZE]
            with torch.no_grad():
                sr_patch = generator(lr_patch)  # shape [1,3,512,512] in [-1,1]

            # Place sr_patch in final canvas
            sr_i, sr_j = i * UPSCALE_FACTOR, j * UPSCALE_FACTOR
            sr_canvas[:, :, sr_i:sr_i+HR_PATCH_SIZE, sr_j:sr_j+HR_PATCH_SIZE] = sr_patch

    # 5) Remove padding
    sr_canvas = sr_canvas[:, :, :H*UPSCALE_FACTOR, :W*UPSCALE_FACTOR]

    # 6) Convert from [-1,1] -> [0,1] exactly once
    sr_canvas = (sr_canvas * 0.5) + 0.5
    sr_canvas = sr_canvas.clamp(0, 1)

    # 7) Convert to PIL
    sr_pil = T.ToPILImage()(sr_canvas.squeeze(0).cpu())
    return sr_pil

# ---------------------------
# Testing Over Validation
# ---------------------------
def test_generator_on_valid(generator, lr_dir, hr_dir, output_dir):
    lr_files = sorted([f for f in os.listdir(lr_dir) if f.lower().endswith((".png",".jpg",".jpeg"))])
    for idx, lr_file in enumerate(tqdm(lr_files, desc="Testing")):
        lr_path = os.path.join(lr_dir, lr_file)
        sr_pil = upscale_image_chunked(lr_path, generator)

        # If matching HR exists, do side-by-side
        hr_path = os.path.join(hr_dir, lr_file)
        if not os.path.exists(hr_path):
            # No HR => just save SR
            sr_pil.save(os.path.join(output_dir, f"SR_{idx}_{lr_file}"))
            continue

        hr_bgr = cv2.imread(hr_path)
        if hr_bgr is None:
            sr_pil.save(os.path.join(output_dir, f"SR_{idx}_{lr_file}"))
            continue

        hr_rgb = cv2.cvtColor(hr_bgr, cv2.COLOR_BGR2RGB)
        hr_pil = Image.fromarray(hr_rgb)

        # For a quick side-by-side, resize HR to same size as SR
        sr_w, sr_h = sr_pil.size
        hr_resize = hr_pil.resize((sr_w, sr_h), Image.BICUBIC)

        combined = Image.new("RGB", (sr_w * 2, sr_h))
        combined.paste(sr_pil, (0, 0))
        combined.paste(hr_resize, (sr_w, 0))

        combined.save(os.path.join(output_dir, f"compare_{idx}_{lr_file}"))
        sr_pil.save(os.path.join(output_dir, f"SR_{idx}_{lr_file}"))

    print(f"Test complete. Results in {output_dir}")

# ---------------------------
# Main
# ---------------------------
if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    test_generator_on_valid(generator, VALID_LR_DIR, VALID_HR_DIR, OUTPUT_DIR)
    print("Done testing!")
