import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from wgan_gp import Generator  # Your generator definition
from dataloader_DIV2K import PatchIterableDataset  # Updated dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
VALID_HR_DIR = "/home/aravos/Code/02-Upscale-Project/Image-Upscaler/dataset/DIV2K/DIV2K_valid_HR"
VALID_LR_DIR = "/home/aravos/Code/02-Upscale-Project/Image-Upscaler/dataset/DIV2K/DIV2K_valid_LR_blurred"
CHECKPOINT_PATH = "/home/aravos/Code/02-Upscale-Project/Image-Upscaler/checkpoints/gan_epoch_5.pth"
VAL_LOG_DIR = "/home/aravos/Code/02-Upscale-Project/Image-Upscaler/logs-WGAN/val"

def main():
    # 1) Instantiate the generator with the same hyperparameters as used in training.
    gen = Generator(channels_img=3, features_g=256).to(DEVICE)

    # 2) Load the trained checkpoint.
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[WARNING] No checkpoint found at {CHECKPOINT_PATH}. Using random weights.")
    else:
        print(f"Loading checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        gen.load_state_dict(checkpoint["gen_state_dict"])

    gen.eval()  # set to inference mode

    # 3) Create the validation dataset & DataLoader.
    # Since each sample (image) returns 12 patches already,
    # we use batch_size=1 so that each batch is one sample.
    val_dataset = PatchIterableDataset(
        hr_dir=VALID_HR_DIR,
        lr_dir=VALID_LR_DIR,
        patch_size=512,      # same patch size as training
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 4) TensorBoard writer for validation logs.
    writer_val = SummaryWriter(VAL_LOG_DIR)
    step = 0

    # 5) Validation loop.
    with torch.no_grad():
        loop = tqdm(enumerate(val_loader), total=len(val_loader), leave=True)
        for batch_idx, (lr_batch, hr_batch) in loop:
            # Each batch has shape: (1, 12, 3, H, W).
            # Remove the batch dimension so that:
            # lr_batch -> (12, 3, 128, 128) and hr_batch -> (12, 3, 512, 512)
            lr_batch = lr_batch.squeeze(0).to(DEVICE)
            hr_batch = hr_batch.squeeze(0).to(DEVICE)

            # Forward pass through the generator.
            fake_hr = gen(lr_batch)  # Expecting fake_hr shape: (12, 3, 512, 512)

            # Log to TensorBoard every 10 batches.
            if batch_idx % 10 == 0:
                # make_grid will arrange patches in a grid.
                real_grid = make_grid(hr_batch, nrow=4, normalize=True)
                fake_grid = make_grid(fake_hr, nrow=4, normalize=True)

                writer_val.add_image("Val_Real", real_grid, global_step=step)
                writer_val.add_image("Val_Fake", fake_grid, global_step=step)
            step += 1

    print("Validation complete. Logged images to TensorBoard.")
    writer_val.close()

if __name__ == "__main__":
    main()
