import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from wgan_gp import Generator  # Your generator definition
from dataloader_DIV2K import PatchIterableDataset  # Same patch-based dataset
# If you have gradient_penalty or Critic here, you don't need them in validation
# from utils import gradient_penalty  # Not needed for validation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Path to your validation sets
VALID_HR_DIR = "/home/aravos/Code/02-Upscale-Project/Image-Upscaler/dataset/DIV2K/DIV2K_valid_HR"
VALID_LR_DIR = "/home/aravos/Code/02-Upscale-Project/Image-Upscaler/dataset/DIV2K/DIV2K_valid_LR_blurred"

# Model & checkpoint
CHECKPOINT_PATH = "/home/aravos/Code/02-Upscale-Project/Image-Upscaler/checkpoints/gan_epoch_5.pth"

# TensorBoard log directory for validation
VAL_LOG_DIR = "/home/aravos/Code/02-Upscale-Project/Image-Upscaler/logs-WGAN/val"

def main():
    # 1) Instantiate your generator (same hyperparams as training)
    gen = Generator(channels_img=3, features_g=256).to(DEVICE)

    # 2) Load the trained weights from epoch 5
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[WARNING] No checkpoint found at {CHECKPOINT_PATH}. Using random weights.")
    else:
        print(f"Loading checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        gen.load_state_dict(checkpoint["gen_state_dict"])

    gen.eval()  # set to inference mode

    # 3) Create validation dataset & loader
    val_dataset = PatchIterableDataset(
        hr_dir=VALID_HR_DIR,
        lr_dir=VALID_LR_DIR,
        patch_size=512,      # same patch size used in training
        shuffle=False        # no shuffle for validation
    )
    val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False, num_workers=0)

    # 4) TensorBoard writer for validation logs
    writer_val = SummaryWriter(VAL_LOG_DIR)

    step = 0

    # 5) Validation loop
    with torch.no_grad():
        loop = tqdm(enumerate(val_loader), total=len(val_loader), leave=True)
        for batch_idx, (lr_batch, hr_batch) in loop:
            lr_batch = lr_batch.to(DEVICE)
            hr_batch = hr_batch.to(DEVICE)

            # Forward pass
            fake_hr = gen(lr_batch)

            if batch_idx % 10 == 0:
                # make_grid normalizes dynamically by default if you pass normalize=True
                # remove or tweak as you prefer
                real_grid = make_grid(hr_batch[:12],nrow=4, normalize=True)
                fake_grid = make_grid(fake_hr[:12],nrow=4, normalize=True)

                writer_val.add_image("Val_Real", real_grid, global_step=step)
                writer_val.add_image("Val_Fake", fake_grid, global_step=step)

            step += 1

    print("Validation complete. Logged images to TensorBoard.")
    writer_val.close()

if __name__ == "__main__":
    main()
