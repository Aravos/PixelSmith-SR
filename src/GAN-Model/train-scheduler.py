import os
import math
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from wgan_gp import Generator, Critic, initialize_weights
from dataloader_DIV2K import PatchIterableDataset
from utils import gradient_penalty

# ======================
# Configuration
# ======================

torch.backends.cudnn.benchmark = True
device = "cuda" if torch.cuda.is_available() else "cpu"

# Training Parameters
PATCH_SIZE = 512
BATCH_SIZE = 1
CHANNELS_IMG = 3
TOTAL_EPOCHS = 200
WARMUP_EPOCHS = 5     # number of epochs to linearly warm up LR
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

# Model Architecture
GEN_FEATURES = 256
CRITIC_FEATURES = 64

# Learning Rates
CRITIC_LR = 1e-4
GEN_LR = 5e-5

# Optimizer Parameters
optim_beta1 = 0.5
optim_beta2 = 0.999

# Paths
HR_TRAIN_DIR = "/home/aravos/Code/02-Upscale-Project/Image-Upscaler/dataset/DIV2K/DIV2K_train_HR"
LR_TRAIN_DIR = "/home/aravos/Code/02-Upscale-Project/Image-Upscaler/dataset/DIV2K/DIV2K_train_LR_blurred"

save_dir = "/home/aravos/Code/02-Upscale-Project/Image-Upscaler/checkpoints"
os.makedirs(save_dir, exist_ok=True)

# ======================
# Model Initialization
# ======================
gen = Generator(channels_img=CHANNELS_IMG, features_g=GEN_FEATURES).to(device)
critic = Critic(channels_img=CHANNELS_IMG, features_d=CRITIC_FEATURES).to(device)
initialize_weights(gen)
initialize_weights(critic)

# ======================
# Optimizer Setup
# ======================
opt_gen = optim.Adam(gen.parameters(), lr=GEN_LR, betas=(optim_beta1, optim_beta2))
opt_critic = optim.Adam(critic.parameters(), lr=CRITIC_LR, betas=(optim_beta1, optim_beta2))

# ======================
# Cosine With Warmup Scheduler
# ======================
def get_cosine_with_warmup_scheduler(optimizer, total_epochs, warmup_epochs, eta_min=1e-6):
    """
    1) Linear warmup for `warmup_epochs`.
    2) Cosine decay from epoch `warmup_epochs` to `total_epochs`.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # linear warmup
            return (epoch + 1) / warmup_epochs
        # remaining epochs -> cosine schedule
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        # eq. => eta_min + 0.5*(1-eta_min)*(1 + cos(pi * progress))
        return eta_min + 0.5 * (1 - eta_min) * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)

scheduler_gen = get_cosine_with_warmup_scheduler(opt_gen, TOTAL_EPOCHS, WARMUP_EPOCHS)
scheduler_critic = get_cosine_with_warmup_scheduler(opt_critic, TOTAL_EPOCHS, WARMUP_EPOCHS)

# ======================
# TensorBoard Writers
# ======================
writer_real = SummaryWriter("/home/aravos/Code/02-Upscale-Project/Image-Upscaler/logs-WGAN/real")
writer_fake = SummaryWriter("/home/aravos/Code/02-Upscale-Project/Image-Upscaler/logs-WGAN/fake")

# ======================
# Resume from Checkpoint
# ======================
start_epoch = 0
checkpoint_files = [f for f in os.listdir(save_dir) if f.startswith("gan_epoch_") and f.endswith(".pth")]
if checkpoint_files:
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    checkpoint_path = os.path.join(save_dir, latest_checkpoint)
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gen.load_state_dict(checkpoint["gen_state_dict"])
    critic.load_state_dict(checkpoint["critic_state_dict"])
    opt_gen.load_state_dict(checkpoint["opt_gen_state_dict"])
    opt_critic.load_state_dict(checkpoint["opt_critic_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
else:
    print("No checkpoint found, starting from scratch.")

gen.train()
critic.train()

# ======================
# Create Dataset
# ======================
train_dataset = PatchIterableDataset(
    hr_dir=HR_TRAIN_DIR,
    lr_dir=LR_TRAIN_DIR,
    patch_size=PATCH_SIZE
)

# ======================
# Training Loop
# ======================
for epoch in range(start_epoch, TOTAL_EPOCHS):
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{TOTAL_EPOCHS}]", leave=True)

    total_gen_loss = 0.0
    total_crit_loss = 0.0
    num_batches = 0

    for batch_idx, (lr_batch, hr_batch) in enumerate(loop):
        lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)

        # Flatten if shape is (B, P, C, H, W)
        if len(lr_batch.shape) == 5:  # e.g., (1,12,3,128,128)
            B, P, C, H_lr, W_lr = lr_batch.shape
            lr_batch = lr_batch.view(B * P, C, H_lr, W_lr)
            hr_batch = hr_batch.view(B * P, C, 512, 512)

        # If we expect 12 patches, randomly select 6
        B, C, H_lr, W_lr = lr_batch.shape
        if B >= 6:
            indices = torch.randperm(B)[:6]
            lr_batch = lr_batch[indices]
            hr_batch = hr_batch[indices]

        # 1) Train Critic
        for _ in range(CRITIC_ITERATIONS):
            opt_critic.zero_grad(set_to_none=True)

            fake_hr = gen(lr_batch)
            critic_real = critic(hr_batch).reshape(-1)
            critic_fake = critic(fake_hr.detach()).reshape(-1)

            gp = gradient_penalty(critic, hr_batch, fake_hr, device)
            loss_critic = LAMBDA_GP * gp - (torch.mean(critic_real) - torch.mean(critic_fake))

            loss_critic.backward(retain_graph=True)
            opt_critic.step()
            total_crit_loss += loss_critic.item()

        # 2) Train Generator
        opt_gen.zero_grad(set_to_none=True)
        fake_hr = gen(lr_batch)
        loss_gen = -torch.mean(critic(fake_hr).reshape(-1))

        loss_gen.backward()
        opt_gen.step()
        total_gen_loss += loss_gen.item()

        num_batches += 1
        loop.set_postfix(
            AvgLoss_C=total_crit_loss / num_batches,
            AvgLoss_G=total_gen_loss / num_batches,
            lr_gen=scheduler_gen.get_last_lr()[0],
            lr_crit=scheduler_critic.get_last_lr()[0]
        )

        # Log images every 50 steps
        if num_batches % 50 == 0:
            gen.eval()
            with torch.no_grad():
                fake_for_log_flat = gen(lr_batch)
            gen.train()
            img_grid_real = make_grid(hr_batch[:6], nrow=3, normalize=True)
            img_grid_fake = make_grid(fake_for_log_flat[:6], nrow=3, normalize=True)
            writer_real.add_image("Train/Real", img_grid_real, global_step=epoch * len(train_loader) + batch_idx)
            writer_fake.add_image("Train/Fake", img_grid_fake, global_step=epoch * len(train_loader) + batch_idx)

    # Step the LR schedulers
    scheduler_gen.step()
    scheduler_critic.step()

    # Save checkpoint every 5 epochs or final epoch
    if (epoch + 1) % 5 == 0 or epoch == TOTAL_EPOCHS - 1:
        torch.save({
            "epoch": epoch,
            "gen_state_dict": gen.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "opt_gen_state_dict": opt_gen.state_dict(),
            "opt_critic_state_dict": opt_critic.state_dict(),
        }, os.path.join(save_dir, f"gan_epoch_{epoch+1}.pth"))
        print(f"Model saved at epoch {epoch+1}")

print("Training completed successfully!")
