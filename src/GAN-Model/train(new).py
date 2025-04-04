import os

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision.utils import make_grid
from tqdm import tqdm

from wgan_gp import Generator, Critic, initialize_weights
from dataloader_DIV2K import PatchIterableDataset
from utils import gradient_penalty

torch.backends.cudnn.benchmark = True  # Optimizes CUDA kernels
torch.cuda.empty_cache()  # Clears unused memory

LEARNING_RATE = 5e-5
PATCH_SIZE = 512
BATCH_SIZE = 1
CHANNELS_IMG = 3
ADDITIONAL_EPOCHS = 25
GEN_FEATURES = 256
CRITIC_FEATURES = 64
CRITIC_ITERATIONS = 5
LAMBDA_GP = 15
optim_beta1 = 0.5
optim_beta2 = 0.999

device = "cuda" if torch.cuda.is_available() else "cpu"

HR_TRAIN_DIR = "/home/aravos/Code/02-Upscale-Project/Image-Upscaler/dataset/DIV2K/DIV2K_train_HR"
LR_TRAIN_DIR = "/home/aravos/Code/02-Upscale-Project/Image-Upscaler/dataset/DIV2K/DIV2K_train_LR_blurred"
VALID_HR_DIR = "/home/aravos/Code/02-Upscale-Project/Image-Upscaler/dataset/DIV2K/DIV2K_valid_HR"
VALID_LR_DIR = "/home/aravos/Code/02-Upscale-Project/Image-Upscaler/dataset/DIV2K/DIV2K_valid_LR_blurred"

save_dir = "/home/aravos/Code/02-Upscale-Project/Image-Upscaler/checkpoints"
os.makedirs(save_dir, exist_ok=True)
writer_real = SummaryWriter("/home/aravos/Code/02-Upscale-Project/Image-Upscaler/logs-WGAN/real")
writer_fake = SummaryWriter("/home/aravos/Code/02-Upscale-Project/Image-Upscaler/logs-WGAN/fake")

gen = Generator(channels_img=CHANNELS_IMG, features_g=GEN_FEATURES).to(device)
critic = Critic(channels_img=CHANNELS_IMG, features_d=CRITIC_FEATURES).to(device)
initialize_weights(gen)
initialize_weights(critic)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(optim_beta1 , optim_beta2))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(optim_beta1 , optim_beta2))

# Define LR schedulers
scheduler_gen = StepLR(opt_gen, step_size=10, gamma=0.5)
scheduler_critic = StepLR(opt_critic, step_size=10, gamma=0.5)

start_epoch = 0
# summary(gen, input_size=(1, 3, 128, 128), device=device)
# summary(critic, input_size=(1, 3, 512, 512), device=device)
# sys.exit()
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
    start_epoch = checkpoint["epoch"]
else:
    print("No checkpoint found, starting from scratch.")

gen.train()
critic.train()

total_epochs = start_epoch + ADDITIONAL_EPOCHS

step = 0

import random

for epoch in range(start_epoch, total_epochs):
    train_dataset = PatchIterableDataset(
        hr_dir=HR_TRAIN_DIR,
        lr_dir=LR_TRAIN_DIR,
        patch_size=PATCH_SIZE
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)

    total_gen_loss = 0
    total_crit_loss = 0
    num_gen_batches = 0
    num_crit_batches = 0
    for batch_idx, (lr_batch, hr_batch) in loop:
        lr_batch = lr_batch.to(device)
        hr_batch = hr_batch.to(device)

        print(f"Original lr_batch shape: {lr_batch.shape}")  # Debugging output
        print(f"Original hr_batch shape: {hr_batch.shape}")

        if len(lr_batch.shape) == 5:  # (B, P, C, H, W) case
            B, P, C, H_lr, W_lr = lr_batch.shape
            lr_batch = lr_batch.view(B * P, C, H_lr, W_lr)
            hr_batch = hr_batch.view(B * P, C, 512, 512)  # Adjusted for HR

        print(f"Processed lr_batch shape: {lr_batch.shape}")
        print(f"Processed hr_batch shape: {hr_batch.shape}")

        B, C, H_lr, W_lr = lr_batch.shape  # (12, 3, 128, 128)
        _, _, H_hr, W_hr = hr_batch.shape   # (12, 3, 512, 512)

        # Randomly pick 6 indices out of 12
        indices = torch.randperm(B)[:6]

        # Select 6 random samples
        lr_batch = lr_batch[indices]
        hr_batch = hr_batch[indices]

        print(f"Randomly selected lr_batch shape: {lr_batch.shape}")
        print(f"Randomly selected hr_batch shape: {hr_batch.shape}")

        # ----- Train Critic -----
        for _ in range(CRITIC_ITERATIONS):
            num_crit_batches+=1
            opt_critic.zero_grad(set_to_none=True)
            fake_hr_flat = gen(lr_batch)
            critic_real = critic(hr_batch).reshape(-1)
            critic_fake = critic(fake_hr_flat.detach()).reshape(-1)
            gp = gradient_penalty(critic, hr_batch, fake_hr_flat, device)
            loss_critic = LAMBDA_GP * gp - (torch.mean(critic_real) - torch.mean(critic_fake))
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        # ----- Train Generator -----
        fake_hr_flat = gen(lr_batch)
        critic_fake_for_gen = critic(fake_hr_flat).reshape(-1)
        loss_gen = -torch.mean(critic_fake_for_gen)

        loss_gen.backward()
        opt_gen.step()
        opt_gen.zero_grad()

        if batch_idx % 250 == 0:
            gen.eval()
            with torch.no_grad():
                fake_for_log_flat = gen(lr_batch)
            gen.train()
            img_grid_real = make_grid(hr_batch[:6], nrow=3, normalize=True)
            img_grid_fake = make_grid(fake_for_log_flat[:6], nrow=3, normalize=True)
            writer_real.add_image("Train/Real", img_grid_real, global_step=epoch * len(train_loader) + batch_idx)
            writer_fake.add_image("Train/Fake", img_grid_fake, global_step=epoch * len(train_loader) + batch_idx)
        
        num_gen_batches += 1
        total_gen_loss += loss_critic.item()
        total_crit_loss += loss_gen.item()
        loop.set_description(f"Epoch [{epoch+1}/{total_epochs}]")
        loop.set_postfix(AvgLoss_C=total_crit_loss/num_crit_batches, AvgLoss_G=total_gen_loss/num_gen_batches)
        step += 1
    
    
    scheduler_gen.step()
    scheduler_critic.step()
    
    if (epoch+1) % 5 == 0:
        checkpoint_path = os.path.join(save_dir, f"gan_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch+1,
            'gen_state_dict': gen.state_dict(),
            'critic_state_dict': critic.state_dict(),
            'opt_gen_state_dict': opt_gen.state_dict(),
            'opt_critic_state_dict': opt_critic.state_dict(),
        }, checkpoint_path)
        print(f"Model saved at epoch {epoch+1}")

print("Training Complete!")
