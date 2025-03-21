import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm
import os
from wgan_gp import Generator, Critic, initialize_weights
from dataloader_DIV2K import PatchIterableDataset
from utils import gradient_penalty

LEARNING_RATE = 1e-4
PATCH_SIZE = 512
BATCH_SIZE = 8
CHANNELS_IMG = 3
NUM_EPOCHS = 5
GEN_FEATURES = 256
CRITIC_FEATURES = 64
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10
DATA_SHUFFLE = True

device = "cuda" if torch.cuda.is_available() else "cpu"

HR_DIR = "/home/aravos/Code/02-Upscale-Project/Image-Upscaler/dataset/DIV2K/DIV2K_train_HR"
LR_DIR = "/home/aravos/Code/02-Upscale-Project/Image-Upscaler/dataset/DIV2K/DIV2K_train_LR_blurred"

gen = Generator(channels_img=3, features_g=GEN_FEATURES).to(device)
critic = Critic(channels_img=3, features_d=CRITIC_FEATURES).to(device)
initialize_weights(gen)
initialize_weights(critic)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

writer_real = SummaryWriter("/home/aravos/Code/02-Upscale-Project/Image-Upscaler/logs-WGAN/real")
writer_fake = SummaryWriter("/home/aravos/Code/02-Upscale-Project/Image-Upscaler/logs-WGAN/fake")

save_dir = "/home/aravos/Code/02-Upscale-Project/Image-Upscaler/checkpoints"
os.makedirs(save_dir, exist_ok=True)

step = 0
gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    if(DATA_SHUFFLE):
        dataset = PatchIterableDataset(HR_DIR, LR_DIR, patch_size=PATCH_SIZE, shuffle=True)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    loop = tqdm(enumerate(loader), total=len(loader), leave=True)
    
    for batch_idx, (lr_batch, hr_batch) in loop:
        lr_batch = lr_batch.to(device)
        hr_batch = hr_batch.to(device)

        for i in range(CRITIC_ITERATIONS):
            fake_hr = gen(lr_batch)
            critic_real = critic(hr_batch).reshape(-1)
            critic_fake = critic(fake_hr.detach()).reshape(-1)
            gp = gradient_penalty(critic, hr_batch, fake_hr, device)
            loss_critic = LAMBDA_GP * gp - (torch.mean(critic_real) - torch.mean(critic_fake))
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        fake_hr = gen(lr_batch)
        critic_fake_for_gen = critic(fake_hr).reshape(-1)
        loss_gen = -torch.mean(critic_fake_for_gen)

        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx % 50 == 0:
            with torch.no_grad():
                fake_for_log = gen(lr_batch)
                img_grid_real = make_grid(hr_batch[:8],nrow=4, normalize=True)
                img_grid_fake = make_grid(fake_for_log[:8],nrow=4, normalize=True)
                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)
            step += 1

        loop.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        loop.set_postfix(Loss_D=loss_critic.item(), Loss_G=loss_gen.item())

    torch.save({
        'epoch': epoch,
        'gen_state_dict': gen.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'opt_gen_state_dict': opt_gen.state_dict(),
        'opt_critic_state_dict': opt_critic.state_dict(),
    }, os.path.join(save_dir, f"gan_epoch_{epoch+1}.pth"))

    print(f"Model saved at epoch {epoch+1}")

print("Training Complete!")