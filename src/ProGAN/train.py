import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from math import log2
import config
from models import Generator, Critic
from utils import gradient_penalty, plot_to_tensorboard, save_checkpoint, load_checkpoint

torch.backends.cudnn.benchmark = True

def train_fn(
    critic,
    gen,
    dataset_list,
    step,
    alpha,
    opt_critic,
    opt_gen,
    tensorboard_step,
    writer,
    scaler_gen,
    scaler_critic,
):
    # Use dataset_list[step] for the current progressive stage.
    dataloader = DataLoader(
        dataset_list[step],
        batch_size=config.BATCH_SIZES[step],
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    loop = tqdm(dataloader, desc=f"Training step {step}", leave=True)
    for batch_idx, (lr_img, hr_img) in enumerate(loop):
        hr_img = hr_img.to(config.DEVICE)
        lr_img = lr_img.to(config.DEVICE)

        # Train Critic: maximize E[critic(real)] - E[critic(fake)]
        for i in range(config.CRITIC_ITERATIONS):
            with torch.amp.autocast(device_type=config.DEVICE):
                fake_hr = gen(lr_img, alpha, step)
                critic_real = critic(hr_img, alpha, step)
                critic_fake = critic(fake_hr.detach(), alpha, step)
                gp = gradient_penalty(critic, hr_img, fake_hr, alpha, step, device=config.DEVICE)
                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake))
                    + config.LAMBDA_GP * gp
                    + (0.001 * torch.mean(critic_real ** 2))
                )

            opt_critic.zero_grad(set_to_none=True)
            scaler_critic.scale(loss_critic).backward()
            scaler_critic.step(opt_critic)
            scaler_critic.update()

        # Train Generator: minimize -E[critic(gen(fake))]
        with torch.amp.autocast(device_type=config.DEVICE):
            gen_fake = critic(fake_hr, alpha, step)
            loss_gen = -torch.mean(gen_fake)

        opt_gen.zero_grad(set_to_none=True)
        scaler_gen.scale(loss_gen).backward()
        scaler_gen.step(opt_gen)
        scaler_gen.update()

        # Update alpha gradually over the dataset for this stage
        alpha += config.BATCH_SIZES[step] / (
            (config.PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset_list[step])
        )
        alpha = min(alpha, 1)

        if batch_idx % 70 == 0:
            plot_to_tensorboard(writer=writer, loss_critic=loss_critic, real=hr_img, fake=fake_hr,tensorboard_step=tensorboard_step, loss_gen=loss_gen)
            tensorboard_step += 1

        loop.set_postfix(gp=gp.item(), loss_critic=loss_critic.item())

    return tensorboard_step, alpha


def main():
    # Initialize models
    generator = Generator(in_channels=config.CHANNELS_IMG, img_channels=config.CHANNELS_IMG).to(config.DEVICE)
    critic = Critic(in_channels=config.CHANNELS_IMG, img_channels=config.CHANNELS_IMG).to(config.DEVICE)
    
    # Set up optimizers and gradient scalers for FP16 training
    opt_gen = optim.Adam(generator.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.9))
    scaler_gen = torch.amp.GradScaler(device=config.DEVICE)
    scaler_critic = torch.amp.GradScaler(device=config.DEVICE)
    
    writer = SummaryWriter(config.LOG_PATH)
    
    # Resume model if LOAD_MODEL is True
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, generator, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC, critic, opt_critic, config.LEARNING_RATE)
    
    generator.train()
    critic.train()
    
    tensorboard_step = 0
    # Compute initial progressive step from START_IMG_SIZE (e.g., if START_IMG_SIZE=128, step=0)
    step = int(log2(config.START_IMG_SIZE / 128))
    dataset_list = config.DATASET
    
    for num_epochs in config.PROGRESSIVE_EPOCHS[step:]:
        alpha = 1e-5
        for epoch in range(num_epochs):
            print(f"Epoch [{epoch+1}/{num_epochs}] at step {step}")
            tensorboard_step, alpha = train_fn(
                critic,
                generator,
                dataset_list,
                step,
                alpha,
                opt_critic,
                opt_gen,
                tensorboard_step,
                writer,
                scaler_gen,
                scaler_critic,
            )
            if config.SAVE_MODEL:
                save_checkpoint(generator, opt_gen, filename=config.CHECKPOINT_GEN)
                save_checkpoint(critic, opt_critic, filename=config.CHECKPOINT_CRITIC)
        step += 1

if __name__ == "__main__":
    main()
