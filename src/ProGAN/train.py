import os
import torch
import sys
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from math import log2
import config
from models import Generator, Critic
from utils import gradient_penalty,plot_to_tensorboard,save_training_state,load_training_state
    
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
    dataloader = DataLoader(dataset_list[step],batch_size=config.BATCH_SIZES[step],shuffle=True,num_workers=config.NUM_WORKERS)
    num_batches = len(dataloader)
    quater_batch = num_batches // 4

    loop = tqdm(dataloader, desc=f"Training step {step}", leave=True)
    for batch_idx, (lr_img, hr_img) in enumerate(loop):
        hr_img = hr_img.to(config.DEVICE)
        lr_img = lr_img.to(config.DEVICE)

        # Train Critic: maximize E[critic(real)] - E[critic(fake)]
        for _ in range(config.CRITIC_ITERATIONS):
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

        alpha += config.BATCH_SIZES[step] / ((config.PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset_list[step]))
        alpha = min(alpha, 1)

        if batch_idx % quater_batch == 0:
            plot_to_tensorboard(writer,loss_critic.item(),loss_gen.item(),hr_img.detach(),fake_hr.detach(),tensorboard_step)
            tensorboard_step += 1

        loop.set_postfix(loss_gen=loss_gen.item(), loss_critic=loss_critic.item())

    return tensorboard_step, alpha

def main():
    generator = Generator(in_channels=config.CHANNELS_IMG, img_channels=config.CHANNELS_IMG).to(config.DEVICE)
    critic = Critic(in_channels=config.CHANNELS_IMG, img_channels=config.CHANNELS_IMG).to(config.DEVICE)
    
    opt_gen = optim.Adam(generator.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.9))
    scaler_gen = torch.amp.GradScaler(device=config.DEVICE)
    scaler_critic = torch.amp.GradScaler(device=config.DEVICE)
    
    writer = SummaryWriter(config.LOG_PATH)
    
    # Resume training state if available and LOAD_MODEL is True
    if config.LOAD_MODEL:
        epoch, step, alpha, tensorboard_step = load_training_state(generator, critic, opt_gen, opt_critic)
        sys.exit()
        # Manually change
        epoch = 0
        step = 2
    else:
        epoch, step, alpha, tensorboard_step = 0, int(log2(config.START_IMG_SIZE / 128)), 1e-5, 0

    dataset_list = config.DATASET  # A list of dataset objects for each progressive stage.
    
    for num_epochs in config.PROGRESSIVE_EPOCHS[step:]:
        for e in range(epoch, num_epochs):
            print(f"Epoch [{e+1}/{num_epochs}] at step {step}")
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

            if (e + 1) % 5 == 0:
                save_training_state(e, step, alpha, tensorboard_step, generator, critic, opt_gen, opt_critic)
        step += 1
        epoch = 0

if __name__ == "__main__":
    main()
