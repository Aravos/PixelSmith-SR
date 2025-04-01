import torch
import random
import numpy as np
import os
import torchvision
import torch.nn as nn
import config
from torchvision.utils import save_image

# Print losses occasionally and print to tensorboard
def plot_to_tensorboard(writer, loss_critic, loss_gen, real, fake, tensorboard_step):
    writer.add_scalar("Loss Critic", loss_critic, global_step=tensorboard_step)
    writer.add_scalar("Loss Generator", loss_gen, global_step=tensorboard_step)

    with torch.no_grad():
        img_grid_real = torchvision.utils.make_grid(real[:4], normalize=True)
        img_grid_fake = torchvision.utils.make_grid(fake[:4], normalize=True)
        writer.add_image("Real", img_grid_real, global_step=tensorboard_step)
        writer.add_image("Fake", img_grid_fake, global_step=tensorboard_step)


def gradient_penalty(critic, real, fake, alpha, train_step, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, alpha, train_step)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def save_training_state(epoch, step, alpha, tensorboard_step, generator, critic, opt_gen, opt_critic):
    state = {
        "epoch": epoch,
        "step": step,
        "alpha": alpha,
        "tensorboard_step": tensorboard_step,
        "gen_state": generator.state_dict(),
        "critic_state": critic.state_dict(),
        "opt_gen_state": opt_gen.state_dict(),
        "opt_critic_state": opt_critic.state_dict()
    }
    os.makedirs(os.path.dirname(config.CHECKPOINT_STATE), exist_ok=True)
    torch.save(state, config.CHECKPOINT_STATE)
    print("=> Saved training state at epoch", epoch+1)

def load_training_state(generator, critic, opt_gen, opt_critic):
    if os.path.exists(config.CHECKPOINT_STATE):
        state = torch.load(config.CHECKPOINT_STATE, map_location=config.DEVICE)
        generator.load_state_dict(state["gen_state"])
        critic.load_state_dict(state["critic_state"])
        opt_gen.load_state_dict(state["opt_gen_state"])
        opt_critic.load_state_dict(state["opt_critic_state"])
        print("=> Resuming training from epoch", state["epoch"], "step", state["step"])
        return state["epoch"], state["step"], state["alpha"], state["tensorboard_step"]
    # If no checkpoint exists, start with default values.
    default_epoch = 0
    default_step = int(log2(config.START_IMG_SIZE / 128))
    default_alpha = 1e-5
    default_tb_step = 0
    return default_epoch, default_step, default_alpha, default_tb_step