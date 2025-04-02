import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2

factors = [32, 64]

class WSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
        super(WSConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (in_channels * (kernel_size ** 2))) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)

class UpBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super(UpBlock, self).__init__()
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.leaky(self.conv2(x))
        return x

class Generator(nn.Module):
    def __init__(self, in_channels, img_channels=3):
        super(Generator, self).__init__()

        self.initial = WSConv2d(in_channels, factors[0], kernel_size=1, stride=1, padding=0)
        self.initial_rgb = WSConv2d(factors[0], img_channels, kernel_size=1, stride=1, padding=0)

        self.prog_blocks = nn.ModuleList([])
        self.rgb_layers = nn.ModuleList([self.initial_rgb])

        for i in range(len(factors) - 1):
            conv_in_c = int(factors[i])
            conv_out_c = int(factors[i + 1])
            self.prog_blocks.append(UpBlock(conv_in_c, conv_out_c))
            self.rgb_layers.append(WSConv2d(conv_out_c, img_channels, kernel_size=1, stride=1, padding=0))

    def fade_in(self, alpha, upscaled, generated):
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

    def forward(self, x, alpha, steps):
        out = self.initial(x)
        
        if steps == 0:
            return self.initial_rgb(out)

        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode="nearest")
            out = self.prog_blocks[step](upscaled)

        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        final_out = self.rgb_layers[steps](out)
        return self.fade_in(alpha, final_upscaled, final_out)

class Critic(nn.Module):
    def __init__(self, in_channels, img_channels=3):
        super(Critic, self).__init__()
        
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.prog_blocks = nn.ModuleList([])
        self.rgb_layers = nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)

        for i in range(len(factors) - 1, 0, -1):
            conv_in = int(factors[i])
            conv_out = int(factors[i - 1])
            self.prog_blocks.append(UpBlock(conv_in, conv_out))
            self.rgb_layers.append(WSConv2d(img_channels, conv_in, kernel_size=1, stride=1, padding=0))

        self.initial_rgb = WSConv2d(img_channels, factors[0], kernel_size=1, stride=1, padding=0)
        self.rgb_layers.append(self.initial_rgb)

        self.final_block = nn.Sequential(
            WSConv2d(factors[0] + 1, factors[0] // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            WSConv2d(factors[0] // 2, factors[0] // 4, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            WSConv2d(factors[0] // 4, 1, kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1)
        )


    def fade_in(self, alpha, downscaled, out):
        return alpha * out + (1 - alpha) * downscaled

    def minibatch_std(self, x):
        batch_statistics = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x, alpha, steps):
        cur_step = len(self.prog_blocks) - steps
        out = self.leaky(self.rgb_layers[cur_step](x))

        if steps == 0:
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)

        downscaled = self.leaky(self.rgb_layers[cur_step + 1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[cur_step](out))
        out = self.fade_in(alpha, downscaled, out)

        for step in range(cur_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)
        # print("Output shape after mini batch before final: ", out.shape)
        return self.final_block(out).view(out.shape[0], -1)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for img_size in [128, 256]:

        num_steps = int(log2(img_size / 128))

        x = torch.randn(1, 3, 128, 128).to(device)
        generator = Generator(in_channels=3, img_channels=3).to(device)
        
        # autocast to run in FP16 precision 
        with torch.amp.autocast(enabled=True, dtype=torch.float16, device_type=device):
            output = generator(x, alpha=1.0, steps=num_steps)
        
        expected_shape = (1, 3, img_size, img_size)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        critic = Critic(in_channels=3, img_channels=3).to(device)
        with torch.amp.autocast(enabled=True, dtype=torch.float16, device_type=device):
            disc_out = critic(output, alpha=1.0, steps=num_steps)
        
        assert disc_out.shape == (1, 1), f"Expected critic output shape (1,1), got {disc_out.shape}"
        
        print(f"Success for {img_size}")