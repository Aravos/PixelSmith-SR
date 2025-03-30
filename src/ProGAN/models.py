import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

factors = [64, 128, 256]

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
        print("Previous: ", upscaled.shape, final_upscaled.shape)
        print("Current: ", out.shape, final_out.shape)
        return self.fade_in(alpha, final_upscaled, final_out)

if __name__ == "__main__":
    x = torch.randn(1, 3, 128, 128)
    generator = Generator(in_channels=3, img_channels=3)
    output = generator(x, alpha=1.0, steps=2)
    print("Output shape:", output.shape)
    print(summary(
        generator,
        input_size=(1, 3, 128, 128),
        verbose=2,
        alpha=1.0,
        steps=2
        )
    )
