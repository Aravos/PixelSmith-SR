import torch
import torch.nn as nn
from torchinfo import summary

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input: N x channels_img x 512 x 512
            nn.Conv2d(channels_img, features_d, 4, 2, 1),                          # 256 x 256
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),                       # 128 x 128
            self._block(features_d * 2, features_d * 4, 4, 2, 1),                   # 64 x 64
            self._block(features_d * 4, features_d * 8, 4, 2, 1),                   # 32 x 32
            self._block(features_d * 8, features_d * 8, 4, 2, 1),                   # 16 x 16
            self._block(features_d * 8, features_d * 8, 4, 2, 1),                   # 8 x 8
            self._block(features_d * 8, features_d * 8, 4, 2, 1),                   # 4 x 4
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),       # 1 x 1
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x 3 x 128 x 128
            nn.Conv2d(channels_img, features_g // 4, 3, 1, 1),           # 128 x 128 x 64
            self._block(features_g // 4, features_g // 2, 3, 1, 1),        # 128 x 128 x 128
            self._block(features_g // 2, features_g, 3, 1, 1),             # 128 x 128 x 256
            nn.PixelShuffle(2),                                           # 256 x 256 x 64
            self._block(features_g // 4, features_g // 2, 3, 1, 1),        # 256 x 256 x 128
            self._block(features_g // 2, features_g, 3, 1, 1),             # 256 x 256 x 256
            nn.PixelShuffle(2),                                           # 512 x 512 x 64
            nn.Conv2d(features_g // 4, channels_img, 3, 1, 1),            # 512 x 512 x 3  
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

if __name__ == "__main__":
    # Example: converting a 128x128 image to 512x512.
    # For a 4x upscaling (128->256->512), we need 2 progressive steps.
    x = torch.randn(1, 3, 128, 128)  # Example input
    generator = Generator(channels_img=3, features_g=256)
    output = generator(x)  # alpha=1.0 means full contribution from the current stage.
    print("Output shape:", output.shape)  # Expected output shape: (1, 3, 512, 512)
    print(summary(generator,input_size=(1, 3, 128, 128),verbose=2))