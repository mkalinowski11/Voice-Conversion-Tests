import torch
import torch.nn as nn

# GLU and PixelShuffle layers for future tests
class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)


class PixelShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        n = input.shape[0]
        c_out = input.shape[1] // 2
        w_new = input.shape[2] * 2
        return input.view(n, c_out, w_new)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_act, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,**kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace = True) if use_act else nn.Identity()
        )
    def forward(self, x):
        return self.conv(x)

class ConvBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace = True)
        )
    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, use_act = True, kernel_size = 3, padding=(1,1)),
            ConvBlock(channels, channels, use_act = False, kernel_size = 3, padding=(1,1))
        )
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, channels, num_features = 64, num_residuals = 9):
        super().__init__()
        self.initial = nn.Sequential(
            ConvBlock(in_channels = channels, out_channels = num_features, use_act = True, kernel_size = 7, stride = 1, padding = (3,3), padding_mode = 'reflect')
        )
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(in_channels = num_features, out_channels = 2*num_features, use_act = True, kernel_size = 3, stride = 2, padding = (1,1), padding_mode = "reflect"),
                ConvBlock(in_channels = 2*num_features, out_channels = 4*num_features, use_act = True, kernel_size = 3, stride = 2, padding = (1,1), padding_mode = "reflect")
            ]
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels = 4*num_features) for _ in range(num_residuals)]
        )
        self.up_blocks = nn.ModuleList(
            [
                ConvBlockUp(in_channels = 4*num_features, out_channels = 2*num_features, kernel_size = 3, stride = 2, padding = (1,1)),
                ConvBlockUp(in_channels = 2*num_features, out_channels = num_features, kernel_size = 3, stride = 2, padding = (1,1))
            ]
        )
        self.out_layer = nn.Conv2d(in_channels = num_features, out_channels = channels, kernel_size=4, stride=1, padding=(3,3), padding_mode="reflect")
  
    def forward(self, x):
        x = self.initial(x)
        for down_block in self.down_blocks:
            x = down_block(x)
        x = self.res_blocks(x)
        for up_block in self.up_blocks:
            x = up_block(x)
        return self.out_layer(x)