import torch
import torch.nn as nn
from generator import GLU

# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()

#         self.convLayer1 = nn.Sequential(nn.Conv2d(in_channels=1,
#                                                   out_channels=128,
#                                                   kernel_size=(3, 3),
#                                                   stride=(1, 1),
#                                                   padding=(1, 1)),
#                                         GLU())

#         # DownSample Layer
#         self.downSample1 = self.downSample(in_channels=128,
#                                            out_channels=256,
#                                            kernel_size=(3, 3),
#                                            stride=(2, 2),
#                                            padding=1)

#         self.downSample2 = self.downSample(in_channels=256,
#                                            out_channels=512,
#                                            kernel_size=(3, 3),
#                                            stride=[2, 2],
#                                            padding=1)

#         self.downSample3 = self.downSample(in_channels=512,
#                                            out_channels=1024,
#                                            kernel_size=[6, 3],
#                                            stride=[2, 2],
#                                            padding=1)

#         # Conv Layer
#         self.outputConvLayer = nn.Sequential(nn.Conv2d(in_channels=1024,
#                                                        out_channels=1,
#                                                        kernel_size=(1, 3),
#                                                        stride=[1, 1],
#                                                        padding=[0, 1]))

#     def downSample(self, in_channels, out_channels, kernel_size, stride, padding):
#         convLayer = nn.Sequential(nn.Conv2d(in_channels=in_channels,
#                                             out_channels=out_channels,
#                                             kernel_size=kernel_size,
#                                             stride=stride,
#                                             padding=padding),
#                                   nn.InstanceNorm2d(num_features=out_channels,
#                                                     affine=True),
#                                   GLU())
#         return convLayer

#     def forward(self, input):
#         input = input.unsqueeze(1)
#         conv_layer_1 = self.convLayer1(input)

#         downsample1 = self.downSample1(conv_layer_1)
#         downsample2 = self.downSample2(downsample1)
#         downsample3 = self.downSample3(downsample2)

#         output = torch.sigmoid(self.outputConvLayer(downsample3))
#         return output

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(Block(in_channels, feature, stride=1 if feature==features[-1] else 2))
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=(4,4), stride=1, padding=1, padding_mode="reflect"))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        return torch.sigmoid(self.model(x))