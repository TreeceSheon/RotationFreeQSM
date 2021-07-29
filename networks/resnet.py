import torch
import torch.nn as nn


class ResNet(nn.Module):

    def __init__(self, num_Ch=16, HG_depth=4):
        super(ResNet, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv3d(1, num_Ch, 3, padding=1),
            nn.BatchNorm3d(num_Ch),
            nn.ReLU(inplace=True),
        )

        self.deepresnet = HG_depth
        self.MidLayers = []

        temp = list(range(1, HG_depth + 1))

        for encodingLayer in temp:

            self.MidLayers.append(HG_block(num_Ch))

        self.MidLayers = nn.ModuleList(self.MidLayers)

        self.output_layer = nn.Conv3d(num_Ch, 1, 1, stride=1, padding=0)

    def forward(self, x):

        INPUT = x
        x = self.input_layer(x)
        temp = list(range(1, self.deepresnet + 1))
        for encodingLayer in temp:
            temp_conv = self.MidLayers[encodingLayer - 1]
            x = temp_conv(x)
        x = self.output_layer(x)
        x = x + INPUT

        return x


class HG_block(nn.Module):

    def __init__(self, num_Ch):
        super(HG_block, self).__init__()
        self.conv_res = nn.Sequential(
            nn.Conv3d(num_Ch, num_Ch, 3, padding=1),
            nn.BatchNorm3d(num_Ch),
            nn.ReLU(inplace=True),
        )
        self.conv_for = nn.Sequential(
            nn.Conv3d(num_Ch, num_Ch, 3, padding=1),
            nn.BatchNorm3d(num_Ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        INPUT = x
        x = self.conv_res(x)
        x = x + INPUT
        x = self.conv_for(x)
        return x