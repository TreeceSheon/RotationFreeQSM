import torch.nn as nn


class WideResNet(nn.Module):

    def __init__(self):
        super(WideResNet, self).__init__()
        self.module = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            self.make_layer(wBasicBlock, 8),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):

        return self.module(x)


class wBasicBlock(nn.Module):

    def __init__(self, inplanes=32, planes=32, dropout_rate=0.5):
        super(wBasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout3d(p=dropout_rate)

        self.conv2 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

    def train_model(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out