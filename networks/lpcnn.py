import torch.nn as nn
import torch
import torch.fft as fft
import torch.nn.functional as F
from meta import AbstractModel


class wBasicBlock(nn.Module):

    def __init__(self, inplanes=32, planes=32, dropout_rate=0.5):
        super(wBasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout3d(p=dropout_rate)

        self.conv2 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)


class LPCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.iter_num = 3

        self.alpha = torch.nn.Parameter(torch.ones(1) * 4)

        self.gen = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            self.make_layer(wBasicBlock, 8),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    # def forward(self, init_chi, y, dk, mask):
    def forward(self, pure_phi, angled_phi, rot, inv_rot, dipole, mask):
        yy = (pure_phi, angled_phi)
        res = []
        for i in range(2):
            y = yy[i]
            dk = dipole[i]
            batch_size, _, x_dim, y_dim, z_dim, = y.shape

            dim1 = dk.shape[2]
            dim2 = dk.shape[3]
            dim3 = dk.shape[4]
            temp = dk * fft.fftn(
                F.pad(y, (0, dim3 - z_dim, 0, dim2 - y_dim, 0, dim1 - x_dim)), dim=[2, 3, 4])
            x_est = self.alpha * torch.real(fft.ifftn(temp)[:, :, :x_dim, :y_dim, :z_dim])
            # x_est += init_chi - self.alpha * torch.real(fft.ifftn(
            #             dk * dk * fft.fftn(F.pad(init_chi, (
            #                 0, dim3 - z_dim, 0, dim2 - y_dim, 0,
            #                 dim1 - x_dim)), dim=[2, 3, 4]), dim=[2, 3, 4]))[:, :, :x_dim, :y_dim, :z_dim]
            for i in range(self.iter_num):
                if i == 0:
                    pn_x_pred = x_est
                else:
                    pn_x_pred = den_x_pred
                    pn_x_pred += x_est - self.alpha * torch.real(fft.ifftn(
                        dk * dk * fft.fftn(F.pad(den_x_pred, (
                            0, dim3 - z_dim, 0, dim2 - y_dim, 0,
                            dim1 - x_dim)), dim=[2, 3, 4]), dim=[2, 3, 4]))[:, :, :x_dim, :y_dim, :z_dim]

                x_input = pn_x_pred * mask
                x_pred = self.gen(x_input)
                den_x_pred = x_pred * mask
            res.append(x_pred)
        return torch.cat(res, dim=0)