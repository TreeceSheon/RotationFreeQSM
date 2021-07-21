from unet import Unet
from resnet import DeepResNet
import torch.nn as nn
from handy.py_rotate import rotate


class Hybrid(nn.Module):

    def __init__(self):
        super(Hybrid, self).__init__()
        self.unet = Unet(4, 16)
        self.res_net = DeepResNet(1, 1)

    def forward(self, phi, rot, inv_rot, d_type):
        if d_type != 'pure':
            phi = rotate(phi, rot)
        pred = self.unet(phi)
        if d_type != 'pure':
            pred = rotate(pred, inv_rot)
            pred = self.res_net(pred)
        return pred
