import torch.nn as nn
from handy.py_rotate.rotate import rotate


class Hybrid(nn.Module):

    def __init__(self, dipole_inv_model, deblurring_model):
        super(Hybrid, self).__init__()
        self.model1 = dipole_inv_model
        self.model2 = deblurring_model

    def forward(self, phi, rot, inv_rot, d_type):
        if d_type != 'pure':
            phi = rotate(phi, rot)
        pred = self.model1(phi)
        if d_type != 'pure':
            pred = rotate(pred, inv_rot)
            pred = self.model2(pred)
        return pred
