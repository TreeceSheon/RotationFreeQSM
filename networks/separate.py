import torch
import torch.nn as nn
from networks.meta import AbstractModel
from handy.py_rotate.rotate import rotate


class Separate(AbstractModel):

    def __init__(self, dipole_inv_model, deblurring_model, is_joint=True, mode='train'):
        keys = ['pure_phi', 'rot', 'inv_rot', 'mask']
        super(Separate, self).__init__(keys, mode)
        self.model1 = dipole_inv_model
        self.model2 = deblurring_model
        self.joint_training = is_joint

    def train_model(self, values):
        angled_phi, rot, inv_rot, mask = values
        self.model1.eval()
        with torch.no_grad():
            pred1 = rotate(self.model1(rotate(angled_phi, rot)), inv_rot) * mask
        pred2 = self.model2(pred1) * mask
        return pred1, pred2