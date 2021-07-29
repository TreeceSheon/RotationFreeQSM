import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class AbstractModel(nn.Module, ABC):

    @abstractmethod
    def forward(self, pured_phi, angled_phi, rot, inv_rot, dipole, mask):
        pass

    def calc_loss(self, preds, label, crit):
        loss = torch.tensor(0).to(label.device, torch.float)
        for pred in preds:
            loss += crit(pred, label)
        return loss
