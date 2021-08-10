import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class AbstractModel(nn.Module, ABC):

    def __init__(self, keys, mode='train'):
        super(AbstractModel, self).__init__()
        self.keys = keys
        self.mode = mode

    @abstractmethod
    def train_model(self, *args):
        pass

    def calc_loss(self, preds, label, crit):
        loss = torch.tensor(0).to(label.device, torch.float)
        for pred in preds:
            loss += crit(pred, label)
        return loss

    def eval_model(self, *args):
        return self.train_model(*args)

    def __call__(self, *args, **kwargs):

        res = self.train_model(*args) if self.mode is 'train' else self.eval_model(*args)

        return res
