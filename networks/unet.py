import torch
import torch.nn as nn
from meta import AbstractModel


class Unet(AbstractModel):

    class Encoder(nn.Module):

        def __init__(self, in_channel, out_channel):
            super(Unet.Encoder, self).__init__()
            self._input = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, 3, 1, 1),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )
            self._output = nn.Sequential(
                nn.Conv3d(out_channel, out_channel, 3, 1, 1),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )

        def forward(self, feature_map):
            mid = self._input(feature_map)
            res = self._output(mid)
            return res

    class Decoder(nn.Module):

        def __init__(self, in_channel, out_channel):
            super(Unet.Decoder, self).__init__()
            self._input = nn.Sequential(
                nn.ConvTranspose3d(in_channel, out_channel, 2, stride=2),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )
            self._mid = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, 3, 1, 1),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )
            self._output = nn.Sequential(
                nn.Conv3d(out_channel, out_channel, 3, 1, 1),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )

        def forward(self, feature_map, skip):
            x = self._input(feature_map)
            mid = self._mid(torch.cat([x, skip], dim=1))
            res = self._output(mid)
            return res

    def __init__(self, depth=4, base=16):
        keys = ('pure_phi', )
        super(Unet, self).__init__(keys)
        self.depth = depth
        self._input = Unet.Encoder(1, base)
        self._encoders = nn.ModuleList([nn.Sequential(nn.MaxPool3d(2),
                                        Unet.Encoder(base * 2 ** i, base * 2 ** (i + 1)))
                                        for i in range(depth)])
        self._decoders = nn.ModuleList([Unet.Decoder(base * 2 ** i, base * 2 ** (i - 1))
                                        for i in range(depth, 0, -1)])
        self._output = nn.Conv3d(base, 1, 1, 1, 0)

    def train_model(self, pure_phi):

        x = pure_phi
        skips = []
        inEncoder = self._input(x)
        skips.append(inEncoder)

        for encoder in self._encoders:
            inEncoder = encoder(inEncoder)
            skips.append(inEncoder)

        inDecoder = inEncoder
        skips.pop()
        skips.reverse()

        for decoder, skip in zip(self._decoders, skips):
            inDecoder = decoder(inDecoder, skip)

        return self._output(inDecoder)

    def calc_loss(self, preds, label, crit):
        return crit(preds, label)