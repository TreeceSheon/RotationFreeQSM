import argparse
from importlib import import_module
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from networks.hybrid import Hybrid
from pathlib import Path
from dataset.TFIDataset import TFIDataset
import time


def main(args):
    batch_size = args.batch_size
    device = torch.device('cuda')
    path = Path('G:\data\\tfi\\')
    file = 'dataset/data.txt'
    dataloader = DataLoader(TFIDataset(path, file), shuffle=True, batch_size=batch_size, drop_last=True)
    print('training data size: ' + str(len(dataloader) * batch_size))
    if args.hybrid:
        model_name = 'hybrid'
        model_dipole_inv = getattr(import_module('networks.unet'), 'Unet')(4, 16)
        deblur_model = args.deblur_model
        deblur_model = getattr(import_module('networks.' + deblur_model.lower()), deblur_model)
        model = nn.DataParallel(Hybrid(model_dipole_inv, deblur_model, args.joint).to(device))
    else:
        model_name = args.model
        model = getattr(import_module('networks.' + model_name.lower()), model_name)()
        model = nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-6, betas=(0.5, 0.999), eps=1e-9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
    criterion = nn.MSELoss(reduction='sum')

    start_time = time.time()
    model.load_state_dict(torch.load('Unet_20.pkl', map_location='cuda')['model_state'])
    for epoch in range(1, 201):
        for batch_no, values in enumerate(dataloader):
            pure_phi, angled_phi, chi, rot, inv_rot, mask = values
            # pred = model(pure_phi, angled_phi, rot, inv_rot, mask)
            # loss = torch.tensor(0).to(device, torch.float)
            # for res in pred:
            #     loss += criterion(res, chi)
            optimizer.zero_grad()
            pred = model(torch.cat([pure_phi, angled_phi], dim=0)) * torch.cat([mask, mask], dim=0)
            loss = criterion(pred, torch.cat([chi, chi], dim=0))
            loss.backward()
            optimizer.step()
            if batch_no % 40 == 0:
                print({'epoch': epoch, 'batch_no': batch_no, 'lr_rate': optimizer.param_groups[0]['lr'],
                       'loss': loss.item(), 'time': int(time.time() - start_time)})
        scheduler.step()
        if epoch % 10 == 0:
            torch.save({'optim': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'model_state': model.state_dict()}, model_name + '_' + str(epoch) + '.pkl'
                       )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Rotation-free QSM')
    parser.add_argument('--hybrid', action='store_true', default=False)
    parser.add_argument('--joint', action='store_true', default=False)
    parser.add_argument('--model', default='Unet', choices=['Unet', 'LPCNN'])
    parser.add_argument('--deblur-model', default='ResNet', choices=['LPCNN', 'Unet', 'PreNet3D', 'ResNet'])
    parser.add_argument('--batch-size', default=3, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    args = parser.parse_args()

    main(args)