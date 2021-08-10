import argparse
import sys
sys.path.append('networks')
from importlib import import_module
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from networks.hybrid import Hybrid
from networks.separate import Separate
from pathlib import Path
from dataset.TFIDataset import TFIDataset
import time


def main(args):
    batch_size = args.batch_size
    device = torch.device('cuda')
    path = Path('/scratch/itee/uqzxion3/data/QSM/tfi/')
    file = '/scratch/itee/uqzxion3/QSM/rotation/RotationFreeQSM/dataset/data.txt'
    dataloader = DataLoader(TFIDataset(file, device, path), shuffle=True, batch_size=batch_size, drop_last=True)
    print('training data size: ' + str(len(dataloader) * batch_size))
    if args.mode == 'hybrid':
        model_name = 'hybrid'
        if args.joint:
            model_name += '_joint'
        model_name += '_' + args.deblur_model
        model_dipole_inv = getattr(import_module('networks.unet'), 'Unet')(4, 16)
        deblur_model = args.deblur_model
        deblur_model = getattr(import_module('networks.' + deblur_model.lower()), deblur_model)()
        model = nn.DataParallel(Hybrid(model_dipole_inv, deblur_model, args.joint)).to(device)
    elif args.mode == 'whole':
        model_name = args.model
        model = getattr(import_module('networks.' + model_name.lower()), model_name)()
        model = nn.DataParallel(model).to(device)
    elif args.mode == 'separate':
        model_name = args.mode
        model_dipole_inv = nn.DataParallel(getattr(import_module('networks.unet'), 'Unet')(4, 16)).to(device)
        model_dipole_inv.load_state_dict(torch.load('Unet_100.pkl', map_location=device)['model_state'], True)
        deblur_model = args.deblur_model
        model_name += deblur_model
        deblur_model = nn.DataParallel(getattr(import_module('networks.' + deblur_model.lower()), deblur_model)()).to(device)
        model = Separate(model_dipole_inv, deblur_model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_rate, betas=(0.5, 0.999), eps=1e-9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
    criterion = nn.MSELoss(reduction='sum')

    start_time = time.time()
    for epoch in range(1, 201):
        print(epoch)
        for batch_no, (items, chi) in enumerate(dataloader):
            optimizer.zero_grad()
            pred = model(items)
            loss = model.module.calc_loss(pred, label=chi, crit=criterion)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_no % 40 == 0:
                print({'epoch': epoch, 'batch_no': batch_no, 'lr_rate': optimizer.param_groups[0]['lr'],
                       'loss': loss.item(), 'time': int(time.time() - start_time)})
        scheduler.step()
        if epoch % 10 == 0:
            torch.save({'optim': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'model_state': model.state_dict()}, model_name + '_64' + str(epoch) + '.pkl'
                       )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Rotation-free QSM')
    parser.add_argument('--mode', default='hybrid', choices=['hybrid', 'whole', 'separate'])
    parser.add_argument('--joint', action='store_true', default=False)
    parser.add_argument('--model', default='Unet', choices=['Unet', 'LPCNN'])
    parser.add_argument('--deblur-model', default='ResNet', choices=['LPCNN', 'Unet', 'PreNet', 'ResNet'])
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--lr-rate', default=2e-5, type=float)
    args = parser.parse_args()

    main(args)