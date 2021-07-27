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
    print(args.is_hybrid)
    batch_size = args.batch_size
    device = torch.device('cuda')
    path = Path('/scratch/itee/uqlguo3/data/')
    file = 'augmented_data.txt'
    # dataloader = DataLoader(TFIDataset(path, file), shuffle=True, batch_size=batch_size, drop_last=True)
    # print('training data size: ' + str(len(dataloader) * batch_size))
    if args.is_hybrid:
        model_name = 'hybrid'
        model_dipole_inv = getattr(import_module('networks.unet'), 'Unet')(4, 16)
        deblur_model = args.deblur_model
        deblur_model = getattr(import_module('network.' + deblur_model.lower()), deblur_model)
        model = nn.DataParallel(Hybrid(model_dipole_inv, deblur_model).to(device))
    else:
        model_name = args.model
        model = nn.DataParallel(eval(args.model)())
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-6, betas=(0.5, 0.999), eps=1e-9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
    criterion = nn.MSELoss(reduction='sum')

    start_time = time.time()

    for epoch in range(1, 201):
        for batch_no, values in enumerate(dataloader):
            phi, chi, rot, inv_rot, data_type = values
            pred = model(phi, rot, inv_rot, data_type)
            loss = criterion(pred, chi)
            loss.backward()
            optimizer.step()
            if batch_no % 40 == 0:
                print({'epoch': epoch, 'batch_no': batch_no, 'lr_rate': optimizer.param_groups[0]['lr'],
                       'loss': loss.item(), 'time': int(time.time() - start_time)})
        scheduler.step()
        if epoch % 50 == 0:
            torch.save({'optim': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'model_state': model.state_dict()}, model_name + '_' + str(epoch) + '.pkl'
                       )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Rotation-free QSM')
    parser.add_argument('--is-hybrid', action='store_true', default=False)
    parser.add_argument('--model', default='Unet', choices=['Unet', 'LPCNN'])
    parser.add_argument('--deblur-model', default='ResNet', choices=['LPCNN', 'Unet', 'PreNet3D', 'ResNet'])
    parser.add_argument('--batch-size', default=24, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    args = parser.parse_args()

    main(args)