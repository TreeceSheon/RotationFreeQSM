import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from networks.hybrid import Hybrid
from networks.unet import Unet
from networks.resnet import DeepResNet
from pathlib import Path
from dataset.TFIDataset import TFIDataset
import time

batch_size = 24
model_name = 'Hybrid'
device = torch.device('cuda')
path = Path('/scratch/itee/uqlguo3/data/')
file = 'augmented_data.txt'
dataloader = DataLoader(TFIDataset(path, file), shuffle=True, batch_size=batch_size, drop_last=True)
print('training data size: ' + str(len(dataloader) * batch_size))

model = nn.DataParallel(Hybrid(Unet(4, 16), DeepResNet(1, 1)).to(device))
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
