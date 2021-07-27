import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from pathlib import Path
import scipy.io as scio


def get_ids(file_path):
    f = open(file_path, 'r')
    ids = []
    for line in f:
        ids.append(line.strip("\n"))
    return ids


class TFIDataset(Dataset):

    def __init__(self, root_path, file, device=torch.device('cuda')):
        super(TFIDataset, self).__init__()
        self.root_path = root_path
        self.names = get_ids(file)
        self.device = device
        self.entries = []
        for index, name in enumerate(self.names):
            fields = name.split(' ')
            data_type = 'pure_data' if fields[1].endswith('63') else 'angled_data'
            chi_path = self.root_path / fields[0] / 'sus' / 'patches' / ('sus_' + fields[1] + '.nii')
            mask_path = self.root_path / fields[0] / 'sus' / 'patches' / ('mask_' + fields[1] + '.nii')
            self.entries.append({
                'pure_phi': self.root_path / fields[0] / 'ori_3_3' / 'patches' / ('field_' + fields[1] + '.nii'),
                'angled_phi': self.root_path / fields[0] / 'ori_6_3' / 'patches' / ('field_' + fields[1] + '.nii'),
                'chi': chi_path,
                'mask': mask_path,
                'rotation': self.root_path / fields[0] / 'ori_6_3' / 'rotation.mat',
                'dtype': data_type
            })

    def __getitem__(self, index):

        pair = self.entries[index]
        pure_phi = torch.from_numpy(nib.load(str(pair['pure_phi'])).get_fdata()[np.newaxis]).to(self.device, torch.float)
        angled_phi = torch.from_numpy(nib.load(str(pair['angled_phi'])).get_fdata()[np.newaxis]).to(self.device, torch.float)
        chi = torch.from_numpy(nib.load(str(pair['chi'])).get_fdata()[np.newaxis]).to(self.device, torch.float)
        inv_rot = torch.from_numpy(np.flip(scio.loadmat(pair['rotation'])['inv_mat']).copy()[np.newaxis]).to(self.device, torch.float)
        rot = torch.from_numpy(np.flip(scio.loadmat(pair['rotation'])['mat']).copy()[np.newaxis]).to(self.device, torch.float)
        mask = torch.from_numpy(nib.load(str(pair['mask'])).get_fdata()[np.newaxis]).to(self.device, torch.float)

        return pure_phi, angled_phi, chi, rot, inv_rot, mask

    def __len__(self):
        return len(self.entries)

