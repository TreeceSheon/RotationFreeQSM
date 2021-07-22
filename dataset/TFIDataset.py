import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
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
            chi_path = self.root_path / fields[0] / 'sus' / ('sus_' + fields[2] + '.nii')
            mask_path = self.root_path / fields[0] / 'sus' / ('mask_' + fields[2] + '.nii')
            self.entries.append({
                'phi': self.root_path / fields[0] / fields[1] /  ('field_' + fields[2] + '.nii'),
                'chi': chi_path,
                'mask': mask_path,
                'rotation': self.root_path / fields[0] / fields[1] / 'rotation.mat',
                'dipole': self.root_path / fields[0] / fields[1] / 'dipole.nii',
                'dtype': data_type
            })

    def __getitem__(self, index):
        pair = self.entries[index]
        phi = torch.from_numpy(nib.load(str(pair['phi'])).get_fdata()[np.newaxis]).to(self.device, torch.float)
        chi = torch.from_numpy(nib.load(str(pair['chi'])).get_fdata()[np.newaxis]).to(self.device, torch.float)
        inv_rot = torch.from_numpy(np.flip(scio.loadmat(pair['rotation'])['inv_mat']).copy()[np.newaxis]).to(self.device, torch.float)
        rot = torch.from_numpy(np.flip(scio.loadmat(pair['rotation'])['mat']).copy()[np.newaxis]).to(self.device, torch.float)
        dipole = torch.from_numpy(pair['dipole']).to(self.device, torch.float)
        data_type = pair['dtype']
        return phi, chi, rot, inv_rot, dipole, data_type

    def __len__(self):
        return len(self.entries)

