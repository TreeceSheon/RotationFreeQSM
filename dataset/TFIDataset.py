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

    def __init__(self, root_path, file, keys, device=torch.device('cuda')):
        super(TFIDataset, self).__init__()
        self.root_path = root_path
        self.names = get_ids(file)
        self.device = device
        self.keys = keys
        self.label = 'chi'
        self.entries = []
        for index, name in enumerate(self.names):
            fields = name.split(' ')
            chi_path = self.root_path / fields[0] / 'sus' / 'patches' / ('sus_' + fields[2] + '.nii')
            mask_path = self.root_path / fields[0] / 'sus' / 'patches' / ('mask_' + fields[2] + '.nii')
            self.entries.append({
                'pure_phi': self.root_path / fields[0] / 'ori_6_3' / 'patches' / ('field_' + fields[2] + '.nii'),
                'angled_phi': self.root_path / fields[0] / fields[1] / 'patches' / ('field_' + fields[2] + '.nii'),
                'chi': chi_path,
                'mask': mask_path,
                'rotation': self.root_path / fields[0] / fields[1] / 'rotation.mat',
                'pure_dipole': self.root_path / fields[0] / 'ori_6_3' / 'dipole.nii',
                'angled_dipole': self.root_path / fields[0] / fields[1] / 'dipole.nii',
            })

    def __getitem__(self, index):

        items = []
        if self.keys is None:
            raise KeyError('keys for items selection not specified.')
        for key in self.keys:
            items.append(self.select_item(key, index))
        label = self.select_item('chi', index)
        return items, label

    def __len__(self):
        return len(self.entries)

    def select_item(self, key, index):

        if key in ['mat', 'inv_mat']:
            return torch.from_numpy(np.flip(scio.loadmat(self.entries[index]['rotation'])
                                            ['inv_mat']).copy()[np.newaxis]).to(self.device, torch.float)
        else:
            return torch.from_numpy(nib.load(str(self.entries[index]['pure_phi'])).get_fdata()[np.newaxis]).\
                                             to(self.device, torch.float)

