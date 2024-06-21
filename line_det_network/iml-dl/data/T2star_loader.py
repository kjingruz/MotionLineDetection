import numpy as np
from torch.utils.data import Dataset

class T2starLoader(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = np.load(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample
