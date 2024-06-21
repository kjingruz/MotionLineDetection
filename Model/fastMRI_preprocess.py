import pathlib
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import numpy as np
import h5py
import logging
from typing import NamedTuple
from torchvision import transforms

class FastMRIDataset(Dataset):
    """Dataset for loading fastMRI data

    Parameters
    ----------
    path : str
        Path to folder containing the relevant h5 files.
    transform : callable
        A function/transform that takes in an ndarray and returns a transformed version.
    """
    def __init__(self, path, transform=None):
        super().__init__()
        self.path = path
        self.raw_samples = []
        self.transform = transform

        files = list(pathlib.Path(self.path).rglob("*.h5"))
        for filename in sorted(files):
            slices_ind = self._get_slice_indices(filename)

            new_samples = []
            for slice_ind in slices_ind:
                new_samples.append(FastMRIRawDataSample(filename, slice_ind))

            self.raw_samples += new_samples

    def _get_slice_indices(self, filename):
        with h5py.File(filename, "r") as hf:
            slices_ind = np.arange(hf["kspace"].shape[0])
        return slices_ind

    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, idx):
        filename, dataslice = self.raw_samples[idx]

        with h5py.File(filename, "r") as hf:
            kspace = hf["kspace"][dataslice]
            kspace = np.stack((kspace.real, kspace.imag), axis=0)

        if self.transform:
            kspace = self.transform(kspace)

        return torch.as_tensor(kspace, dtype=torch.float32), dataslice


class FastMRILoader(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        akeys = args.keys()
        self.batch_size = args['batch_size'] if 'batch_size' in akeys else 8
        self.transform = args['transform'] if 'transform' in akeys else None
        self.num_workers = args['num_workers'] if 'num_workers' in akeys else 2
        self.data_dir = args['data_dir'] if 'data_dir' in akeys else None
        assert type(self.data_dir) is dict, 'FastMRILoader::init():  data_dir variable should be a dictionary'

    def train_dataloader(self):
        trainset = FastMRIDataset(self.data_dir['train'], transform=self.transform)
        logging.info(f"Size of the train dataset: {trainset.__len__()}.")

        dataloader = DataLoader(
            trainset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        valset = FastMRIDataset(self.data_dir['val'], transform=self.transform)
        logging.info(f"Size of the validation dataset: {valset.__len__()}.")

        dataloader = DataLoader(
            valset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        testset = FastMRIDataset(self.data_dir['test'], transform=self.transform)
        logging.info(f"Size of the test dataset: {testset.__len__()}.")

        dataloader = DataLoader(
            testset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader


class FastMRIRawDataSample(NamedTuple):
    """Generate named tuples consisting of filename and slice index"""
    fname: pathlib.Path
    slice_ind: int


class Norm98:
    def __init__(self, max_val=255.0):
        self.max_val = max_val
        super(Norm98, self).__init__()

    def __call__(self, img):
        q = np.percentile(img, 98)
        img = img / q
        img[img > 1] = 1
        return img


class GaussianNoise:
    def __init__(self, noise_std=0.2, noise_res=16):
        super(GaussianNoise, self).__init__()
        self.noise_std = noise_std
        self.noise_res = noise_res

    def __call__(self, x):
        ns = torch.normal(mean=torch.zeros(x.shape[0], x.shape[1], self.noise_res, self.noise_res),
                          std=self.noise_std).to(x.device)
        ns = torch.nn.functional.upsample_bilinear(ns, size=[x.shape[2], x.shape[3]])
        roll_x = np.random.choice(range(x.shape[2]))
        roll_y = np.random.choice(range(x.shape[3]))
        ns = torch.roll(ns, shifts=[roll_x, roll_y], dims=[-2, -1])
        mask = x.sum(dim=1, keepdim=True) > 0.01
        ns *= mask
        res = x + ns
        return res

# Define the transform pipeline
transform = transforms.Compose([
    Norm98(),
    GaussianNoise()
])

# Define the arguments for the data loader
args = {
    'batch_size': 8,
    'transform': transform,
    'num_workers': 2,
    'data_dir': {
        'train': '../path/to/train_data',
        'val': '../path/to/val_data',
        'test': '../path/to/test_data'
    }
}

# Create the data loader
data_loader = FastMRILoader(args)

# Get the train, validation, and test data loaders
train_loader = data_loader.train_dataloader()
val_loader = data_loader.val_dataloader()
test_loader = data_loader.test_dataloader()

# Example: iterate over the train data loader and print batch shapes
for batch in train_loader:
    kspace, dataslice = batch
    print(kspace.shape, dataslice)
    break
