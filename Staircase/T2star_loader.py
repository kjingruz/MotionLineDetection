import pathlib
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import numpy as np
import h5py
import logging
from typing import NamedTuple
import nibabel as nib
import glob
import os


class T2starDataset(Dataset):
    def __init__(self, path, normalize="line_wise", input_2d=False):
        super().__init__()
        self.path = path
        self.normalize = normalize
        self.input_2d = input_2d
        self.files = [f for f in os.listdir(path) if f.endswith('.h5')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.path, self.files[idx])
        with h5py.File(file_path, 'r') as hf:
            kspace = hf['kspace'][()]
            motion_mask = hf['motion_mask'][()]

        if self.normalize == "line_wise":
            norm = np.sqrt(np.sum(abs(kspace) ** 2, axis=(0, 2))) + 1e-9
            kspace = kspace / norm[None, :, None]

        kspace_realv = np.stack((np.real(kspace), np.imag(kspace)), axis=0)

        if self.input_2d:
            kspace_realv = kspace_realv.reshape((2*kspace.shape[0], *kspace.shape[1:]))

        return torch.as_tensor(kspace_realv).float(), torch.as_tensor(motion_mask).float(), self.files[idx]

    

class T2starLoader(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        akeys = args.keys()
        self.batch_size = args['batch_size'] if 'batch_size' in akeys else 8
        self.only_brainmask_slices = args['only_brainmask_slices'] if 'only_brainmask_slices' in akeys else False
        self.bm_thr = args['bm_thr'] if 'bm_thr' in akeys else 0.1
        self.normalize = args['normalize'] if 'normalize' in akeys else "abs_image"
        self.soft_mask = args['soft_mask'] if 'soft_mask' in akeys else False
        self.subst_with_orig = args['subst_with_orig'] if 'subst_with_orig' in akeys else None
        self.crop_readout = args['crop_readout'] if 'crop_readout' in akeys else False
        self.overfit_one_sample = args['overfit_one_sample'] if 'overfit_one_sample' in akeys else False
        self.magn_phase = args['magn_phase'] if 'magn_phase' in akeys else False
        self.input_2d = args['input_2d'] if 'input_2d' in akeys else False
        self.drop_last = False if self.overfit_one_sample else True
        self.num_workers = args['num_workers'] if 'num_workers' in akeys else 2
        # if 'train_data_path' in akeys:
        #     self.train_data_path = args['train_data_path']
        # else:
        #     logging.info('No training data path specified.')
        # if 'val_data_path' in akeys:
        #     self.val_data_path = args['val_data_path']
        # else:
        #     logging.info('No validation data path specified.')
        # if 'test_data_path' in akeys:
        #     self.test_data_path = args['test_data_path']
        # else:
        #     logging.info('No test data path specified.')
        self.data_dir = args['data_dir'] if 'data_dir' in akeys else None
        assert type(self.data_dir) is dict, 'DefaultDataset::init():  data_dir variable should be a dictionary'


    def train_dataloader(self):
        """Loads a batch of training data consisting of kspace data, target
        mask, filenames and slice indices of the associated h5 files.
        """
        trainset = T2starDataset(self.data_dir['train'],    #path=self.train_data_path,
                                 only_bm_slices=self.only_brainmask_slices,
                                 bm_thr=self.bm_thr,
                                 normalize=self.normalize,
                                 soft_mask=self.soft_mask,
                                 subst_with_orig=self.subst_with_orig,
                                 magn_phase=self.magn_phase,
                                 input_2d=self.input_2d,
                                 crop_readout=self.crop_readout,
                                 overfit_one_sample=self.overfit_one_sample)
        logging.info(f"Size of the train dataset: {trainset.__len__()}.")

        dataloader = DataLoader(
            trainset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=self.drop_last,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        """Loads a batch of validation data consisting of kspace data, target
        mask, filenames and slice indices of the associated h5 files.
        """
        valset = T2starDataset(self.data_dir['val'],    #self.val_data_path,
                               only_bm_slices=self.only_brainmask_slices,
                               bm_thr=self.bm_thr,
                               normalize=self.normalize,
                               soft_mask=self.soft_mask,
                               subst_with_orig=self.subst_with_orig,
                               magn_phase=self.magn_phase,
                               input_2d=self.input_2d,
                               crop_readout=self.crop_readout,
                               overfit_one_sample=self.overfit_one_sample)
        logging.info(f"Size of the validation dataset: {valset.__len__()}.")

        dataloader = DataLoader(
            valset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=self.drop_last,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        """Loads a batch of testing data consisting of kspace data, target
        mask, filenames and slice indices of the associated h5 files. For the
        test data loader 'drop_last' is enabled, which means that no data will
        be loaded if the batch size is larger than the size of the test set.
        """
        testset = T2starDataset(self.data_dir['test'],    #self.test_data_path,
                                only_bm_slices=self.only_brainmask_slices,
                                bm_thr=self.bm_thr,
                                normalize=self.normalize,
                                soft_mask=self.soft_mask,
                                subst_with_orig=self.subst_with_orig,
                                magn_phase=self.magn_phase,
                                input_2d=self.input_2d,
                                crop_readout=self.crop_readout,
                                overfit_one_sample=self.overfit_one_sample)
        logging.info(f"Size of the test dataset: {testset.__len__()}.")

        if testset.__len__() < self.batch_size:
            logging.info('The batch size ({}) is larger than the size of the test set({})! Since the dataloader has '
                         'drop_last enabled, no data will be loaded!'.format(self.batch_size, testset.__len__()))

        dataloader = DataLoader(
            testset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader


class T2StarRawDataSample(NamedTuple):
    """Generate named tuples consisting of filename and slice index"""
    fname: pathlib.Path
    slice_ind: int


def fft2c(x, shape=None, dim=(-2, -1)):
    """Centered Fourier transform"""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=dim), axes=dim, norm='ortho', s=shape), axes=dim)


def load_all_echoes(in_folder, descr, nr_pe_steps=92, nr_echoes=12, offset=2047):
    """Load all echoes for one acquisition into one array as complex dataset

    :param in_folder: input folder
    :param descr: string describing the files to look for
    :param nr_pe_steps: number of phase encoding steps to be simulated.
    :param nr_echoes: number of echoes that are expected. The default is 12.
    :param offset: offset of scanner for saved intensity values. Default for Philips is 2047.
    :return: a complex array of shape [nr_echoes, nr_slices, n_y, n_x]
    :return: an array of shape [4, 4] containing the affine transform for saving as nifti. For saving, please
             remember to apply np.rollaxis of the images to get the shape [n_y, n_x, n_slices]
    """

    files_real = sorted(glob.glob(in_folder+'*real*'+descr))
    files_imag = sorted(glob.glob(in_folder + '*imaginary*' + descr))

    if len(files_real) != nr_echoes:
        print('ERROR: Less than 12 real images found.')
        print(files_real)

    if len(files_imag) != nr_echoes:
        print('ERROR: Less than 12 imaginary images found.')
        print(files_imag)

    shape = np.shape(np.rollaxis(nib.load(files_real[0]).get_fdata(), 2, 0))
    dataset = np.zeros((nr_echoes, shape[0], nr_pe_steps, shape[2]), dtype=complex)

    for re, im, i in zip(files_real, files_imag, range(0, nr_echoes)):
        # load the unscaled nifti (intensities need to match exactly with what scanner saved)
        # subtract offset of 2047 since scanner (Philips) shifted the intensities
        tmp = (np.rollaxis(nib.load(files_real[i]).dataobj.get_unscaled(), 2, 0) - offset) + \
              1j * (np.rollaxis(nib.load(files_imag[i]).dataobj.get_unscaled(), 2, 0) - offset)
        dataset[i] = tmp[:, int((np.shape(tmp)[1]-nr_pe_steps)/2):-int((np.shape(tmp)[1]-nr_pe_steps)/2)]

    affine = nib.load((files_real[0])).affine
    header = nib.load(files_real[0]).header

    return dataset, affine, header
