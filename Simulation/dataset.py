import os
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
from torchvision import transforms

class T2StarDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_size=(640, 320), max_slices=20, motion_intensity=0.5):
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
        self.transform = transform
        self.target_size = target_size
        self.max_slices = max_slices
        self.motion_intensity = motion_intensity

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        with h5py.File(file_path, 'r') as hf:
            kspace = hf['kspace'][()]
        
        motion_kspace, motion_mask = self.add_motion_artifact(kspace)
        image = self.reconstruct_image(motion_kspace)
        
        image_tensor = torch.from_numpy(image).float()
        mask_tensor = torch.from_numpy(motion_mask).float()
        
        # Resize image_tensor if necessary
        if image_tensor.shape[-2:] != self.target_size:
            resize = transforms.Resize(self.target_size)
            image_tensor = resize(image_tensor)
        
        # Resize mask_tensor to match target_size[0]
        mask_tensor = torch.nn.functional.interpolate(mask_tensor.unsqueeze(0).unsqueeze(0), 
                                                      size=(self.target_size[0],), 
                                                      mode='nearest').squeeze()
        
        # Add channel dimension if it's not present
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Pad or crop to max_slices
        current_slices = image_tensor.shape[1]
        if current_slices < self.max_slices:
            padding = torch.zeros((image_tensor.shape[0], self.max_slices - current_slices, *self.target_size))
            image_tensor = torch.cat([image_tensor, padding], dim=1)
            slice_mask = torch.cat([torch.ones(current_slices), torch.zeros(self.max_slices - current_slices)])
        else:
            image_tensor = image_tensor[:, :self.max_slices]
            slice_mask = torch.ones(self.max_slices)
        
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        return image_tensor, mask_tensor, slice_mask

    def add_motion_artifact(self, kspace):
        num_slices, num_coils, num_lines, num_columns = kspace.shape
        motion_mask = np.zeros(num_lines)
        artifact_start = num_lines // 2 - num_lines // 20
        artifact_end = num_lines // 2 + num_lines // 20
        motion_mask[artifact_start:artifact_end] = 1
        
        motion_kspace = kspace.copy()
        for s in range(num_slices):
            for c in range(num_coils):
                for i in range(artifact_start, artifact_end):
                    random_phase = np.exp(1j * 2 * np.pi * np.random.rand())
                    motion_kspace[s, c, i, :] *= (1 + self.motion_intensity * random_phase)
        
        return motion_kspace, motion_mask

    def reconstruct_image(self, kspace):
        kspace_torch = torch.from_numpy(kspace).to(torch.complex64)
        image_complex = self.ifft2c(kspace_torch)
        image_magnitude = self.complex_abs(image_complex)
        image = self.rss(image_magnitude, dim=1)
        return image.numpy()

    def ifft2c(self, kspace):
        return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(kspace, dim=(-2, -1)), dim=(-2, -1)), dim=(-2, -1))

    def complex_abs(self, data):
        return torch.sqrt(data.real**2 + data.imag**2)

    def rss(self, data, dim=0):
        return torch.sqrt(torch.sum(data**2, dim=dim))