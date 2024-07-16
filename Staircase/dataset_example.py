import os
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.ndimage import zoom


class T2StarDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.h5')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        with h5py.File(file_path, 'r') as hf:
            kspace = hf['kspace'][()]
        return kspace, self.file_list[idx]

    def get_comparison_data(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        with h5py.File(file_path, 'r') as hf:
            kspace = hf['kspace'][()]
            print(f"Original kspace shape: {kspace.shape}")
            
            print(f"Available keys in the h5 file: {list(hf.keys())}")

        original_image = self.reconstruct_image(kspace)
        print(f"Reconstructed original image. Shape: {original_image.shape}")

        motion_kspace, motion_mask = self.add_motion_artifact(kspace)
        motion_image = self.reconstruct_image(motion_kspace)
        print(f"Reconstructed motion-affected image. Shape: {motion_image.shape}")

        # Debug: Check reconstruction
        original_energy = np.sum(original_image**2)
        motion_energy = np.sum(motion_image**2)
        print(f"Original image energy: {original_energy}")
        print(f"Motion-affected image energy: {motion_energy}")
        print(f"Image energy difference: {motion_energy - original_energy}")

        # Ensure motion_image has the same shape as original_image
        if motion_image.shape != original_image.shape:
            motion_image = self.resize_image(motion_image, original_image.shape)
            print(f"Resized motion image to: {motion_image.shape}")

        # Reshape motion_mask to 2D
        motion_mask_2d = np.repeat(motion_mask[:, np.newaxis], original_image.shape[1], axis=1)

        print(f"Original image shape: {original_image.shape}")
        print(f"Motion image shape: {motion_image.shape}")
        print(f"Motion mask shape: {motion_mask_2d.shape}")

        return original_image, motion_image, motion_mask_2d, kspace, motion_kspace

    def reconstruct_image(self, kspace):
        print(f"Reconstruct_image input shape: {kspace.shape}")
        if kspace.ndim == 4:  # (coils, slices, height, width)
            image = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(kspace, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))
            image = np.sqrt(np.sum(np.abs(image)**2, axis=0))
            image = image[image.shape[0]//2]  # Return middle slice
        elif kspace.ndim == 3:  # (slices, height, width) or (coils, height, width)
            image = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(kspace, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))
            image = np.abs(image[image.shape[0]//2])  # Return middle slice or coil
        elif kspace.ndim == 2:  # (height, width)
            image = np.abs(np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(kspace))))
        else:
            raise ValueError(f"Unexpected kspace shape: {kspace.shape}")
        
        # Scale the image to [0, 1] range while preserving energy
        image_energy = np.sum(image**2)
        image_scaled = image / np.max(image)
        image_scaled *= np.sqrt(image_energy / np.sum(image_scaled**2))
        
        print(f"Reconstructed image shape: {image_scaled.shape}, min: {np.min(image_scaled)}, max: {np.max(image_scaled)}, mean: {np.mean(image_scaled)}")
        return image_scaled

    def add_motion_artifact(self, kspace):
        print(f"Add_motion_artifact input shape: {kspace.shape}")
        num_coils, num_slices, num_lines, num_columns = kspace.shape

        # Initialize motion mask
        motion_mask = np.zeros(num_lines)
        
        # Choose a random line where motion occurs
        motion_line = np.random.randint(num_lines // 2, num_lines)
        motion_mask[motion_line:] = 1

        # Generate random motion parameters
        translation = np.random.uniform(-2, 2, 3)  # dx, dy, dz in mm
        rotation = np.random.uniform(-2, 2, 3) * np.pi / 180  # theta_x, theta_y, theta_z in radians

        print(f"Motion occurs at line: {motion_line}")
        print(f"Translation (mm): {translation}")
        print(f"Rotation (degrees): {rotation * 180 / np.pi}")

        motion_kspace = kspace.copy()
        for c in range(num_coils):
            for s in range(num_slices):
                for i in range(motion_line, num_lines):
                    # Apply translation
                    phase_shift = 2 * np.pi * (
                        translation[0] * np.arange(num_columns) / num_columns +
                        translation[1] * i / num_lines +
                        translation[2] * s / num_slices
                    )
                    motion_kspace[c, s, i, :] *= np.exp(1j * phase_shift)
                    
                    # Apply rotation (simplified, only considering in-plane rotation)
                    rotation_matrix = np.array([
                        [np.cos(rotation[2]), -np.sin(rotation[2])],
                        [np.sin(rotation[2]), np.cos(rotation[2])]
                    ])
                    coordinates = np.array(np.meshgrid(np.arange(num_columns), i)).T.reshape(-1, 2)
                    rotated_coordinates = np.dot(coordinates - np.array([num_columns/2, num_lines/2]), rotation_matrix.T) + np.array([num_columns/2, num_lines/2])
                    motion_kspace[c, s, i, :] = np.interp(rotated_coordinates[:, 0], np.arange(num_columns), motion_kspace[c, s, i, :])

        # Debug: Check if motion has been applied
        original_energy = np.sum(np.abs(kspace)**2)
        motion_energy = np.sum(np.abs(motion_kspace)**2)
        print(f"Original k-space energy: {original_energy}")
        print(f"Motion-affected k-space energy: {motion_energy}")
        print(f"Energy difference: {motion_energy - original_energy}")

        return motion_kspace, motion_mask

    def resize_image(self, image, target_shape):
        # Use scipy's zoom function for resizing
        zoom_factors = np.array(target_shape) / np.array(image.shape)
        return zoom(image, zoom_factors, order=1)

    def ifft2c(self, kspace):
        return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(kspace, dim=(-2, -1)), dim=(-2, -1)), dim=(-2, -1))

    def complex_abs(self, data):
        return torch.sqrt(data.real**2 + data.imag**2)

    def rss(self, data, dim=0):
        return torch.sqrt(torch.sum(data**2, dim=dim))