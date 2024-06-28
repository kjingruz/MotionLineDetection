import os
import h5py
import numpy as np
from tqdm import tqdm
from scipy.ndimage import affine_transform

def simulate_motion(kspace, max_displacement=2.0, max_rotation=0.1):
    """Simulate motion artifacts in k-space"""
    num_lines = kspace.shape[-2]
    motion_params = np.random.rand(num_lines, 3)  # [dx, dy, rotation]
    motion_params[:, 0] = motion_params[:, 0] * 2 * max_displacement - max_displacement
    motion_params[:, 1] = motion_params[:, 1] * 2 * max_displacement - max_displacement
    motion_params[:, 2] = motion_params[:, 2] * 2 * max_rotation - max_rotation

    motion_mask = np.zeros(num_lines, dtype=bool)
    for i in range(num_lines):
        if np.linalg.norm(motion_params[i, :2]) > 0.5 or abs(motion_params[i, 2]) > 0.05:
            kspace[..., i, :] *= np.exp(1j * 2 * np.pi * (
                motion_params[i, 0] * np.arange(kspace.shape[-1]) / kspace.shape[-1] +
                motion_params[i, 1] * i / num_lines
            ))
            kspace[..., i, :] = affine_transform(kspace[..., i, :].real, 
                                                 [[np.cos(motion_params[i, 2]), -np.sin(motion_params[i, 2])],
                                                  [np.sin(motion_params[i, 2]), np.cos(motion_params[i, 2])]])
            motion_mask[i] = True

    return kspace, motion_mask

def preprocess_fastmri_batch(input_dir, output_dir, start_idx, end_idx):
    """Preprocess a batch of FastMRI data files"""
    os.makedirs(output_dir, exist_ok=True)
    
    file_list = [f for f in os.listdir(input_dir) if f.endswith('.h5')]
    file_list = file_list[start_idx:end_idx]
    
    for idx, filename in enumerate(tqdm(file_list, desc=f"Processing files {start_idx}-{end_idx}")):
        with h5py.File(os.path.join(input_dir, filename), 'r') as hf:
            kspace = hf['kspace'][()]
        
        num_slices, num_coils, height, width = kspace.shape
        
        for slice_idx in range(num_slices):
            slice_kspace = kspace[slice_idx]
            motion_affected_kspace, motion_mask = simulate_motion(slice_kspace)
            kspace_magnitude = np.abs(motion_affected_kspace)
            kspace_normalized = (kspace_magnitude - np.min(kspace_magnitude)) / (np.max(kspace_magnitude) - np.min(kspace_magnitude))
            
            kspace_real = np.real(motion_affected_kspace).astype(np.float32)
            kspace_imag = np.imag(motion_affected_kspace).astype(np.float32)
            
            output_filename = f"{start_idx + idx:06d}_{slice_idx:03d}.npz"
            np.savez_compressed(os.path.join(output_dir, output_filename),
                                kspace_real=kspace_real,
                                kspace_imag=kspace_imag,
                                kspace_normalized=kspace_normalized,
                                motion_mask=motion_mask)

def preprocess_dataset(input_dir, output_dir, batch_size=5000):
    """Preprocess all FastMRI data files in a dataset"""
    os.makedirs(output_dir, exist_ok=True)
    file_list = [f for f in os.listdir(input_dir) if f.endswith('.h5')]
    total_files = len(file_list)
    
    print(f"Processing {total_files} files from {input_dir} to {output_dir}")
    
    for start_idx in range(0, total_files, batch_size):
        end_idx = min(start_idx + batch_size, total_files)
        preprocess_fastmri_batch(input_dir, output_dir, start_idx, end_idx)
        print(f"Completed batch {start_idx//batch_size + 1} of {(total_files-1)//batch_size + 1}")

# Main execution
if __name__ == "__main__":
    # Define directories
    base_input_dir = "../../Data"
    base_output_dir = "./"

    datasets = {
        "valid": ("multicoil_val", "Valid_data"),
        "test": ("multicoil_test", "Test_data")
    }

    for dataset, (input_folder, output_folder) in datasets.items():
        input_dir = os.path.join(base_input_dir, input_folder)
        output_dir = os.path.join(base_output_dir, output_folder)
        
        print(f"\nProcessing {dataset} dataset:")
        preprocess_dataset(input_dir, output_dir)

    print("\nAll datasets have been preprocessed.")
