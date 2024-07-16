import os
import numpy as np
import matplotlib.pyplot as plt
from dataset import T2StarDataset

def save_comparisons(dataset, num_images=5, output_dir='comparison_images'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(num_images):
        idx = np.random.randint(len(dataset))
        try:
            original_image, motion_image, motion_mask, original_kspace, motion_kspace = dataset.get_comparison_data(idx)
            
            fig, axes = plt.subplots(3, 2, figsize=(20, 30))
            
            # Plot original image (middle slice if 3D)
            original_slice = original_image[original_image.shape[0]//2] if original_image.ndim == 3 else original_image
            im = axes[0, 0].imshow(np.abs(original_slice), cmap='gray')
            axes[0, 0].set_title(f'Original Image\nShape: {original_image.shape}')
            axes[0, 0].axis('off')
            plt.colorbar(im, ax=axes[0, 0])
            
            # Plot motion-affected image
            im = axes[0, 1].imshow(np.abs(motion_image), cmap='gray')
            axes[0, 1].set_title(f'Motion-Affected Image\nShape: {motion_image.shape}')
            axes[0, 1].axis('off')
            plt.colorbar(im, ax=axes[0, 1])
            
            # Plot original k-space (log magnitude, middle slice and coil)
            original_kspace_slice = np.log(np.abs(original_kspace[original_kspace.shape[0]//2, original_kspace.shape[1]//2]) + 1)
            im = axes[1, 0].imshow(original_kspace_slice, cmap='viridis')
            axes[1, 0].set_title(f'Original K-space (log magnitude)\nShape: {original_kspace.shape}')
            axes[1, 0].axis('off')
            plt.colorbar(im, ax=axes[1, 0])
            
            # Plot motion-affected k-space (log magnitude, middle slice and coil)
            motion_kspace_slice = np.log(np.abs(motion_kspace[motion_kspace.shape[0]//2, motion_kspace.shape[1]//2]) + 1)
            im = axes[1, 1].imshow(motion_kspace_slice, cmap='viridis')
            axes[1, 1].set_title(f'Motion-Affected K-space (log magnitude)\nShape: {motion_kspace.shape}')
            axes[1, 1].axis('off')
            plt.colorbar(im, ax=axes[1, 1])
            
            # Plot motion mask as a 1D plot
            axes[2, 0].plot(motion_mask[:, 0])
            axes[2, 0].set_title(f'Motion Mask (Two-State)\nShape: {motion_mask.shape}')
            axes[2, 0].set_ylim(-0.1, 1.1)
            axes[2, 0].set_xlabel('k-space line')
            axes[2, 0].set_ylabel('Motion State')
            
            # Plot difference image
            diff_image = np.abs(original_slice - motion_image)
            im = axes[2, 1].imshow(diff_image, cmap='hot')
            axes[2, 1].set_title(f'Difference Image\nShape: {diff_image.shape}')
            axes[2, 1].axis('off')
            plt.colorbar(im, ax=axes[2, 1])
            
            plt.tight_layout()
            
            # Save the figure
            filename = f'comparison_{i+1}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Saved comparison image {i+1} to {filepath}")
        except Exception as e:
            print(f"Error processing image {i+1}: {str(e)}")

def main():
    # Data loading
    data_dir = "../../Data/multicoil_train"  # Adjust this path as needed
    dataset = T2StarDataset(data_dir)
    
    print(f"Total number of files in the dataset: {len(dataset)}")
    
    # Save comparison images
    save_comparisons(dataset, num_images=5, output_dir='comparison_images')

if __name__ == "__main__":
    main()