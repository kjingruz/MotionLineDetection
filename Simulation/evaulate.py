import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import T2StarDataset
from model import LineDetectionNet

def evaluate(model, test_loader, device):
    model.eval()
    all_preds = []
    all_masks = []
    with torch.no_grad():
        for image, mask, slice_mask in test_loader:
            image, mask, slice_mask = image.to(device), mask.to(device), slice_mask.to(device)
            outputs = model(image, slice_mask)
            all_preds.extend(outputs.cpu().numpy())
            all_masks.extend(mask.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_masks = np.array(all_masks)
    
    accuracy = np.mean((all_preds > 0.5) == all_masks)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Visualize a sample prediction
    idx = np.random.randint(len(all_preds))
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(311)
    plt.plot(all_masks[idx])
    plt.title('True Mask')
    plt.ylim(-0.1, 1.1)
    
    plt.subplot(312)
    plt.plot(all_preds[idx])
    plt.title('Predicted Mask')
    plt.ylim(-0.1, 1.1)
    
    plt.subplot(313)
    plt.plot((all_preds[idx] > 0.5).astype(float))
    plt.title('Thresholded Prediction')
    plt.ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.show()
    
    # Print shapes for debugging
    print(f"Mask shape: {all_masks[idx].shape}")
    print(f"Prediction shape: {all_preds[idx].shape}")
    
    # Print some statistics
    print(f"Mask min: {all_masks[idx].min()}, max: {all_masks[idx].max()}")
    print(f"Prediction min: {all_preds[idx].min()}, max: {all_preds[idx].max()}")
    
    # Add these lines after the existing code
    thresholded_pred = (all_preds[idx] > 0).astype(int)
    motion_affected_lines = np.where(thresholded_pred == 1)[0]
    
    print(f"Number of motion-affected lines: {len(motion_affected_lines)}")
    print(f"Indices of motion-affected lines: {motion_affected_lines}")

    # Compare with true mask
    true_motion_lines = np.where(all_masks[idx] == 1)[0]
    print(f"Number of true motion-affected lines: {len(true_motion_lines)}")
    print(f"Indices of true motion-affected lines: {true_motion_lines}")

    # Calculate precision and recall
    true_positives = np.intersect1d(motion_affected_lines, true_motion_lines)
    precision = len(true_positives) / len(motion_affected_lines) if len(motion_affected_lines) > 0 else 0
    recall = len(true_positives) / len(true_motion_lines) if len(true_motion_lines) > 0 else 0

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    target_size = (640, 320)
    max_slices = 20
    
    test_dir = "../../Data/multicoil_test"
    
    test_dataset = T2StarDataset(test_dir, target_size=target_size, max_slices=max_slices)
    test_loader = DataLoader(test_dataset, batch_size=4)
    
    model = LineDetectionNet(input_channels=1, max_slices=max_slices, input_size=target_size, output_size=640).to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    
    evaluate(model, test_loader, device)

if __name__ == "__main__":
    main()