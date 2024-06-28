import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import os

class LineDetectionNet(nn.Module):
    def __init__(self, input_size=(640, 320)):
        super(LineDetectionNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate the size after convolutions and pooling
        conv_out_size = (input_size[0] // 8, input_size[1] // 8)
        self.fc1 = nn.Linear(128 * conv_out_size[0] * conv_out_size[1], 1024)
        self.fc2 = nn.Linear(1024, input_size[0])  # Output size matches the number of lines in the image
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

def load_and_preprocess_image(image_path, target_size=(640, 320)):
    # Load the image (you might need to adjust this based on how your images are stored)
    image = np.load(image_path)
    
    # Normalize the image
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    
    # Convert to PyTorch tensor and add batch dimension
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor

def predict_motion_lines(model, image_tensor):
    model.eval()
    with torch.no_grad():
        predictions = model(image_tensor)
    return predictions.squeeze().numpy()

def visualize_and_save_results(image, predictions, output_dir, threshold=0.5):
    plt.figure(figsize=(12, 6))
    
    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot predictions
    plt.subplot(1, 2, 2)
    plt.imshow(image.squeeze(), cmap='gray')
    motion_lines = np.where(predictions > threshold)[0]
    for line in motion_lines:
        plt.axhline(y=line, color='r', linestyle='-', linewidth=0.5, alpha=0.5)
    plt.title('Detected Motion Lines')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'motion_detection_result.png'))
    plt.close()

    # Save predictions
    np.save(os.path.join(output_dir, 'predictions.npy'), predictions)

def main():
    # Load the trained model
    model = LineDetectionNet()
    model.load_state_dict(torch.load('../Simulation/line_detection_model_final.pth'))
    
    # Path to a test image
    test_image_path = '../Simulation/reorganized_output/val/slice_0/motion_affected_image.npy'
    
    # Load and preprocess the image
    image_tensor = load_and_preprocess_image(test_image_path)
    
    # Make predictions
    predictions = predict_motion_lines(model, image_tensor)
    
    # Visualize and save the results
    output_dir = 'motion_detection_output'
    visualize_and_save_results(image_tensor, predictions, output_dir)
    print(f"Results saved in {output_dir}")

if __name__ == "__main__":
    main()
