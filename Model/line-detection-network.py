import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
import logging
from torchvision import transforms
import torch.nn.functional as F

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Script started")

class MotionArtifactDataset(Dataset):
    def __init__(self, root_dir, target_size=(640, 320)):
        self.root_dir = root_dir
        self.slice_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.slice_dirs)

    def __getitem__(self, idx):
        slice_dir = self.slice_dirs[idx]
        motion_affected_image = np.load(os.path.join(slice_dir, "motion_affected_image.npy"))
        motion_labels = np.load(os.path.join(slice_dir, "motion_labels.npy"))
        
        # Normalize the image
        motion_affected_image = (motion_affected_image - np.min(motion_affected_image)) / (np.max(motion_affected_image) - np.min(motion_affected_image))
        
        # Convert to PyTorch tensor and resize image
        motion_affected_image = self.transform(motion_affected_image)
        
        # Convert labels to PyTorch tensor
        motion_labels = torch.from_numpy(motion_labels).float()
        
        # Resize labels if necessary
        if len(motion_labels) != self.target_size[0]:
            motion_labels = F.interpolate(motion_labels.unsqueeze(0).unsqueeze(0), 
                                          size=(self.target_size[0],), 
                                          mode='linear', 
                                          align_corners=False).squeeze()
        
        return motion_affected_image, motion_labels
    
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
        logging.debug(f"Input shape: {x.shape}")
        x = self.pool(self.relu(self.conv1(x)))
        logging.debug(f"After conv1: {x.shape}")
        x = self.pool(self.relu(self.conv2(x)))
        logging.debug(f"After conv2: {x.shape}")
        x = self.pool(self.relu(self.conv3(x)))
        logging.debug(f"After conv3: {x.shape}")
        x = x.view(x.size(0), -1)
        logging.debug(f"After flatten: {x.shape}")
        x = self.relu(self.fc1(x))
        logging.debug(f"After fc1: {x.shape}")
        x = self.sigmoid(self.fc2(x))
        logging.debug(f"Output shape: {x.shape}")
        return x
    
def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
    for batch_idx, (inputs, labels) in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        if batch_idx % 10 == 0:
            logging.info(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    epoch_loss = running_loss / len(train_loader)
    return epoch_loss

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    return running_loss / len(val_loader)

def save_checkpoint(model, optimizer, epoch, loss, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)
    logging.info(f"Checkpoint saved: {filename}")

def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    logging.info(f"Checkpoint loaded: {filename}")
    return model, optimizer, epoch, loss

def main():
    # Set up paths
    train_dir = "../Simulation/reorganized_output/train"
    val_dir = "../Simulation/reorganized_output/val"
    
    # Check if directories exist
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        logging.error(f"Train or validation directory not found. Please check the paths.")
        return

    # Set up dataset and dataloader
    target_size = (640, 320)
    train_dataset = MotionArtifactDataset(train_dir, target_size=target_size)
    val_dataset = MotionArtifactDataset(val_dir, target_size=target_size)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Set up model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    model = LineDetectionNet(input_size=target_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training parameters
    num_epochs = 100  # Increased number of epochs
    best_val_loss = float('inf')
    start_epoch = 0

    # Check for existing checkpoint
    checkpoint_path = 'line_detection_checkpoint.pth'
    if os.path.exists(checkpoint_path):
        model, optimizer, start_epoch, best_val_loss = load_checkpoint(model, optimizer, checkpoint_path)
        logging.info(f"Resuming training from epoch {start_epoch}")
    else:
        logging.info("Starting training from scratch")

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        logging.info(f"Starting epoch {epoch+1}/{num_epochs}")
        train_loss = train(model, train_loader, criterion, optimizer, device, epoch+1)
        val_loss = validate(model, val_loader, criterion, device)
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch+1, val_loss, 'line_detection_best_model.pth')
        
        # Save regular checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch+1, val_loss, checkpoint_path)

    # Save the final model
    torch.save(model.state_dict(), "line_detection_model_final.pth")
    logging.info("Training completed. Final model saved.")

if __name__ == "__main__":
    main()
