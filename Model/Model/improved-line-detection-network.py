import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler

class T2StarDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_size=(640, 320), max_slices=20):
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        self.transform = transform
        self.target_size = target_size
        self.max_slices = max_slices
        self.resize = transforms.Resize(target_size)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = np.load(os.path.join(self.data_dir, self.file_list[idx]))
        
        # Combine real and imaginary parts
        kspace = np.stack([data['kspace_real'], data['kspace_imag']], axis=0)
        
        # Get the motion mask
        motion_mask = data['motion_mask']
        
        # Convert to PyTorch tensors
        kspace_tensor = torch.from_numpy(kspace).float()
        mask_tensor = torch.from_numpy(motion_mask).float()
        
        # Resize kspace_tensor
        kspace_tensor = self.resize(kspace_tensor)
        
        # Pad or crop to max_slices
        current_slices = kspace_tensor.shape[1]
        if current_slices < self.max_slices:
            padding = torch.zeros((2, self.max_slices - current_slices, *kspace_tensor.shape[2:]))
            kspace_tensor = torch.cat([kspace_tensor, padding], dim=1)
            slice_mask = torch.cat([torch.ones(current_slices), torch.zeros(self.max_slices - current_slices)])
        else:
            kspace_tensor = kspace_tensor[:, :self.max_slices]
            slice_mask = torch.ones(self.max_slices)
        
        # Adjust mask_tensor to match the number of lines in resized kspace
        if len(mask_tensor) != self.target_size[0]:
            mask_tensor = torch.nn.functional.interpolate(mask_tensor.unsqueeze(0).unsqueeze(0), 
                                                          size=(self.target_size[0],), 
                                                          mode='nearest').squeeze()
        
        if self.transform:
            kspace_tensor = self.transform(kspace_tensor)
        
        return kspace_tensor, mask_tensor, slice_mask

class LineDetectionNet(nn.Module):
    def __init__(self, input_channels=2, max_slices=20, input_size=(640, 320), output_size=640):
        super(LineDetectionNet, self).__init__()
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, slice_mask):
        # Apply slice mask
        x = x * slice_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.relu(self.conv5(x))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.sigmoid(self.fc(x))
        return x

def evaluate(model, test_loader, device):
    model.eval()
    all_preds = []
    all_masks = []
    with torch.no_grad():
        for kspace, mask in test_loader:
            kspace, mask = kspace.to(device), mask.to(device)
            outputs = model(kspace)
            all_preds.extend(outputs.cpu().numpy())
            all_masks.extend(mask.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_masks = np.array(all_masks)
    
    accuracy = np.mean((all_preds > 0.5) == all_masks)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Visualize a sample prediction
    idx = np.random.randint(len(all_preds))
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(all_masks[idx].squeeze(), cmap='gray')
    plt.title('True Mask')
    plt.subplot(132)
    plt.imshow(all_preds[idx].squeeze(), cmap='gray')
    plt.title('Predicted Mask')
    plt.subplot(133)
    plt.imshow((all_preds[idx].squeeze() > 0.5).astype(float), cmap='gray')
    plt.title('Thresholded Prediction')
    plt.show()


class LineDetectionNet(nn.Module):
    def __init__(self, input_channels=2, max_slices=20, input_size=(640, 320), output_size=640):
        super(LineDetectionNet, self).__init__()
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, output_size)
        self.relu = nn.ReLU()
        # Remove sigmoid activation

    def forward(self, x, slice_mask):
        # Apply slice mask
        x = x * slice_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.relu(self.conv5(x))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50, accumulation_steps=4):
    scaler = GradScaler()
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        for i, (kspace, mask, slice_mask) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            kspace, mask, slice_mask = kspace.to(device), mask.to(device), slice_mask.to(device)
            with autocast():
                outputs = model(kspace, slice_mask)
                loss = criterion(outputs, mask)
            
            # Normalize loss to account for accumulation
            loss = loss / accumulation_steps
            
            # Scales loss and calls backward() to create scaled gradients
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                # Unscales gradients and calls or skips optimizer.step()
                scaler.step(optimizer)
                # Updates the scale for next iteration
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item() * accumulation_steps
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for kspace, mask, slice_mask in val_loader:
                kspace, mask, slice_mask = kspace.to(device), mask.to(device), slice_mask.to(device)
                with autocast():
                    outputs = model(kspace, slice_mask)
                    loss = criterion(outputs, mask)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    target_size = (640, 320)
    max_slices = 20 
    
    # Create datasets and dataloaders
    train_dataset = T2StarDataset('../Simulation/3D Slice/Train_data', target_size=target_size, max_slices=max_slices)
    val_dataset = T2StarDataset('../Simulation/3D Slice/Valid_data', target_size=target_size, max_slices=max_slices)
    test_dataset = T2StarDataset('../Simulation/3D Slice/Test_data', target_size=target_size, max_slices=max_slices)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    test_loader = DataLoader(test_dataset, batch_size=4)
    
    model = LineDetectionNet(input_channels=2, max_slices=max_slices, input_size=target_size, output_size=640).to(device)
    criterion = nn.BCEWithLogitsLoss()  
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Check if there's a checkpoint to resume from
    resume_from = None
    checkpoints = [f for f in os.listdir('.') if f.startswith('checkpoint_epoch_')]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[2].split('.')[0]))
        resume_from = latest_checkpoint
        print(f"Resuming from checkpoint: {resume_from}")
    
    train(model, train_loader, val_loader, criterion, optimizer, device, resume_from=resume_from)
    
    # Load best model for evaluation
    model.load_state_dict(torch.load('best_model.pth')['model_state_dict'])
    evaluate(model, test_loader, device)

if __name__ == "__main__":
    main()
