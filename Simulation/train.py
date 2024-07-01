import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
from dataset import T2StarDataset
from model import LineDetectionNet

def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50, accumulation_steps=4, checkpoint_dir='checkpoints'):
    scaler = GradScaler()
    best_val_loss = float('inf')
    start_epoch = 0

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Check if there's a checkpoint to resume from
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[2].split('.')[0]))
        checkpoint = torch.load(os.path.join(checkpoint_dir, latest_checkpoint))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        for i, (image, mask, slice_mask) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            image, mask, slice_mask = image.to(device), mask.to(device), slice_mask.to(device)
            with autocast():
                outputs = model(image, slice_mask)
                loss = criterion(outputs, mask)
            
            loss = loss / accumulation_steps
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item() * accumulation_steps
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for image, mask, slice_mask in val_loader:
                image, mask, slice_mask = image.to(device), mask.to(device), slice_mask.to(device)
                with autocast():
                    outputs = model(image, slice_mask)
                    loss = criterion(outputs, mask)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    target_size = (640, 320)
    max_slices = 20
    
    train_dir = "../../Data/multicoil_train"
    val_dir = "../../Data/multicoil_val"
    
    train_dataset = T2StarDataset(train_dir, target_size=target_size, max_slices=max_slices)
    val_dataset = T2StarDataset(val_dir, target_size=target_size, max_slices=max_slices)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    
    model = LineDetectionNet(input_channels=1, max_slices=max_slices, input_size=target_size, output_size=target_size[0]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50)

if __name__ == "__main__":
    main()