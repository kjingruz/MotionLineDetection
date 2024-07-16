import torch
from torch.utils.data import DataLoader
from dataset import T2StarDataset
from cnn_line_detection import RealValCNNLineDetNew
from Trainer import PTrainer
import os

class DataWrapper:
    def __init__(self, train_loader, val_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

def main():
    # Data loading
    train_data_dir = "../../Data/multicoil_train"
    val_data_dir = "../../Data/multicoil_val"
    num_motion_artifacts = 3  # Increase this number for more motion artifacts
    target_size = (320, 320)  # Reduced target size
    num_slices = 16  # Fixed number of slices
    train_dataset = T2StarDataset(train_data_dir, num_motion_artifacts=num_motion_artifacts, target_size=target_size, num_slices=num_slices)
    val_dataset = T2StarDataset(val_data_dir, num_motion_artifacts=1, target_size=target_size, num_slices=num_slices)
    
    # Reduced batch size
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

    # Wrap the data loaders
    data = DataWrapper(train_loader, val_loader)

    # Model
    model = RealValCNNLineDetNew(input_dim=2, input_size=[16, 320, 320], output_size=[num_slices, target_size[0]], dtype=torch.float32)

    # Training parameters
    training_params = {
        "nr_epochs": 200,  # Increased number of epochs
        "learning_rate": 1e-4,  # Reduced learning rate
        "optimizer_params": {"lr": 1e-4},
        "checkpoint_path": "checkpoints",
        "loss": {
            "module_name": "torch.nn",
            "class_name": "BCELoss",
            "params": None
        }
    }

    # Trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = PTrainer(training_params, model, data, device)

    # Train
    trainer.train()

    # Save the final model
    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(trainer.model.state_dict(), "models/final_model.pt")

if __name__ == "__main__":
    main()