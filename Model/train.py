import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class FastMRIDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.raw_samples = []
        self.transform = transform

        files = list(pathlib.Path(self.path).rglob("*.h5"))
        for filename in sorted(files):
            slices_ind = self._get_slice_indices(filename)
            for slice_ind in slices_ind:
                self.raw_samples.append((filename, slice_ind))

    def _get_slice_indices(self, filename):
        with h5py.File(filename, "r") as hf:
            slices_ind = np.arange(hf["kspace"].shape[0])
        return slices_ind

    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, idx):
        filename, dataslice = self.raw_samples[idx]
        with h5py.File(filename, "r") as hf:
            kspace = hf["kspace"][dataslice]
            kspace = np.stack((kspace.real, kspace.imag), axis=0)
        if self.transform:
            kspace = self.transform(kspace)
        return torch.as_tensor(kspace, dtype=torch.float32)

class RealValCNNLineDet2D(nn.Module):
    def __init__(self, input_dim=2, input_size=[92, 112], output_size=[92], first_filters=16, num_layer=4):
        super(RealValCNNLineDet2D, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_dim, first_filters, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(first_filters, first_filters*2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(first_filters*2, first_filters*4, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(first_filters*4, first_filters*8, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(first_filters*8*6*7, output_size[0])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

# Load dataset
dataset = FastMRIDataset(path='../Data/')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Initialize model, loss function, and optimizer
model = RealValCNNLineDet2D()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        inputs = data
        labels = torch.ones(inputs.size(0), 92)  # Placeholder labels
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 9:
            print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 10:.3f}')
            running_loss = 0.0

print('Finished Training')
