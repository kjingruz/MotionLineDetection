import torch.nn as nn  
        
class LineDetectionNet(nn.Module):
    def __init__(self, input_channels=1, max_slices=20, input_size=(640, 320), output_size=640):
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

    def forward(self, x, slice_mask):
        # Reshape slice_mask to match x's dimensions
        slice_mask = slice_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        x = x * slice_mask
        
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.relu(self.conv5(x))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x