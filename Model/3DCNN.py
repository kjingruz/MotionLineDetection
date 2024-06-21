import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class RealValCNNLineDetNew(nn.Module):
    def __init__(self, activation="relu", input_dim=2, input_size=[12, 92, 112],
                 crop_readout=False, output_size=[12, 92], first_filters=16, last_filters=False,
                 kernel_size=5, num_layer=4, dropout=False, use_bias=True,
                 dtype=torch.float64, normalization=None, **kwargs):
        super(RealValCNNLineDetNew, self).__init__()
        conv_layer = nn.Conv3d
        act_layer = nn.ReLU if activation == "relu" else None
        norm_layer = nn.BatchNorm3d if normalization == "BatchNorm" else None

        filters = [first_filters * i if i < num_layer // 2 else first_filters * (i + 1) for i in range(1, num_layer + 1)]
        padding = kernel_size // 2
        pool_kernel_size = [(2, 1, 2) if i < 3 else (1, 1, 2) for i in range(num_layer + 1 if last_filters else num_layer)]
        pool_stride = pool_kernel_size

        self.ops = nn.Sequential()
        self.ops.add_module('conv1', conv_layer(input_dim, filters[0], kernel_size, padding=padding, bias=use_bias, dtype=dtype, **kwargs))
        if norm_layer: self.ops.add_module('bn1', norm_layer(filters[0], dtype=dtype))
        self.ops.add_module('act1', act_layer())
        if dropout: self.ops.add_module('drop1', nn.Dropout3d(p=dropout))
        self.ops.add_module('pool1', nn.MaxPool3d(pool_kernel_size[0], pool_stride[0]))

        for i in range(1, num_layer):
            self.ops.add_module(f'conv{i+1}', conv_layer(filters[i-1], filters[i], kernel_size, padding=padding, bias=use_bias, dtype=dtype, **kwargs))
            if norm_layer: self.ops.add_module(f'bn{i+1}', norm_layer(filters[i], dtype=dtype))
            self.ops.add_module(f'act{i+1}', act_layer())
            if dropout: self.ops.add_module(f'drop{i+1}', nn.Dropout3d(p=dropout))
            self.ops.add_module(f'pool{i+1}', nn.MaxPool3d(pool_kernel_size[i], pool_stride[i]))

        if last_filters:
            for i in range(len(last_filters)):
                self.ops.add_module(f'last_conv{i+1}', conv_layer(filters[-1] if i == 0 else last_filters[i-1], last_filters[i], kernel_size, padding=padding, bias=use_bias, dtype=dtype, **kwargs))
                if norm_layer: self.ops.add_module(f'last_bn{i+1}', norm_layer(last_filters[i], dtype=dtype))
                self.ops.add_module(f'last_act{i+1}', act_layer())
                if dropout: self.ops.add_module(f'last_drop{i+1}', nn.Dropout3d(p=dropout))

        size1, size2 = input_size[0], crop_readout if crop_readout else input_size[2]
        for j in range(min(3, num_layer + 1)): size1 //= 2
        for j in range(num_layer + 1): size2 //= 2
        num_filters = filters[-1] if not last_filters else last_filters[-1]

        self.fc = nn.Linear(num_filters * input_size[1] * size1 * size2, output_size[0] * output_size[1], dtype=dtype)
        self.dropout = nn.Dropout(p=dropout) if dropout else None
        self.sigmoid = nn.Sigmoid()
        self.apply(self.weight_init)

    def weight_init(self, module):
        if isinstance(module, (nn.Conv3d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight, gain=1)
            if module.bias is not None:
                module.bias.data.fill_(0)

    def forward(self, x):
        x = self.ops(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.sigmoid(x)
        return x

# Define the 3D model 
# Instantiate the model for 3D data
model_3d = RealValCNNLineDetNew()

# Prepare the 3D k-space data (simulated)
depth = 12  # Simulating 12 echo times
simulated_kspace_3d = np.repeat(motion_affected_kspace_2d[np.newaxis, :, :], depth, axis=0)
simulated_kspace_3d = np.stack((simulated_kspace_3d.real, simulated_kspace_3d.imag), axis=0)
motion_affected_kspace_tensor_3d = torch.tensor(simulated_kspace_3d, dtype=torch.float64).unsqueeze(0)

# Forward pass through the 3D model
model_3d.eval()
with torch.no_grad():
    output_3d = model_3d(motion_affected_kspace_tensor_3d)

# Output shape should match the model's output_size
print("Output shape for 3D data:", output_3d.shape)

# Visualize the 3D output (summing along the depth dimension for visualization)
output_mask_3d = output_3d.view(12, 92).cpu().numpy().sum(axis=0)
plt.figure(figsize=(6, 6))
plt.title("Predicted Artifact Mask (3D)")
plt.imshow(output_mask_3d, cmap='gray')
plt.show()
