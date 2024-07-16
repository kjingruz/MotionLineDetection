import torch
import torch.nn as nn

class RealValCNNLineDetNew(nn.Module):
    def __init__(self, activation="relu", input_dim=2, input_size=[16, 16, 320, 320],
                 crop_readout=False, output_size=[16, 320], first_filters=16, last_filters=False,
                 kernel_size=5, num_layer=4, dropout=False, use_bias=True,
                 dtype=torch.float32, normalization=None, **kwargs):

        super(RealValCNNLineDetNew, self).__init__()
        conv_layer = nn.Conv3d

        if activation == "relu":
            act_layer = nn.ReLU

        if normalization == "BatchNorm":
            norm_layer = nn.BatchNorm3d

        filters = [input_dim, 32, 64, 128, 256]  # Adjust filter sizes as needed

        padding = kernel_size // 2
        pool_kernel_size = (1, 2, 2)
        pool_stride = (1, 2, 2)

        # create layers
        self.ops = nn.ModuleList()
        for i in range(len(filters) - 1):
            self.ops.append(conv_layer(in_channels=filters[i],
                                       out_channels=filters[i+1],
                                       kernel_size=kernel_size, padding=padding,
                                       bias=use_bias, dtype=dtype, **kwargs))

            if normalization == "BatchNorm":
                self.ops.append(norm_layer(filters[i+1], dtype=dtype))
            self.ops.append(act_layer())
            if dropout is not False:
                self.ops.append(nn.Dropout3d(p=dropout))
            self.ops.append(nn.MaxPool3d(kernel_size=pool_kernel_size,
                                         stride=pool_stride))

        # Calculate the size of the flattened features
        self.feature_size = self._get_conv_output(input_size, input_dim)

        self.fc = nn.Linear(self.feature_size, output_size[0] * output_size[1], dtype=dtype)
        if dropout is not False:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = False
        self.Sigmoid = nn.Sigmoid()

        self.output_size = output_size
        self.apply(self.weight_init)

    def _get_conv_output(self, shape, input_dim):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, input_dim, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def _forward_features(self, x):
        for layer in self.ops:
            x = layer(x)
        return x

    def weight_init(self, module):
        if isinstance(module, (nn.Conv3d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight, gain=1)
            if module.bias is not None:
                module.bias.data.fill_(0)

    def forward(self, x):
        # x shape: [batch_size, 2, 16, 16, 320, 320]
        # Reshape to [batch_size * 16, 2, 16, 320, 320]
        x = x.view(-1, x.size(1), x.size(3), x.size(4), x.size(5))
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.Sigmoid(x)
        return x.view(-1, *self.output_size)