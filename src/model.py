import torch
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F


class GNNRegression(torch.nn.Module):
    def __init__(self, hidden_channels, in_channels, out_channels, num_layers=2):
        super().__init__()

        # Build conv layers dynamically to see how they are affected
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))   # first layer: in_channels → hidden
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))  # remaining: hidden → hidden

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # Iterate over all conv layers, applying ReLU after each except the last
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu()
                #x = F.dropout(x, p=0.5, training=self.training) # toggle using model.train() or model.eval()

        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x

