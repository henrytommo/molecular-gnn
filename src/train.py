import torch

from de import train_dataset, train_loader
from model import GNNRegression

def train(num_epochs, hidden_channels=32, lr=0.01, weight_decay=1e-4, num_layers=2):
    model = GNNRegression(
        hidden_channels=hidden_channels,
        in_channels=train_dataset.num_node_features,
        out_channels=1,
        num_layers=num_layers
    )
    mse_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    ) # weight decay for regularisation

    for _ in range(1, num_epochs + 1):
        model.train()
        for data in train_loader:
            optimizer.zero_grad()  # Clear gradients
            
            # Forward pass
            out = model(data.x, data.edge_index, data.batch)
            
            # Calculate MSE loss
            loss = mse_loss(out, data.y)
            
            # Backward pass
            loss.backward()
            optimizer.step()

    return model