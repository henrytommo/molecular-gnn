import torch

from src.de import train_dataset, train_loader
from src.model import GNNRegression

def train(num_epochs):
    model = GNNRegression(hidden_channels=64, in_channels=train_dataset.num_node_features, out_channels=1)
    mse_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        
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