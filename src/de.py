# note - using pre-loaded data set so no de
import os
import torch
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import DataLoader
from torch_geometric.datasets import MoleculeNet

dataset = MoleculeNet(root='./data', name='ESOL')
dataset.data.x = dataset.data.x.to(torch.float)

# 70:20:10 split
batch_size = 32
train_dataset = dataset[:770]
val_dataset = dataset[770:990]
test_dataset = dataset[990:]
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)