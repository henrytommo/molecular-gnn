import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


@torch.no_grad()
def get_predictions(loader, model):
    model.eval()
    actuals = []
    predictions = []
    
    for data in loader:
        out = model(data.x.float(), data.edge_index, data.batch)
        
        # Collect values
        actuals.append(data.y.float().cpu())
        predictions.append(out.cpu())

    # Concatenate list of tensors into a single array
    actuals = torch.cat(actuals, dim=0).numpy()
    predictions = torch.cat(predictions, dim=0).numpy()
    mse = mean_squared_error(actuals, predictions)
    return actuals, predictions, mse
