import torch
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def train_hgat(model, dataloader, optimizer, criterion, device, epochs=100, patience=20):
    model.train()
    best_loss = float('inf')
    epochs_no_improve = 0
    history = {'loss': [], 'mse': [], 'bound': [], 'mono': []}
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_mse = 0
        
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            out = model(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
            
            # Calculate loss (no monotonicity loss for now as it requires specific flat features)
            loss, mse, bound, mono = criterion(out, batch.y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_mse += mse.item()
            
        avg_loss = epoch_loss / len(dataloader)
        history['loss'].append(avg_loss)
        history['mse'].append(epoch_mse / len(dataloader))
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
            
    return history

def train_ann(model, dataloader, optimizer, criterion, device, epochs=100, patience=20):
    model.train()
    best_loss = float('inf')
    epochs_no_improve = 0
    history = {'loss': [], 'mse': [], 'bound': [], 'mono': []}
    
    for X_batch, y_batch in dataloader:
        pass # To be implemented in notebook
