import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path

# ======================
# 1. Data Preparation
# ======================

def load_and_align_data(filepath, shift_i=2):
    """Load and align sensor data with proper time shifting"""
    data = pd.read_csv(filepath)
    
    # Zero all measurements
    for col in ['Fx', 'Fy', 'Fz', 'Bx1', 'By1', 'Bz1', 'Bx2', 'By2', 'Bz2']:
        data[col] = data[col] - data[col].iloc[0]
    
    # Apply time shift
    features = data[['Bx1', 'By1', 'Bz1', 'Bx2', 'By2', 'Bz2']].values[shift_i:]
    targets = data[['Fx', 'Fy', 'Fz']].values[:-shift_i]
    
    # Ensure perfect alignment
    min_length = min(len(features), len(targets))
    return features[:min_length], targets[:min_length]

def create_sequences(features, targets, n_lags):
    """Create time-lagged sequences"""
    X, Y = [], []
    for i in range(n_lags, len(features)):
        X.append(features[i-n_lags:i].flatten())  # Flatten time steps
        Y.append(targets[i])
    return np.array(X), np.array(Y)

def prepare_datasets(filepath, n_lags=3, val_ratio=0.15, test_ratio=0.15):
    """Full data preparation pipeline"""
    features, targets = load_and_align_data(filepath)
    X, Y = create_sequences(features, targets, n_lags)
    
    # Split indices
    test_split = int(len(X) * (1 - test_ratio))
    val_split = int(test_split * (1 - val_ratio))
    
    # Split datasets
    X_train, X_val, X_test = X[:val_split], X[val_split:test_split], X[test_split:]
    Y_train, Y_val, Y_test = Y[:val_split], Y[val_split:test_split], Y[test_split:]
    
    # Normalize using training stats
    X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
    Y_mean, Y_std = Y_train.mean(axis=0), Y_train.std(axis=0)
    
    def normalize(x, mean, std): 
        return (x - mean) / (std + 1e-8)  # Small epsilon to avoid division by zero
    
    return (
        normalize(X_train, X_mean, X_std), normalize(Y_train, Y_mean, Y_std),
        normalize(X_val, X_mean, X_std), normalize(Y_val, Y_mean, Y_std),
        normalize(X_test, X_mean, X_std), normalize(Y_test, Y_mean, Y_std),
        X_mean, X_std, Y_mean, Y_std
    )

# ======================
# 2. Model Definition
# ======================

class ForceCalibrationModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            
            nn.Linear(64, 3)
        )
    
    def forward(self, x):
        return self.net(x)

# ======================
# 3. Training Utilities
# ======================

def create_loaders(X_train, Y_train, X_val, Y_val, X_test, Y_test, batch_size=64):
    """Create PyTorch data loaders"""
    train_data = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
    val_data = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(Y_val))
    test_data = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(Y_test))
    
    return (
        DataLoader(train_data, batch_size=batch_size, shuffle=True),
        DataLoader(val_data, batch_size=batch_size),
        DataLoader(test_data, batch_size=batch_size)
    )

def train_model(model, train_loader, val_loader, epochs=300, patience=20):
    """Training loop with early stopping"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    
    train_losses, val_losses = [], []
    best_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                outputs = model(x)
                val_loss += criterion(outputs, y).item()
        
        # Record metrics
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        
        # Early stopping check
        if avg_val < best_loss:
            best_loss = avg_val
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_model.pth')
        
        # Learning rate scheduling
        scheduler.step(avg_val)
        
        # Progress reporting
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}: Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}')
        
        # Early stopping
        if epoch - best_epoch > patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return train_losses, val_losses

# ======================
# 4. Evaluation & Visualization
# ======================

def evaluate_model(model, test_loader, Y_mean, Y_std):
    """Evaluate model on test set"""
    model.eval()
    predictions, truths = [], []
    
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x)
            predictions.append(pred.numpy())
            truths.append(y.numpy())
    
    # Denormalize predictions
    preds = np.concatenate(predictions) * Y_std + Y_mean
    trues = np.concatenate(truths) * Y_std + Y_mean
    
    # Calculate metrics
    errors = trues - preds
    mae = np.mean(np.abs(errors), axis=0)
    rmse = np.sqrt(np.mean(errors**2, axis=0))
    r2 = 1 - np.sum(errors**2, axis=0) / np.sum((trues - np.mean(trues, axis=0))**2, axis=0)
    
    return preds, trues, mae, rmse, r2

def plot_results(train_loss, val_loss, preds, trues):
    """Generate diagnostic plots"""
    plt.figure(figsize=(15, 10))
    
    # Loss curves
    plt.subplot(2, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History')
    
    # Predictions vs Actual (Fx)
    plt.subplot(2, 2, 2)
    plt.plot(trues[:200, 0], label='True Fx')
    plt.plot(preds[:200, 0], label='Pred Fx')
    plt.legend()
    plt.title('Fx Prediction')
    
    # Predictions vs Actual (Fy)
    plt.subplot(2, 2, 3)
    plt.plot(trues[:200, 1], label='True Fy')
    plt.plot(preds[:200, 1], label='Pred Fy')
    plt.legend()
    plt.title('Fy Prediction')
    
    # Predictions vs Actual (Fz)
    plt.subplot(2, 2, 4)
    plt.plot(trues[:200, 2], label='True Fz')
    plt.plot(preds[:200, 2], label='Pred Fz')
    plt.legend()
    plt.title('Fz Prediction')
    
    plt.tight_layout()
    plt.show()

# ======================
# 5. Main Execution
# ======================

def main():
    # Configuration
    config = {
        'data_path': 'percy81.csv',
        'n_lags': 3,
        'batch_size': 64,
        'epochs': 300,
        'patience': 20
    }
    
    # Prepare data
    (X_train, Y_train, X_val, Y_val, 
     X_test, Y_test, X_mean, X_std, Y_mean, Y_std) = prepare_datasets(
        config['data_path'], 
        n_lags=config['n_lags']
    )
    
    # Create model
    model = ForceCalibrationModel(X_train.shape[1])
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_loaders(
        X_train, Y_train, X_val, Y_val, X_test, Y_test,
        batch_size=config['batch_size']
    )
    
    # Train model
    print("Starting training...")
    train_loss, val_loss = train_model(
        model, train_loader, val_loader,
        epochs=config['epochs'],
        patience=config['patience']
    )
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Evaluate
    preds, trues, mae, rmse, r2 = evaluate_model(model, test_loader, Y_mean, Y_std)
    
    # Print results
    print("\n=== Final Evaluation ===")
    print(f"MAE:  {mae}")
    print(f"RMSE: {rmse}")
    print(f"R²:   {r2}")
    
    # Plot results
    plot_results(train_loss, val_loss, preds, trues)
    
    # Save model and normalization parameters
    torch.save(model.state_dict(), 'force_calibration_model_final.pth')
    np.savez('normalization_params.npz',
             X_mean=X_mean, X_std=X_std,
             Y_mean=Y_mean, Y_std=Y_std,
             n_lags=config['n_lags'])

if __name__ == "__main__":
    main()
