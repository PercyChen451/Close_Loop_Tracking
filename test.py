import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 1. Data Loading with Strict Alignment
def load_and_align_data(filepath):
    data = pd.read_csv(filepath)
    
    # Extract and zero all columns
    cols = ['Fx', 'Fy', 'Fz', 'Bx1', 'By1', 'Bz1', 'Bx2', 'By2', 'Bz2']
    data = data[cols].apply(lambda x: x - x.iloc[0])
    
    # Apply time shift
    shift_i = 2
    features = data[['Bx1', 'By1', 'Bz1', 'Bx2', 'By2', 'Bz2']].values[shift_i:]
    targets = data[['Fx', 'Fy', 'Fz']].values[:-shift_i]
    
    # Ensure perfect alignment
    min_length = min(len(features), len(targets))
    features = features[:min_length]
    targets = targets[:min_length]
    
    return features, targets

# 2. Sequence Creation with Validation
def create_sequences(features, targets, n_lags):
    X, Y = [], []
    for i in range(n_lags, len(features)):
        X.append(features[i-n_lags:i].flatten())
        Y.append(targets[i])
    return np.array(X), np.array(Y)

# 3. Data Preparation Pipeline
def prepare_data(filepath, n_lags=3, test_size=0.2):
    # Load and align
    features, targets = load_and_align_data(filepath)
    
    # Create sequences
    X, Y = create_sequences(features, targets, n_lags)
    
    # Split indices
    split_idx = int(len(X) * (1 - test_size))
    
    # Train/Test split
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]
    
    # Normalize using train stats
    X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
    Y_mean, Y_std = Y_train.mean(axis=0), Y_train.std(axis=0)
    
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    Y_train = (Y_train - Y_mean) / Y_std
    Y_test = (Y_test - Y_mean) / Y_std
    
    return (X_train, Y_train, X_test, Y_test, 
            X_mean, X_std, Y_mean, Y_std)

# 4. Main Execution
if __name__ == "__main__":
    # Parameters
    n_lags = 3
    batch_size = 64
    epochs = 300
    
    # Prepare data
    X_train, Y_train, X_test, Y_test, X_mean, X_std, Y_mean, Y_std = prepare_data('percy81.csv', n_lags)
    
    # Convert to tensors
    train_data = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
    test_data = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(Y_test))
    
    # Create loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    
    # Model definition
    class ForceModel(nn.Module):
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
                nn.Linear(128, 3)
            )
        
        def forward(self, x):
            return self.net(x)
    
    model = ForceModel(X_train.shape[1])
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                pred = model(x)
                test_loss += criterion(pred, y).item()
        
        # Update scheduler
        avg_test_loss = test_loss/len(test_loader)
        scheduler.step(avg_test_loss)
        
        # Early stopping check
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}: Train Loss {train_loss/len(train_loader):.4f}, Test Loss {avg_test_loss:.4f}')
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.FloatTensor(X_test)).numpy()
        y_pred = y_pred * Y_std + Y_mean
        y_true = Y_test * Y_std + Y_mean
    
    # Calculate metrics
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    rmse = np.sqrt(np.mean((y_true - y_pred)**2, axis=0)
    r2 = 1 - np.sum((y_true - y_pred)**2, axis=0) / np.sum((y_true - np.mean(y_true, axis=0))**2, axis=0)
    
    print(f"\nFinal Performance:")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"R²: {r2}")
# [Previous fixed code up to the evaluation section]

    # Evaluation
    model.eval()
    with torch.no_grad():
        # Get predictions for test set
        Y_pred_norm = model(torch.FloatTensor(X_test)).numpy()
        Y_pred_test = (Y_pred_norm * Y_std) + Y_mean
        Y_true_test = (Y_test * Y_std) + Y_mean
        
        # Get predictions for full dataset (for plotting)
        Y_pred_full_norm = model(torch.FloatTensor(np.vstack([X_train, X_test]))).numpy()
        Y_pred_full = (Y_pred_full_norm * Y_std) + Y_mean
        Y_true_full = (np.vstack([Y_train, Y_test]) * Y_std) + Y_mean

    # 1. Training and Validation Loss Plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)
    plt.show()

    # 2. Component-wise Time Series Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    components = ['Fx', 'Fy', 'Fz']
    time_points = min(200, len(Y_true_test))  # Show first 200 points or all if less
    
    for i, ax in enumerate(axes):
        ax.plot(Y_true_test[:time_points, i], label='True')
        ax.plot(Y_pred_test[:time_points, i], label='Predicted')
        ax.set_title(components[i])
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.show()

    # 3. Error Distribution Plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    errors = Y_true_test - Y_pred_test
    
    for i, ax in enumerate(axes):
        ax.hist(errors[:, i], bins=50)
        ax.set_title(f'{components[i]} Error Distribution')
        ax.grid(True)
    plt.tight_layout()
    plt.show()

    # 4. 3D Scatter Plot (True vs Predicted)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot subset of points for clarity
    plot_every = 10  # Plot every 10th point
    ax.scatter(Y_true_full[::plot_every, 0], 
               Y_true_full[::plot_every, 1], 
               Y_true_full[::plot_every, 2], 
               c='b', marker='o', label='True Force', alpha=0.5)
    
    ax.scatter(Y_pred_full[::plot_every, 0], 
               Y_pred_full[::plot_every, 1], 
               Y_pred_full[::plot_every, 2], 
               c='r', marker='x', label='Predicted Force', alpha=0.5)
    
    ax.set_xlabel('Fx')
    ax.set_ylabel('Fy')
    ax.set_zlabel('Fz')
    ax.legend()
    plt.title('True vs Predicted Forces (Full Dataset)')
    plt.show()

    # 5. Prediction Error vs Magnitude Plot
    magnitudes = np.linalg.norm(Y_true_test, axis=1)
    error_magnitudes = np.linalg.norm(errors, axis=1)
    
    plt.figure(figsize=(10, 5))
    plt.scatter(magnitudes, error_magnitudes, alpha=0.5)
    plt.xlabel('Force Magnitude (True)')
    plt.ylabel('Prediction Error Magnitude')
    plt.title('Prediction Error vs Force Magnitude')
    plt.grid(True)
    plt.show()

    # Calculate and print metrics
    SS_res = np.sum((Y_true_test - Y_pred_test)**2, axis=0)
    SS_tot = np.sum((Y_true_test - np.mean(Y_true_test, axis=0))**2, axis=0)
    R2 = 1 - (SS_res / SS_tot)
    
    errors = Y_true_test - Y_pred_test
    rmse = np.sqrt(np.mean(errors**2, axis=0))
    mae = np.mean(np.abs(errors), axis=0)
    
    print('\n=== Performance Metrics ===')
    print('R² for [Fx, Fy, Fz]:')
    print(R2)
    print('\nRMSE for [Fx, Fy, Fz]:')
    print(rmse)
    print('\nMAE for [Fx, Fy, Fz]:')
    print(mae)

    # Save model and normalization parameters
    torch.save(model.state_dict(), 'best_force_model.pth')
    np.savez('normalization_params.npz',
             X_mean=X_mean, X_std=X_std,
             Y_mean=Y_mean, Y_std=Y_std,
             n_lags=n_lags)
