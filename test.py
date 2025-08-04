import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. Enhanced Data Preparation with Feature Engineering
def prepare_data(filepath, n_lags=5, test_size=0.15, val_size=0.15):
    data = pd.read_csv(filepath)
    
    # Feature engineering
    def add_features(df):
        df['B_mag'] = np.sqrt(df['Bx1']**2 + df['By1']**2 + df['Bz1']**2)
        df['B2_mag'] = np.sqrt(df['Bx2']**2 + df['By2']**2 + df['Bz2']**2)
        df['Bx_diff'] = df['Bx1'] - df['Bx2']
        df['By_diff'] = df['By1'] - df['By2']
        df['Bz_diff'] = df['Bz1'] - df['Bz2']
        return df
    
    data = add_features(data)
    
    # Select features and targets
    feature_cols = ['Bx1', 'By1', 'Bz1', 'Bx2', 'By2', 'Bz2', 
                   'B_mag', 'B2_mag', 'Bx_diff', 'By_diff', 'Bz_diff']
    target_cols = ['Fx', 'Fy', 'Fz']
    
    # Robust scaling
    def robust_scale(x):
        median = np.median(x, axis=0)
        iqr = np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0)
        return (x - median) / (iqr + 1e-8)
    
    features = robust_scale(data[feature_cols].values)
    targets = robust_scale(data[target_cols].values)
    
    # Create sequences
    def create_sequences(feats, targs, n_lags):
        X, Y = [], []
        for i in range(n_lags, len(feats)):
            X.append(feats[i-n_lags:i+1].flatten())
            Y.append(targs[i])
        return np.array(X), np.array(Y)
    
    X, Y = create_sequences(features, targets, n_lags)
    
    # Split data preserving temporal order
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, shuffle=False)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size=val_size, shuffle=False)
    
    return (torch.FloatTensor(X_train), torch.FloatTensor(Y_train),
            torch.FloatTensor(X_val), torch.FloatTensor(Y_val),
            torch.FloatTensor(X_test), torch.FloatTensor(Y_test))

# 2. Enhanced Model Architecture with Separate Heads
class ForceCalibrationModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # Shared backbone
        self.shared = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3)
        )
        
        # Component-specific heads
        self.fx_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1))
        
        self.fy_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1))
        
        self.fz_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        shared = self.shared(x)
        return torch.cat([
            self.fx_head(shared),
            self.fy_head(shared),
            self.fz_head(shared)
        ], dim=1)

# 3. Custom Weighted Loss Function
class ComponentWeightedLoss(nn.Module):
    def __init__(self, weights=[1.0, 2.0, 0.8]):
        super().__init__()
        self.weights = torch.tensor(weights)
        
    def forward(self, pred, target):
        abs_errors = (pred - target).abs()
        weighted_errors = abs_errors * self.weights.to(pred.device)
        return weighted_errors.mean()

# 4. Enhanced Training Process
def train_model(model, train_loader, val_loader, epochs=500):
    criterion = ComponentWeightedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=15, factor=0.5, verbose=True)
    
    best_val_loss = float('inf')
    history = {'train': [], 'val': [], 'components': []}
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_comps = torch.zeros(3)
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_comps += (pred - y).abs().mean(dim=0).detach()
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_comps = torch.zeros(3)
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x)
                val_loss += criterion(pred, y).item()
                val_comps += (pred - y).abs().mean(dim=0)
        
        # Calculate averages
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        train_comps /= len(train_loader)
        val_comps /= len(val_loader)
        
        # Update scheduler
        scheduler.step(avg_val)
        
        # Save best model
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), 'best_model.pth')
        
        # Store history
        history['train'].append(avg_train)
        history['val'].append(avg_val)
        history['components'].append(val_comps.numpy())
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1}/{epochs} | LR: {lr:.2e}')
            print(f'Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}')
            print(f'Component MAEs: Fx={val_comps[0]:.4f}, Fy={val_comps[1]:.4f}, Fz={val_comps[2]:.4f}\n')
    
    return history

# 5. Evaluation and Visualization
def evaluate_model(model, test_loader):
    model.eval()
    predictions, truths = [], []
    
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x)
            predictions.append(pred.numpy())
            truths.append(y.numpy())
    
    preds = np.concatenate(predictions)
    trues = np.concatenate(truths)
    
    # Calculate metrics
    errors = trues - preds
    mae = np.mean(np.abs(errors), axis=0)
    rmse = np.sqrt(np.mean(errors**2, axis=0))
    r2 = 1 - np.sum(errors**2, axis=0)/np.sum((trues - np.mean(trues, axis=0))**2, axis=0)
    
    return preds, trues, mae, rmse, r2

def plot_results(history, preds, trues):
    plt.figure(figsize=(18, 6))
    
    # Loss curves
    plt.subplot(1, 3, 1)
    plt.plot(history['train'], label='Train')
    plt.plot(history['val'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History')
    
    # Component errors over time
    plt.subplot(1, 3, 2)
    components = np.array(history['components'])
    plt.plot(components[:, 0], label='Fx')
    plt.plot(components[:, 1], label='Fy')
    plt.plot(components[:, 2], label='Fz')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.title('Component Validation Errors')
    
    # Predictions vs Actual
    plt.subplot(1, 3, 3)
    for i, color in enumerate(['r', 'g', 'b']):
        plt.scatter(trues[:200,i], preds[:200,i], c=color, alpha=0.5, label=f'F{["x","y","z"][i]}')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.legend()
    plt.title('Prediction vs True')
    
    plt.tight_layout()
    plt.show()

# Main Execution
def main():
    # Prepare data
    X_train, Y_train, X_val, Y_val, X_test, Y_test = prepare_data(
        'percy81.csv', n_lags=5)
    
    # Create loaders
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=128)
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=128)
    
    # Initialize model
    model = ForceCalibrationModel(X_train.shape[1])
    
    # Train
    print("Starting training...")
    history = train_model(model, train_loader, val_loader, epochs=300)
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Evaluate
    preds, trues, mae, rmse, r2 = evaluate_model(model, test_loader)
    
    # Results
    print("\n=== Final Test Results ===")
    print(f"MAE:  {mae}")
    print(f"RMSE: {rmse}")
    print(f"R²:   {r2}")
    
    # Plot
    plot_results(history, preds, trues)

if __name__ == "__main__":
    main()
