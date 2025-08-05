
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. Enhanced Data Preparation with Normalization Parameters
def prepare_data(filepath, n_lags=3, test_size=0.15, val_size=0.15, shift_i=2):
    # Load and verify data
    data = pd.read_csv(filepath)
    print(f"Original data shape: {data.shape}")
    # Apply time shift with validation
    print(f"Applying time shift of {shift_i} samples")
    features = data[['Bx1', 'By1', 'Bz1', 'Bx2', 'By2', 'Bz2']].values[shift_i:]
    targets = data[['Fx', 'Fy', 'Fz']].values[:-shift_i]
    # Verify alignment
    assert len(features) == len(targets), \
        f"Alignment failed! Features: {len(features)}, Targets: {len(targets)}"
    # Feature engineering
    def add_features(feats):
        B_mag = np.sqrt(feats[:,0]**2 + feats[:,1]**2 + feats[:,2]**2)
        B2_mag = np.sqrt(feats[:,3]**2 + feats[:,4]**2 + feats[:,5]**2)
        return np.column_stack((feats, B_mag, B2_mag))
    features = add_features(features)
    print(f"Features shape after engineering: {features.shape}")
    # Create sequences with size validation
    def create_sequences(feats, targs, n_lags):
        X, Y = [], []
        for i in range(n_lags, len(feats)):
            X.append(feats[i-n_lags:i+1].flatten())
            Y.append(targs[i])
        return np.array(X), np.array(Y)
    X, Y = create_sequences(features, targets, n_lags)
    print(f"Final sequence shapes - X: {X.shape}, Y: {Y.shape}")
    # Split with size validation
    test_split = int(len(X) * (1 - test_size))
    val_split = int(test_split * (1 - val_size))
    X_train, X_val, X_test = X[:val_split], X[val_split:test_split], X[test_split:]
    Y_train, Y_val, Y_test = Y[:val_split], Y[val_split:test_split], Y[test_split:]
    print(f"Train shapes: X{X_train.shape}, Y{Y_train.shape}")
    print(f"Val shapes: X{X_val.shape}, Y{Y_val.shape}")
    print(f"Test shapes: X{X_test.shape}, Y{Y_test.shape}")
    # Calculate robust scaling parameters (median and IQR)
    def calculate_robust_params(data):
        median = np.median(data, axis=0)
        iqr = np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0)
        return median, iqr
    
    X_median, X_iqr = calculate_robust_params(X_train)
    Y_median, Y_iqr = calculate_robust_params(Y_train)
    
    # Apply normalization
    def normalize(data, median, iqr):
        return (data - median) / (iqr + 1e-8)
    
    X_train_norm = normalize(X_train, X_median, X_iqr)
    X_val_norm = normalize(X_val, X_median, X_iqr)
    X_test_norm = normalize(X_test, X_median, X_iqr)
    
    Y_train_norm = normalize(Y_train, Y_median, Y_iqr)
    Y_val_norm = normalize(Y_val, Y_median, Y_iqr)
    Y_test_norm = normalize(Y_test, Y_median, Y_iqr)
    
    # Prepare normalization parameters dictionary
    norm_params = {
        'X_median': X_median,
        'X_iqr': X_iqr,
        'Y_median': Y_median,
        'Y_iqr': Y_iqr,
        'n_lags': n_lags,
        'shift_i': shift_i,
        'feature_cols': ['Bx1', 'By1', 'Bz1', 'Bx2', 'By2', 'Bz2', 'B_mag', 'B2_mag'],
        'target_cols': ['Fx', 'Fy', 'Fz']
    }
    
    # Convert to tensors with explicit size check
    def safe_tensor_convert(x, y):
        assert x.shape[0] == y.shape[0], f"Size mismatch! X: {x.shape[0]}, Y: {y.shape[0]}"
        return torch.FloatTensor(x), torch.FloatTensor(y)
    
    X_train_t, Y_train_t = safe_tensor_convert(X_train_norm, Y_train_norm)
    X_val_t, Y_val_t = safe_tensor_convert(X_val_norm, Y_val_norm)
    X_test_t, Y_test_t = safe_tensor_convert(X_test_norm, Y_test_norm)
    
    return (X_train_t, Y_train_t,
            X_val_t, Y_val_t,
            X_test_t, Y_test_t,
            norm_params)



# 2. Enhanced Model Architecture with Separate Heads
class ForceCalibrationModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # Shared backbone
        self.shared = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3)
        )
        
        # Component-specific heads
        self.fx_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 1))
        
        self.fy_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1))
        
        self.fz_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 1))
        
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
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    
    # Modified scheduler without verbose
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.7,
        patience=50,
        min_lr=1e-6
    )
    
    best_val_loss = float('inf')
    history = {'train': [], 'val': [], 'components': [], 'lr': []}
    
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
        
        # Track learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        
        # Manual LR change detection
        old_lr = current_lr
        scheduler.step(avg_val)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"\nLearning rate reduced from {old_lr:.2e} to {new_lr:.2e}")
        
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
            print(f'Epoch {epoch+1}/{epochs} | LR: {current_lr:.2e}')
            print(f'Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}')
            print(f'Component MAEs: Fx={val_comps[0]:.4f}, Fy={val_comps[1]:.4f}, Fz={val_comps[2]:.4f}\n')
    
    return history

# 5. Evaluation and Visualization
def evaluate_model(model, test_loader, norm_params):
    model.eval()
    predictions, truths = [], []
    
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x)
            predictions.append(pred.numpy())
            truths.append(y.numpy())
    
    # Denormalize predictions and truths
    def denormalize(data, median, iqr):
        return data * iqr + median
    
    preds = np.concatenate(predictions)
    trues = np.concatenate(truths)
    
    preds_denorm = denormalize(preds, norm_params['Y_median'], norm_params['Y_iqr'])
    trues_denorm = denormalize(trues, norm_params['Y_median'], norm_params['Y_iqr'])
    
    # Calculate metrics on denormalized data
    errors = trues_denorm - preds_denorm
    mae = np.mean(np.abs(errors), axis=0)
    rmse = np.sqrt(np.mean(errors**2, axis=0))
    r2 = 1 - np.sum(errors**2, axis=0)/np.sum((trues_denorm - np.mean(trues_denorm, axis=0))**2, axis=0)
    
    return preds_denorm, trues_denorm, mae, rmse, r2

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
    (X_train, Y_train, X_val, Y_val, 
     X_test, Y_test, norm_params) = prepare_data('percy81.csv', n_lags=3)
    
    # Save normalization parameters
    np.save('normalization_params.npy', norm_params)
    print("Saved normalization parameters to normalization_params.npy")
    
    # Print normalization parameters for verification
    print("\nNormalization Parameters:")
    print(f"X median: {norm_params['X_median']}")
    print(f"X IQR: {norm_params['X_iqr']}")
    print(f"Y median: {norm_params['Y_median']}")
    print(f"Y IQR: {norm_params['Y_iqr']}")
    print(f"Time lags: {norm_params['n_lags']}")
    print(f"Time shift: {norm_params['shift_i']}")
    # Create loaders
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=512, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=256)
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=256)
    
    # Initialize model
    model = ForceCalibrationModel(X_train.shape[1])
    
    # Train
    print("Starting training...")
    history = train_model(model, train_loader, val_loader, epochs=1500)
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Evaluate
    preds, trues, mae, rmse, r2 = evaluate_model(model, test_loader, norm_params)
    # Save model in TorchScript format
    traced_model = torch.jit.trace(model, torch.randn(1, X_train.shape[1]))
    traced_model.save('force_calibration_model_optimized.pt')
    print("Saved TorchScript model for MCU deployment")
    # Also save regular model if needed
    torch.save(model.state_dict(), 'force_calibration_model.pth')
    # Results
    print("\n=== Final Test Results ===")
    print(f"MAE:  {mae}")
    print(f"RMSE: {rmse}")
    print(f"R²:   {r2}")
    
    # Plot
    plot_results(history, preds, trues)

if __name__ == "__main__":
    main()

