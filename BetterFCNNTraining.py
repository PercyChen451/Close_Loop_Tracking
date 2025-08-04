^^^^^^^^^^^^^^^
RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.FloatTensor [128, 1]], which is output 0 of AsStridedBackward0, is at version 3; expected version 1 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True).
                                         and at val_loss += criterion(pred, y).item(), criterion got an error marker


from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. Enhanced Data Preparation with Normalization Parameters
def prepare_data(filepath, n_lags=5, test_size=0.15, val_size=0.15, shift_i=2):
    # [Previous data loading and feature engineering code remains the same until after train/val/test split]
    
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
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1)
        )
        
        # Component-specific heads with isolated parameters
        self.fx_head = self._build_head(512, 128)
        self.fy_head = self._build_head(512, 256, depth=3)  # Deeper for Fy
        self.fz_head = self._build_head(512, 128)
    
    def _build_head(self, in_features, hidden_size, depth=2):
        layers = []
        for i in range(depth-1):
            layers.extend([
                nn.Linear(in_features if i==0 else hidden_size, hidden_size),
                nn.SiLU()
            ])
        layers.append(nn.Linear(hidden_size, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        shared = self.shared(x)
        return torch.cat([
            self.fx_head(shared),
            self.fy_head(shared),
            self.fz_head(shared)
        ], dim=1)
    
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
    # Separate optimizers for each component head
    optimizers = {
        'fx': optim.AdamW(model.fx_head.parameters(), lr=1e-4, weight_decay=1e-5),
        'fy': optim.AdamW(model.fy_head.parameters(), lr=5e-4, weight_decay=1e-5),  # Higher LR for Fy
        'fz': optim.AdamW(model.fz_head.parameters(), lr=1e-4, weight_decay=1e-5),
        'shared': optim.AdamW(model.shared.parameters(), lr=1e-4, weight_decay=1e-5)
    }
    
    # Component-specific schedulers
    schedulers = {
        'fx': CosineAnnealingLR(optimizers['fx'], T_max=epochs),
        'fy': OneCycleLR(optimizers['fy'], max_lr=1e-3, 
                        steps_per_epoch=len(train_loader), epochs=epochs),
        'fz': CosineAnnealingLR(optimizers['fz'], T_max=epochs),
        'shared': CosineAnnealingLR(optimizers['shared'], T_max=epochs)
    }
    
    best_val_loss = float('inf')
    history = {'train': [], 'val': [], 'components': [], 'lr': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_comps = torch.zeros(3)
        
        for x, y in train_loader:
            # Zero all gradients
            for opt in optimizers.values():
                opt.zero_grad()
            
            # Forward pass
            pred = model(x)
            
            # Component-specific losses
            losses = {
                'fx': F.l1_loss(pred[:,0], y[:,0]),  # MAE for Fx
                'fy': F.huber_loss(pred[:,1], y[:,1]),  # More robust for Fy
                'fz': F.l1_loss(pred[:,2], y[:,2])  # MAE for Fz
            }
            
            # Physics regularization (optional)
            physics_reg = 0.1 * F.relu(-pred[:,1]).mean()  # Penalize negative Fy
            total_loss = sum(losses.values()) + physics_reg
            
            # Backward passes
            losses['fx'].backward(retain_graph=True)
            optimizers['fx'].step()
            
            losses['fy'].backward(retain_graph=True)
            optimizers['fy'].step()
            
            losses['fz'].backward(retain_graph=True)
            optimizers['fz'].step()
            
            optimizers['shared'].step()
            
            # Track metrics
            train_loss += total_loss.item()
            train_comps += torch.stack([losses['fx'], losses['fy'], losses['fz']]).detach()
        
        # Update schedulers
        for sched in schedulers.values():
            sched.step()
        
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
        
        # Track learning rates
        current_lrs = {
            'fx': optimizers['fx'].param_groups[0]['lr'],
            'fy': optimizers['fy'].param_groups[0]['lr'],
            'fz': optimizers['fz'].param_groups[0]['lr']
        }
        
        # Save best model
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), 'best_model.pth')
        
        # Store history
        history['train'].append(avg_train)
        history['val'].append(avg_val)
        history['components'].append(val_comps.numpy())
        history['lr'].append(current_lrs)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'\nEpoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}')
            print('Component MAEs:')
            print(f'  Fx: {val_comps[0]:.6f} (LR: {current_lrs["fx"]:.2e})')
            print(f'  Fy: {val_comps[1]:.6f} (LR: {current_lrs["fy"]:.2e})')
            print(f'  Fz: {val_comps[2]:.6f} (LR: {current_lrs["fz"]:.2e})')
    
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
     X_test, Y_test, norm_params) = prepare_data('percy81.csv', n_lags=5)
    
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
    history = train_model(model, train_loader, val_loader, epochs=1000)
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Evaluate
        preds, trues, mae, rmse, r2 = evaluate_model(model, test_loader, norm_params)
    
    # Results
    print("\n=== Final Test Results ===")
    print(f"MAE:  {mae}")
    print(f"RMSE: {rmse}")
    print(f"R²:   {r2}")
    
    # Plot
    plot_results(history, preds, trues)

if __name__ == "__main__":
    main()




