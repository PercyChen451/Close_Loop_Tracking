from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

# --- 1. Enhanced Data Preparation (Upgraded for 6-DOF) ---
def prepare_data(filepath, n_lags=5, test_size=0.15, val_size=0.15, shift_i=2):
    """
    Loads data, adds 12 features (B-fields + magnitudes), 
    and creates 6-step sequences (12 features * 6 steps = 72-dim input).
    """
    # Load and verify data
    data = pd.read_csv(filepath)
    print(f"Original data shape: {data.shape}")
    
    # Apply time shift
    print(f"Applying time shift of {shift_i} samples")
    
    # 9 B-field features
    features = data[['Bx1', 'By1', 'Bz1', 'Bx2', 'By2', 'Bz2', 'Bx3', 'By3', 'Bz3']].values[shift_i:]
    # 6 Target outputs
    targets = data[['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']].values[:-shift_i]
    
    # Verify alignment
    assert len(features) == len(targets), \
        f"Alignment failed! Features: {len(features)}, Targets: {len(targets)}"
    
    # Feature engineering (B-magnitudes)
    def add_features(feats):
        B_mag = np.sqrt(feats[:,0]**2 + feats[:,1]**2 + feats[:,2]**2)
        B2_mag = np.sqrt(feats[:,3]**2 + feats[:,4]**2 + feats[:,5]**2)
        B3_mag = np.sqrt(feats[:,6]**2 + feats[:,7]**2 + feats[:,8]**2)
        # Returns (N, 12) array
        return np.column_stack((feats, B_mag, B2_mag, B3_mag))
    
    features = add_features(features)
    print(f"Features shape after engineering: {features.shape}")
    
    # Create sequences
    def create_sequences(feats, targs, n_lags):
        X, Y = [], []
        # n_lags=5 means i starts at 5.
        # First sequence is feats[0:6] (6 steps) to predict targs[5]
        for i in range(n_lags, len(feats)):
            X.append(feats[i-n_lags:i+1].flatten())
            Y.append(targs[i])
        return np.array(X), np.array(Y)
    
    X, Y = create_sequences(features, targets, n_lags)
    # With n_lags=5, X.shape[1] will be 12 features * (5+1) steps = 72
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
        # Add epsilon to IQR to prevent division by zero
        return median, iqr + 1e-8 
    
    X_median, X_iqr = calculate_robust_params(X_train)
    Y_median, Y_iqr = calculate_robust_params(Y_train)
    
    # Apply normalization
    def normalize(data, median, iqr):
        return (data - median) / iqr
    
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
        'feature_cols': ['Bx1', 'By1', 'Bz1', 'Bx2', 'By2', 'Bz2', 'Bx3', 'By3', 'Bz3', 'B_mag', 'B2_mag', 'B3_mag'],
        'target_cols': ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
    }
    
    # Convert to tensors
    def safe_tensor_convert(x, y):
        return torch.FloatTensor(x), torch.FloatTensor(y)
    
    X_train_t, Y_train_t = safe_tensor_convert(X_train_norm, Y_train_norm)
    X_val_t, Y_val_t = safe_tensor_convert(X_val_norm, Y_val_norm)
    X_test_t, Y_test_t = safe_tensor_convert(X_test_norm, Y_test_norm)
    
    return (X_train_t, Y_train_t,
            X_val_t, Y_val_t,
            X_test_t, Y_test_t,
            norm_params)

# --- 2. Enhanced Model Architecture (Upgraded for 6-DOF) ---
class ForceCalibrationModel(nn.Module):
    def __init__(self, input_size, output_size=6):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4)
        )
        
        # Component-specific heads
        # Use ModuleList for a flexible number of outputs
        self.heads = nn.ModuleList([
            self._build_head(512, 256) for _ in range(output_size)
        ])
        
        self._init_weights()

    def _build_head(self, in_features, hidden_size, depth=2):
        layers = []
        for i in range(depth - 1):
            layers.extend([
                nn.Linear(in_features if i == 0 else hidden_size, hidden_size),
                nn.SiLU() # SiLU is a high-performing activation
            ])
        layers.append(nn.Linear(hidden_size, 1))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        shared_features = self.shared(x)
        # Pass shared features to each head and concatenate outputs
        return torch.cat([head(shared_features) for head in self.heads], dim=1)

# --- 3. Custom Weighted Loss Function (From your Code 2) ---
class WeightedHuberLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        # Ensure weights are a tensor
        self.weights = torch.tensor(weights)

    def forward(self, pred, target):
        # Use Huber loss for robustness to outliers
        loss = F.huber_loss(pred, target, reduction='none')
        
        # Apply weights
        # Move weights to the same device as the loss tensor
        weighted_loss = loss * self.weights.to(loss.device)
        
        # Return the mean of the weighted loss
        return weighted_loss.mean()

# --- 4. Enhanced Training Process ---
def train_model(model, train_loader, val_loader, epochs=500):
    
    # Weights from your 'ForceTorquePredictor' code
    output_weights = [1.0, 1.0, 0.6, 0.03, 0.03, 0.1] 
    criterion = WeightedHuberLoss(weights=output_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    
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
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Update scheduler
        scheduler.step()
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'\nEpoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} (Best: {best_val_loss:.6f})')
            print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.2e}')
            
    return history

# --- 5. Evaluation and Visualization (Upgraded for 6-DOF) ---
def evaluate_model(model, test_loader, norm_params):
    model.eval()
    predictions, truths = [], []
    
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x)
            predictions.append(pred.cpu().numpy())
            truths.append(y.cpu().numpy())
    
    preds = np.concatenate(predictions)
    trues = np.concatenate(truths)
    
    # Denormalize predictions and truths
    def denormalize(data, median, iqr):
        return (data * iqr) + median
    
    Y_median = norm_params['Y_median']
    Y_iqr = norm_params['Y_iqr']
    
    preds_denorm = denormalize(preds, Y_median, Y_iqr)
    trues_denorm = denormalize(trues, Y_median, Y_iqr)
    
    # Calculate metrics on denormalized data
    errors = trues_denorm - preds_denorm
    mae = np.mean(np.abs(errors), axis=0)
    rmse = np.sqrt(np.mean(errors**2, axis=0))
    
    # Calculate R^2 for each output
    ss_res = np.sum(errors**2, axis=0)
    ss_tot = np.sum((trues_denorm - np.mean(trues_denorm, axis=0))**2, axis=0)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    return preds_denorm, trues_denorm, mae, rmse, r2

def plot_results(history, preds, trues, norm_params):
    plt.figure(figsize=(18, 12))
    
    # Plot Loss curves
    plt.subplot(3, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training & Validation Loss')
    plt.semilogy() # Use log scale for loss

    # Plot Learning Rate
    plt.subplot(3, 2, 2)
    plt.plot(history['lr'], label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.title('Learning Rate Schedule')

    target_cols = norm_params['target_cols']
    
    # Predictions vs True - Forces
    plt.subplot(3, 2, 3)
    for i in range(3): # Fx, Fy, Fz
        plt.scatter(trues[:500, i], preds[:500, i], alpha=0.3, label=target_cols[i])
    plt.plot([-1, 1], [-1, 1], 'r--', label='Ideal') # Add identity line
    plt.xlabel('True Values (Force)')
    plt.ylabel('Predictions (Force)')
    plt.legend()
    plt.title('Predictions vs True (Forces)')
    plt.axis('equal')
    plt.grid(True)

    # Predictions vs True - Torques
    plt.subplot(3, 2, 4)
    for i in range(3, 6): # Tx, Ty, Tz
        plt.scatter(trues[:500, i], preds[:500, i], alpha=0.3, label=target_cols[i])
    plt.plot([-0.1, 0.1], [-0.1, 0.1], 'r--', label='Ideal') # Add identity line
    plt.xlabel('True Values (Torque)')
    plt.ylabel('Predictions (Torque)')
    plt.legend()
    plt.title('Predictions vs True (Torques)')
    plt.axis('equal')
    plt.grid(True)

    # Plot Force Components Over Time (sample)
    plt.subplot(3, 2, 5)
    plt.plot(trues[:500, 0], label='True Fx', linestyle='--')
    plt.plot(preds[:500, 0], label='Pred Fx', alpha=0.8)
    plt.plot(trues[:500, 1], label='True Fy', linestyle='--')
    plt.plot(preds[:500, 1], label='Pred Fy', alpha=0.8)
    plt.plot(trues[:500, 2], label='True Fz', linestyle='--')
    plt.plot(preds[:500, 2], label='Pred Fz', alpha=0.8)
    plt.legend(ncol=3)
    plt.xlabel('Sample Index')
    plt.ylabel('Force (N)')
    plt.title('Test Set Performance (Forces)')
    
    # Plot Torque Components Over Time (sample)
    plt.subplot(3, 2, 6)
    plt.plot(trues[:500, 3], label='True Tx', linestyle='--')
    plt.plot(preds[:500, 3], label='Pred Tx', alpha=0.8)
    plt.plot(trues[:500, 4], label='True Ty', linestyle='--')
    plt.plot(preds[:500, 4], label='Pred Ty', alpha=0.8)
    plt.plot(trues[:500, 5], label='True Tz', linestyle='--')
    plt.plot(preds[:500, 5], label='Pred Tz', alpha=0.8)
    plt.legend(ncol=3)
    plt.xlabel('Sample Index')
    plt.ylabel('Torque (Nm)')
    plt.title('Test Set Performance (Torques)')

    plt.tight_layout()
    plt.savefig('training_results.png')
    print("\nSaved training results plot to training_results.png")

# --- Main Execution ---
def main():
    # --- THIS IS THE KEY FIX (n_lags=5) ---
    # This creates the 72-dimensional input (12 features * 6 steps)
    # that your team identified as necessary for "memory effects".
    N_LAGS = 5 
    
    # Use the dataset you specified
    FILE_PATH = 'cali_1_11-11.csv'
    
    if not os.path.exists(FILE_PATH):
        print(f"Error: File not found at {FILE_PATH}")
        print("Please make sure 'cali_1_11-11.csv' is in the same directory.")
        return

    # Prepare data
    (X_train, Y_train, X_val, Y_val, 
     X_test, Y_test, norm_params) = prepare_data(FILE_PATH, n_lags=N_LAGS)
    
    # Save normalization parameters
    np.save('normalization_params.npy', norm_params)
    print("Saved normalization parameters to normalization_params.npy")
    
    # Print normalization parameters for verification
    print("\n--- Normalization Parameters ---")
    print(f"Feature columns: {norm_params['feature_cols']}")
    print(f"Number of features: {len(norm_params['feature_cols'])}")
    print(f"Target columns: {norm_params['target_cols']}")
    print(f"Time lags (n_lags): {norm_params['n_lags']} (results in {N_LAGS+1} time steps)")
    print(f"Time shift: {norm_params['shift_i']}")
    
    # Create loaders
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=512, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=1024, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=1024, shuffle=False)
    
    # Initialize model
    INPUT_SIZE = X_train.shape[1]
    OUTPUT_SIZE = Y_train.shape[1]
    
    model = ForceCalibrationModel(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE)
    print(f"\nModel initialized with input size: {INPUT_SIZE} and output size: {OUTPUT_SIZE}")
    
    # Train
    print("Starting training...")
    history = train_model(model, train_loader, val_loader, epochs=500)
    
    # Load best model
    print("\nLoading best model from 'best_model.pth' for evaluation...")
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Evaluate
    preds, trues, mae, rmse, r2 = evaluate_model(model, test_loader, norm_params)
    
    # Save model in TorchScript format
    try:
        example_input = torch.randn(1, INPUT_SIZE)
        traced_model = torch.jit.trace(model, example_input)
        traced_model.save('force_calibration_model_optimized.pt')
        print("Saved TorchScript model to 'force_calibration_model_optimized.pt'")
    except Exception as e:
        print(f"Could not save TorchScript model: {e}")
    
    # Results
    print("\n--- Final Test Results (Denormalized) ---")
    print("Metrics per axis: [Fx, Fy, Fz, Tx, Ty, Tz]")
    print(f"MAE:  {np.array2string(mae, precision=4)}")
    print(f"RMSE: {np.array2string(rmse, precision=4)}")
    print(f"R²:   {np.array2string(r2, precision=4)}")
    
    # Plot
    plot_results(history, preds, trues, norm_params)

if __name__ == "__main__":
    main()