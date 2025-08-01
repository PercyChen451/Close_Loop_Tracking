import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import os

# Load data from file
file = 'percy81.csv'
data = pd.read_csv(file)

# Extract data columns
time = data['Time'].values
fx = data['Fx'].values
fy = data['Fy'].values
fz = data['Fz'].values
tx = data['Tx'].values
ty = data['Ty'].values
tz = data['Tz'].values
bx = data['Bx1'].values
by = data['By1'].values
bz = data['Bz1'].values
bx2 = data['Bx2'].values
by2 = data['By2'].values
bz2 = data['Bz2'].values

# Zero data
fx = fx - fx[0]
fy = fy - fy[0]
fz = fz - fz[0]
bx = bx - bx[0]
by = by - by[0]
bz = bz - bz[0]
bx2 = bx2 - bx2[0]
by2 = by2 - by2[0]
bz2 = bz2 - bz2[0]

shift_i = 4
print("Shift", shift_i)
fx = fx[:-shift_i]
fy = fy[:-shift_i]
fz = fz[:-shift_i]
bx = bx[shift_i:]
by = by[shift_i:]
bz = bz[shift_i:]
bx2 = bx2[shift_i:]
by2 = by2[shift_i:]
bz2 = bz2[shift_i:]
time = time[:-shift_i]

# Parameters for time-lagged features
n_lags = 3
n_features = 6

def create_sequences(data, targets, n_lags):
    X, Y = [], []
    for i in range(n_lags, len(data)):
        X.append(data[i-n_lags:i+1].reshape(-1, n_features))
        Y.append(targets[i])
    return np.array(X), np.array(Y)

current_input = np.column_stack((bx, by, bz, bx2, by2, bz2))
Y = np.column_stack((fx, fy, fz))

X, Y_aligned = create_sequences(current_input, Y, n_lags)

# Normalization
X_mean = np.mean(X, axis=(0,1))
X_std = np.std(X, axis=(0,1))
X_norm = (X - X_mean) / X_std

Y_mean = np.mean(Y_aligned, axis=0)
Y_std = np.std(Y_aligned, axis=0)
Y_norm = (Y_aligned - Y_mean) / Y_std

def continuous_train_test_split(X, Y, test_size=0.2, val_segments=3, val_segment_size=0.05):
    n_samples = len(X)
    test_split = int(n_samples * (1 - test_size))
    
    X_test = X[test_split:]
    Y_test = Y[test_split:]
    
    X_train_full = X[:test_split]
    Y_train_full = Y[:test_split]
    
    val_indices = []
    segment_length = int(len(X_train_full) * val_segment_size)
    
    rng = np.random.RandomState(42)
    possible_starts = np.arange(0, len(X_train_full) - segment_length)
    val_starts = rng.choice(possible_starts, size=val_segments, replace=False)
    
    X_val, Y_val = [], []
    for start in sorted(val_starts):
        end = start + segment_length
        X_val.append(X_train_full[start:end])
        Y_val.append(Y_train_full[start:end])
    
    X_val = np.concatenate(X_val)
    Y_val = np.concatenate(Y_val)
    
    train_mask = np.ones(len(X_train_full), dtype=bool)
    for start in sorted(val_starts):
        end = start + segment_length
        train_mask[start:end] = False
    
    X_train = X_train_full[train_mask]
    Y_train = Y_train_full[train_mask]
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test

X_train, X_val, X_test, Y_train, Y_val, Y_test = continuous_train_test_split(
    X_norm, Y_norm, test_size=0.2, val_segments=3, val_segment_size=0.05
)

# Enhanced LSTM Model with Dual Attention and Residual Connections
class EnhancedForceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.4)
        
        # Temporal Attention
        self.temporal_attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Feature Attention
        self.feature_attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(), 
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        # Residual blocks
        self.res_block1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*2),
            nn.BatchNorm1d(hidden_size*2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size*2, hidden_size),
            nn.BatchNorm1d(hidden_size)
        )
        
        self.res_block2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.BatchNorm1d(hidden_size//2),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        self.output = nn.Linear(hidden_size//2, 3)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        # Temporal attention
        attn_weights = self.temporal_attn(lstm_out)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        # Feature attention
        feature_weights = self.feature_attn(context)
        context = context * feature_weights
        
        # Residual learning
        residual = context
        out = self.res_block1(context)
        out = out + residual
        
        out = self.res_block2(out)
        return self.output(out)

# Ensemble Model
class ForceEnsemble(nn.Module):
    def __init__(self, num_models=5, input_size=n_features, hidden_size=256, num_layers=4):
        super().__init__()
        self.models = nn.ModuleList([
            EnhancedForceLSTM(input_size, hidden_size, num_layers) 
            for _ in range(num_models)
        ])
        
    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)

# Bayesian Optimization Setup
def train_model(config):
    model = EnhancedForceLSTM(
        input_size=n_features,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"]
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = nn.MSELoss()
    
    # Convert data to tensors
    X_train_t = torch.FloatTensor(X_train)
    Y_train_t = torch.FloatTensor(Y_train)
    X_val_t = torch.FloatTensor(X_val)
    Y_val_t = torch.FloatTensor(Y_val)
    
    train_dataset = TensorDataset(X_train_t, Y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    
    for epoch in range(100):  # Short training for tuning
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, Y_val_t).item()
        
        tune.report(loss=val_loss)

def run_bayesian_optimization():
    config = {
        "hidden_size": tune.choice([128, 256, 512]),
        "num_layers": tune.choice([2, 3, 4]),
        "lr": tune.loguniform(1e-5, 1e-3),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        "batch_size": tune.choice([32, 64, 128])
    }
    
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=100,
        grace_period=10,
        reduction_factor=2
    )
    
    reporter = CLIReporter(
        metric_columns=["loss", "training_iteration"]
    )
    
    result = tune.run(
        train_model,
        resources_per_trial={"cpu": 4, "gpu": 0.5},
        config=config,
        num_samples=50,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="force_prediction_tune"
    )
    
    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial validation loss: {best_trial.last_result['loss']}")
    
    return best_trial.config

# Main Training Function
def main_train(best_config=None):
    if best_config is None:
        print("Running Bayesian optimization...")
        best_config = run_bayesian_optimization()
    
    # Initialize ensemble
    ensemble = ForceEnsemble(
        num_models=5,
        input_size=n_features,
        hidden_size=best_config["hidden_size"],
        num_layers=best_config["num_layers"]
    )
    
    # Convert data to tensors
    X_train_t = torch.FloatTensor(X_train)
    Y_train_t = torch.FloatTensor(Y_train)
    X_val_t = torch.FloatTensor(X_val)
    Y_val_t = torch.FloatTensor(Y_val)
    X_test_t = torch.FloatTensor(X_test)
    Y_test_t = torch.FloatTensor(Y_test)
    
    train_dataset = TensorDataset(X_train_t, Y_train_t)
    val_dataset = TensorDataset(X_val_t, Y_val_t)
    test_dataset = TensorDataset(X_test_t, Y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=best_config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_config["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=best_config["batch_size"])
    
    # Train each model in the ensemble
    for i, model in enumerate(ensemble.models):
        print(f"\nTraining model {i+1} in ensemble")
        
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=best_config["lr"], 
            weight_decay=best_config["weight_decay"]
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=20, factor=0.5
        )
        
        criterion = nn.MSELoss()
        best_val_loss = float('inf')
        patience = 0
        max_patience = 50
        
        for epoch in range(500):
            model.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()
            
            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                torch.save(model.state_dict(), f"best_model_{i}.pth")
            else:
                patience += 1
                if patience >= max_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.6f}, Val Loss: {val_loss:.6f}")
        
        # Load best model
        model.load_state_dict(torch.load(f"best_model_{i}.pth"))
    
    # Evaluate ensemble
    ensemble.eval()
    with torch.no_grad():
        # Test set evaluation
        test_outputs = ensemble(X_test_t)
        test_loss = criterion(test_outputs, Y_test_t).item()
        
        # Full dataset predictions
        Y_pred_norm = ensemble(torch.FloatTensor(X_norm)).numpy()
        Y_pred = (Y_pred_norm * Y_std) + Y_mean
    
    print(f"\nFinal Ensemble Test Loss: {test_loss:.6f}")
    
    # Plotting
    Y_true_for_plot = (Y_norm * Y_std) + Y_mean
    
    plt.figure(figsize=(10, 6))
    plt.plot(Y_true_for_plot[:, 0], label='True Fx')
    plt.plot(Y_pred[:, 0], label='Predicted Fx')
    plt.xlabel('Sample')
    plt.ylabel('Force (N)')
    plt.legend()
    plt.title('Fx: True vs Predicted')
    plt.savefig('fx_prediction.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(Y_true_for_plot[:, 1], label='True Fy')
    plt.plot(Y_pred[:, 1], label='Predicted Fy')
    plt.xlabel('Sample')
    plt.ylabel('Force (N)')
    plt.legend()
    plt.title('Fy: True vs Predicted')
    plt.savefig('fy_prediction.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(Y_true_for_plot[:, 2], label='True Fz')
    plt.plot(Y_pred[:, 2], label='Predicted Fz')
    plt.xlabel('Sample')
    plt.ylabel('Force (N)')
    plt.legend()
    plt.title('Fz: True vs Predicted')
    plt.savefig('fz_prediction.png')
    plt.close()
    
    # 3D Plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Y_true_for_plot[:, 0], Y_true_for_plot[:, 1], Y_true_for_plot[:, 2], 
               c='b', marker='o', label='True Force', alpha=0.5)
    ax.scatter(Y_pred[:, 0], Y_pred[:, 1], Y_pred[:, 2], 
               c='r', marker='x', label='Predicted Force', alpha=0.5)
    ax.set_xlabel('Fx')
    ax.set_ylabel('Fy')
    ax.set_zlabel('Fz')
    ax.legend()
    plt.title('True vs Predicted Forces (3D)')
    plt.savefig('3d_force_prediction.png')
    plt.close()
    
    # Save models and normalization parameters
    torch.save(ensemble.state_dict(), 'force_ensemble.pth')
    np.save('normalization_params.npy', {
        'mean': X_mean,
        'std': X_std,
        'Y_mean': Y_mean,
        'Y_std': Y_std,
        'n_lags': n_lags,
        'n_features': n_features
    })
    
    print("\nTraining complete. Models and plots saved.")

if __name__ == "__main__":
    # Uncomment to run Bayesian optimization first
    # best_config = run_bayesian_optimization()
    
    # For direct training with known good parameters
    best_config = {
        "hidden_size": 256,
        "num_layers": 3,
        "lr": 0.0005,
        "weight_decay": 1e-4,
        "batch_size": 64
    }
    
    main_train(best_config)
