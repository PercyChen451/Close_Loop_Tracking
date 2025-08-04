import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

# Apply time shift
shift_i = 2
print(f"Applying time shift of {shift_i} samples")
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
n_lags = 3  # Number of previous timestamps to include
n_features = 6  # bx, by, bz, bx2, by2, bz2

# Create time-lagged features and properly aligned outputs
def create_sequences(data, targets, n_lags):
    X, Y = [], []
    for i in range(n_lags, len(data)):
        X.append(data[i-n_lags:i].flatten())  # Only previous n_lags points
        Y.append(targets[i])  # Output corresponds to current timestep
    return np.array(X), np.array(Y)

current_input = np.column_stack((bx, by, bz, bx2, by2, bz2))
Y = np.column_stack((fx, fy, fz))

# Verify lengths match before creating sequences
assert len(current_input) == len(Y), "Input and output lengths don't match"

X, Y_aligned = create_sequences(current_input, Y, n_lags)

# Verify we have matching lengths after sequence creation
assert len(X) == len(Y_aligned), "Sequence creation resulted in mismatched lengths"

# Split data first, then normalize
def continuous_train_test_split(X, Y, test_size=0.2, val_segments=300, val_segment_size=0.0005):
    """
    Split data into training, validation, and test sets with continuous segments
    
    Parameters:
    - test_size: fraction of data for test set (taken from end)
    - val_segments: number of validation segments
    - val_segment_size: size of each validation segment as fraction of training data
    """
    n_samples = len(X)
    test_split = int(n_samples * (1 - test_size))
    
    # Test set is last 20%
    X_test = X[test_split:]
    Y_test = Y[test_split:]
    
    # Training data is everything before test split
    X_train_full = X[:test_split]
    Y_train_full = Y[:test_split]
    
    # Create validation segments
    val_indices = []
    segment_length = int(len(X_train_full) * val_segment_size)
    
    # Randomly select starting points for validation segments
    rng = np.random.RandomState(42)
    possible_starts = np.arange(0, len(X_train_full) - segment_length)
    val_starts = rng.choice(possible_starts, size=val_segments, replace=False)
    
    # Extract validation segments
    X_val, Y_val = [], []
    for start in sorted(val_starts):
        end = start + segment_length
        X_val.append(X_train_full[start:end])
        Y_val.append(Y_train_full[start:end])
    
    X_val = np.concatenate(X_val)
    Y_val = np.concatenate(Y_val)
    
    # Remove validation segments from training data
    train_mask = np.ones(len(X_train_full), dtype=bool)
    for start in sorted(val_starts):
        end = start + segment_length
        train_mask[start:end] = False
    
    X_train = X_train_full[train_mask]
    Y_train = Y_train_full[train_mask]
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test

# Split data using continuous segments
X_train, X_val, X_test, Y_train, Y_val, Y_test = continuous_train_test_split(
    X, Y, test_size=0.2, val_segments=300, val_segment_size=0.0005
)

# Normalize using only training statistics
X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)
X_train_norm = (X_train - X_mean) / X_std
X_val_norm = (X_val - X_mean) / X_std
X_test_norm = (X_test - X_mean) / X_std

Y_mean = np.mean(Y_train, axis=0)
Y_std = np.std(Y_train, axis=0)
Y_train_norm = (Y_train - Y_mean) / Y_std
Y_val_norm = (Y_val - Y_mean) / Y_std
Y_test_norm = (Y_test - Y_mean) / Y_std

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_norm)
Y_train_tensor = torch.FloatTensor(Y_train_norm)
X_val_tensor = torch.FloatTensor(X_val_norm)
Y_val_tensor = torch.FloatTensor(Y_val_norm)
X_test_tensor = torch.FloatTensor(X_test_norm)
Y_test_tensor = torch.FloatTensor(Y_test_norm)

# Create datasets and dataloaders
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Define the improved neural network architecture
input_size = n_lags * n_features

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        
        self.fc4 = nn.Linear(64, 3)
        
    def forward(self, x):
        x = self.drop1(self.relu1(self.bn1(self.fc1(x))))
        x = self.drop2(self.relu2(self.bn2(self.fc2(x))))
        x = self.relu3(self.bn3(self.fc3(x)))
        return self.fc4(x)

model = NeuralNetwork(input_size)

# Training settings
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
epochs = 300

# Training loop with early stopping
train_losses = []
val_losses = []
best_val_loss = float('inf')
early_stop_patience = 20
patience_counter = 0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
        running_loss += loss.item()
    
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    scheduler.step(val_loss)
    
    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

# Load best model
model.load_state_dict(torch.load('best_model.pth'))

# Save normalization parameters
np.save('normalization_params.npy', {
    'X_mean': X_mean,
    'X_std': X_std,
    'Y_mean': Y_mean,
    'Y_std': Y_std,
    'n_lags': n_lags
})

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.grid(True)
plt.show()

# Evaluate on test set
model.eval()
with torch.no_grad():
    Y_pred_norm = model(X_test_tensor).numpy()
    Y_pred = (Y_pred_norm * Y_std) + Y_mean
    Y_true = (Y_test_norm * Y_std) + Y_mean

# Plot predicted vs actual for each component
fig, axes = plt.subplots(3, 1, figsize=(12, 8))
components = ['Fx', 'Fy', 'Fz']
for i, ax in enumerate(axes):
    ax.plot(Y_true[:200, i], label='True')
    ax.plot(Y_pred[:200, i], label='Predicted')
    ax.set_title(components[i])
    ax.legend()
plt.tight_layout()
plt.show()

# Error distribution plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, ax in enumerate(axes):
    errors = Y_true[:, i] - Y_pred[:, i]
    ax.hist(errors, bins=50)
    ax.set_title(f'{components[i]} Error Distribution')
plt.tight_layout()
plt.show()

# 3D plot of predicted vs actual
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Y_true[:, 0], Y_true[:, 1], Y_true[:, 2], c='b', marker='o', label='True Force', alpha=0.3)
ax.scatter(Y_pred[:, 0], Y_pred[:, 1], Y_pred[:, 2], c='r', marker='x', label='Predicted Force', alpha=0.3)
ax.set_xlabel('Fx')
ax.set_ylabel('Fy')
ax.set_zlabel('Fz')
ax.legend()
plt.title('True vs Predicted Forces (Test Set)')
plt.show()

# Compute metrics
SS_res = np.sum((Y_true - Y_pred)**2, axis=0)
SS_tot = np.sum((Y_true - np.mean(Y_true, axis=0))**2, axis=0)
R2 = 1 - (SS_res / SS_tot)
print('R² for [Fx, Fy, Fz]:')
print(R2)

errors = Y_true - Y_pred
rmse = np.sqrt(np.mean(errors**2, axis=0))
print('RMSE for [Fx, Fy, Fz]:')
print(rmse)

mae = np.mean(np.abs(errors), axis=0)
print('MAE for [Fx, Fy, Fz]:')
print(mae)

# Save model
traced_model = torch.jit.trace(model, torch.randn(1, input_size))
traced_model.save('force_calibration_model_optimized.pt')
