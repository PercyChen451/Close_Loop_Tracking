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

# Extract data columns (same as before)
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

# Zero data (same as before)
fx = fx - fx[0]
fy = fy - fy[0]
fz = fz - fz[0]
bx = bx - bx[0]
by = by - by[0]
bz = bz - bz[0]
bx2 = bx2 - bx2[0]
by2 = by2 - by2[0]
bz2 = bz2 - bz2[0]

shift_i = 2
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
n_lags = 3  # Number of previous timestamps to include
n_features = 6  # bx, by, bz, bx2, by2, bz2

# Create time-lagged features and properly aligned outputs
def create_sequences(data, targets, n_lags):
    X, Y = [], []
    for i in range(n_lags, len(data)):
        X.append(data[i-n_lags:i+1].flatten())  # Current + n_lags previous
        Y.append(targets[i])  # Output corresponds to current timestep
    return np.array(X), np.array(Y)

current_input = np.column_stack((bx, by, bz, bx2, by2, bz2))
Y = np.column_stack((fx, fy, fz))

# Verify lengths match before creating sequences
assert len(current_input) == len(Y), "Input and output lengths don't match"

X, Y_aligned = create_sequences(current_input, Y, n_lags)

# Verify we have matching lengths after sequence creation
assert len(X) == len(Y_aligned), "Sequence creation resulted in mismatched lengths"

# Normalize input data
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_norm = (X - X_mean) / X_std

# Normalize output data
Y_mean = np.mean(Y_aligned, axis=0)
Y_std = np.std(Y_aligned, axis=0)
Y_norm = (Y_aligned - Y_mean) / Y_std

# New: Split data into continuous segments for validation
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
    X_norm, Y_norm, test_size=0.2, val_segments=300, val_segment_size=0.0005
)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
Y_train_tensor = torch.FloatTensor(Y_train)
X_val_tensor = torch.FloatTensor(X_val)
Y_val_tensor = torch.FloatTensor(Y_val)
X_test_tensor = torch.FloatTensor(X_test)
Y_test_tensor = torch.FloatTensor(Y_test)

# Create datasets and dataloaders
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Define the neural network architecture (same as before)
input_size = (n_lags + 1) * n_features

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 128)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, 3)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x

model = NeuralNetwork(input_size)

# Training settings (same as before)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 300

# Training loop (same as before)
train_losses = []
val_losses = []
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
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
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

# Save normalization parameters
np.save('normalization_params.npy', {
    'mean': X_mean,
    'std': X_std,
    'Y_mean': Y_mean,
    'Y_std': Y_std,
    'n_lags': n_lags
})

# Plot training and validation loss
plt.figure()
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Evaluate on test set
model.eval()
with torch.no_grad():
    Y_pred_norm = model(torch.FloatTensor(X_norm)).numpy()
    Y_pred = (Y_pred_norm * Y_std) + Y_mean

# The predictions are already aligned with Y_aligned
Y_true_for_plot = Y_aligned

# Final verification
assert Y_true_for_plot.shape == Y_pred.shape, f"Final shape mismatch: {Y_true_for_plot.shape} vs {Y_pred.shape}"

# Plot predicted vs actual
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Y_true_for_plot[:, 0], Y_true_for_plot[:, 1], Y_true_for_plot[:, 2], c='b', marker='o', label='True Force')
ax.scatter(Y_pred[:, 0], Y_pred[:, 1], Y_pred[:, 2], c='r', marker='x', label='Predicted Force')
ax.set_xlabel('Fx')
ax.set_ylabel('Fy')
ax.set_zlabel('Fz')
ax.legend()
plt.title('True vs Predicted Forces')
plt.show()

# Compute metrics
SS_res = np.sum((Y_true_for_plot - Y_pred)**2, axis=0)
SS_tot = np.sum((Y_true_for_plot - np.mean(Y_true_for_plot, axis=0))**2, axis=0)
R2 = 1 - (SS_res / SS_tot)
print('R² for [Fx, Fy, Fz]:')
print(R2)

errors = Y_true_for_plot - Y_pred
rmse = np.sqrt(np.mean(errors**2, axis=0))
print('RMSE for [Fx, Fy, Fz]:')
print(rmse)

# Save model
traced_model = torch.jit.trace(model, torch.randn(1, input_size))
traced_model.save('force_calibration_model_optimized.pt')