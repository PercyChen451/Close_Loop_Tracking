(venv) cardio@cardio-PC:~/Documents/camera_tracking$ /home/cardio/Documents/Haptics-main/venv/bin/python /home/cardio/Documents/Force_sensor/Force_Sensor_Cali/NNtraining.py
Shift 4
Epoch 100/2000, Train Loss: 0.020754, Val Loss: 0.030244
Epoch 200/2000, Train Loss: 0.011749, Val Loss: 0.027273
Epoch 300/2000, Train Loss: 0.009437, Val Loss: 0.025518
Epoch 400/2000, Train Loss: 0.007617, Val Loss: 0.024409
Epoch 500/2000, Train Loss: 0.006758, Val Loss: 0.023918
Epoch 600/2000, Train Loss: 0.006193, Val Loss: 0.022727
Epoch 700/2000, Train Loss: 0.005392, Val Loss: 0.022453
Epoch 800/2000, Train Loss: 0.005119, Val Loss: 0.022449
Epoch 900/2000, Train Loss: 0.004242, Val Loss: 0.022641
Epoch 1000/2000, Train Loss: 0.005093, Val Loss: 0.023956
Epoch 1100/2000, Train Loss: 0.004417, Val Loss: 0.022085
Epoch 1200/2000, Train Loss: 0.003554, Val Loss: 0.022205
Epoch 1300/2000, Train Loss: 0.004082, Val Loss: 0.022386
Epoch 1400/2000, Train Loss: 0.003397, Val Loss: 0.021401
Epoch 1500/2000, Train Loss: 0.003717, Val Loss: 0.022489
Epoch 1600/2000, Train Loss: 0.003583, Val Loss: 0.022494
Epoch 1700/2000, Train Loss: 0.003308, Val Loss: 0.022253
Epoch 1800/2000, Train Loss: 0.003134, Val Loss: 0.023039
Epoch 1900/2000, Train Loss: 0.003128, Val Loss: 0.021332
Epoch 2000/2000, Train Loss: 0.003710, Val Loss: 0.022249
Traceback (most recent call last):
  File "/home/cardio/Documents/Force_sensor/Force_Sensor_Cali/NNtraining.py", line 211, in <module>
    SS_res = np.sum((Y_true_for_plot - Y_pred)**2, axis=0)
                     ~~~~~~~~~~~~~~~~^~~~~~~~
ValueError: operands could not be broadcast together with shapes (8088,3) (8091,3) 
(venv) cardio@cardio-PC:~/Documents/camera_tracking$ /home/cardio/Documents/Haptics-main/venv/bin/python /home/cardio/Documents/Force_sensor/Force_Sensor_Cali/NNtraining.py
Shift 4
Traceback (most recent call last):
  File "/home/cardio/Documents/Force_sensor/Force_Sensor_Cali/NNtraining.py", line 123, in <module>
    outputs = model(inputs)
              ^^^^^^^^^^^^^
  File "/home/cardio/Documents/Haptics-main/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/cardio/Documents/Haptics-main/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/cardio/Documents/Force_sensor/Force_Sensor_Cali/NNtraining.py", line 101, in forward
    x = self.fc1(x)
        ^^^^^^^^^^^
  File "/home/cardio/Documents/Haptics-main/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/cardio/Documents/Haptics-main/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/cardio/Documents/Haptics-main/venv/lib/python3.12/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x12 and 6x128)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load data from file
file = 'PercyFS729.csv'
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

# Zero data (keep your original approach)
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

# Prepare input and output data
X = np.column_stack((bx, by, bz, bx2, by2, bz2, bx, by, bz, bx2, by2, bz2))
Y = np.column_stack((fx, fy, fz))

# Normalize input data (keep your original approach)
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_norm = (X - X_mean) / X_std

# NEW: Normalize output data for better training
Y_mean = np.mean(Y, axis=0)
Y_std = np.std(Y, axis=0)
Y_norm = (Y - Y_mean) / Y_std

# Split data into training, validation, and test sets
X_train, X_temp, Y_train, Y_temp = train_test_split(X_norm, Y_norm, test_size=0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

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

# Define the neural network architecture (keep your original architecture)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(6, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128,128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 3)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

model = NeuralNetwork()

# Training settings
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 2000

# Training loop
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

# Save normalization parameters (modified to include Y normalization)
np.save('normalization_params.npy', {
    'mean': X_mean,
    'std': X_std,
    'Y_mean': Y_mean,  # NEW: Added output normalization
    'Y_std': Y_std     # NEW: Added output normalization
})

# Plot training and validation loss (keep your original graphing)
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
    # Get predictions for all normalized data
    Y_pred_norm = model(torch.FloatTensor(X_norm)).numpy()
    Y_pred = (Y_pred_norm * Y_std) + Y_mean  # Reverse normalization for comparison

# Since we used time-lagged features (n_lags=3), we need to adjust the comparison
# The first 3 samples of Y don't have corresponding predictions because we don't have enough history
Y_true_for_plot = Y[3:]  # This should match Y_pred's length

# Verify shapes match
print(f"Y_true_for_plot shape: {Y_true_for_plot.shape}")
print(f"Y_pred shape: {Y_pred.shape}")

# Plot predicted vs actual (only if shapes match)
if Y_true_for_plot.shape == Y_pred.shape:
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

    # Compute R² and RMSE (using adjusted true values)
    SS_res = np.sum((Y_true_for_plot - Y_pred)**2, axis=0)
    SS_tot = np.sum((Y_true_for_plot - np.mean(Y_true_for_plot, axis=0))**2, axis=0)
    R2 = 1 - (SS_res / SS_tot)
    print('R² for [Fx, Fy, Fz]:')
    print(R2)

    errors = Y_true_for_plot - Y_pred
    rmse = np.sqrt(np.mean(errors**2, axis=0))
    print('RMSE for [Fx, Fy, Fz]:')
    print(rmse)

    # Plot error scatter plots
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Y_true_for_plot[:, 0], Y_true_for_plot[:, 1], errors[:, 2])
    ax.set_xlabel('Fx')
    ax.set_ylabel('Fy')
    ax.set_zlabel('Error in Fz')
    plt.title('Error in Fz vs Fx and Fy')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Y_true_for_plot[:, 0], Y_true_for_plot[:, 2], errors[:, 1])
    ax.set_xlabel('Fx')
    ax.set_ylabel('Fz')
    ax.set_zlabel('Error in Fy')
    plt.title('Error in Fy vs Fx and Fz')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Y_true_for_plot[:, 1], Y_true_for_plot[:, 2], errors[:, 0])
    ax.set_xlabel('Fy')
    ax.set_ylabel('Fz')
    ax.set_zlabel('Error in Fx')
    plt.title('Error in Fx vs Fy and Fz')
    plt.show()
else:
    print(f"Shape mismatch: Y_true_for_plot {Y_true_for_plot.shape} vs Y_pred {Y_pred.shape}")
    print("Cannot compute metrics due to shape mismatch")
# Save optimized model (keep your original format)
traced_model = torch.jit.trace(model, torch.randn(1, 6))  # Input shape: (1, 6)
traced_model.save('force_calibration_model_optimized.pt')
