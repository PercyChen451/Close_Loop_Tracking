import numpy as np
import torch
import serial
import time
from collections import deque
from collections import defaultdict

# Load normalization and model
norm_params = np.load('normalization_params.npy', allow_pickle=True).item()
X_mean = norm_params['mean'].astype(np.float32)
X_std = norm_params['std'].astype(np.float32)
Y_mean = norm_params.get('Y_mean', 0).astype(np.float32)
Y_std = norm_params.get('Y_std', 1).astype(np.float32)
n_lags = norm_params.get('n_lags', 3)  # Get number of lags used during training

model = torch.jit.load('force_calibration_model_optimized.pt')
model.eval()
kf = KalmanFilter3D(process_noise=0.01, measurement_noise=0.1)
# Serial port configuration
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 115200
TIMEOUT = 1

# Data smoothing and history buffer
SMOOTHING_WINDOW = 5
HISTORY_BUFFER_SIZE = n_lags + 1  # Need to store current + n_lags previous samples

# Initialize buffers
bx_history = deque(maxlen=HISTORY_BUFFER_SIZE)
by_history = deque(maxlen=HISTORY_BUFFER_SIZE)
bz_history = deque(maxlen=HISTORY_BUFFER_SIZE)
bx2_history = deque(maxlen=HISTORY_BUFFER_SIZE)
by2_history = deque(maxlen=HISTORY_BUFFER_SIZE)
bz2_history = deque(maxlen=HISTORY_BUFFER_SIZE)

# Smoothing buffers
bx_buffer = deque(maxlen=SMOOTHING_WINDOW)
by_buffer = deque(maxlen=SMOOTHING_WINDOW)
bz_buffer = deque(maxlen=SMOOTHING_WINDOW)
bx2_buffer = deque(maxlen=SMOOTHING_WINDOW)
by2_buffer = deque(maxlen=SMOOTHING_WINDOW)
bz2_buffer = deque(maxlen=SMOOTHING_WINDOW)
class KalmanFilter3D:
    def __init__(self, process_noise=0.01, measurement_noise=0.1):
        """
        Initialize 3D Kalman filter for force vectors (Fx, Fy, Fz)
        
        Parameters:
        - process_noise: Process noise covariance (how much we expect the force to change)
        - measurement_noise: Measurement noise covariance (how much we trust the sensor)
        """
        # State vector [Fx, Fy, Fz, dFx, dFy, dFz]
        self.state = np.zeros(6)
        
        # State covariance matrix
        self.covariance = np.eye(6)
        
        # Process noise covariance
        self.Q = np.eye(6) * process_noise
        
        # Measurement noise covariance
        self.R = np.eye(3) * measurement_noise
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix (we only observe position)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
    
    def predict(self):
        """Predict next state"""
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        return self.state[:3]  # Return only position components
    
    def update(self, measurement):
        """Update state with new measurement"""
        # Kalman gain
        S = self.H @ self.covariance @ self.H.T + self.R
        K = self.covariance @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        y = measurement - self.H @ self.state
        self.state = self.state + K @ y
        self.covariance = (np.eye(6) - K @ self.H) @ self.covariance
        
        return self.state[:3]
        
def calibrate_force_sensor(sample_delay=0.01, serial_port='/dev/ttyUSB0', baud_rate=115200):
    """Collects 100 samples from serial-connected force sensor and returns average offsets."""
    import serial  # Local import to avoid dependency if not using serial
    
    print("Starting calibration - keep sensor at rest...")
    print("Collecting 500 samples", end='', flush=True)
    
    # Initialize data storage
    samples = defaultdict(list)
    ser = None
    
    try:
        # Initialize serial connection
        ser = serial.Serial(serial_port, baud_rate, timeout=1)
        time.sleep(2)  # Allow time for connection
        
        # Collect samples
        for i in range(500):
            try:
                line = ser.readline().decode('ascii', errors='ignore').strip()
                values = list(map(float, line.split(',')))
                
                if len(values) == 6:  # Expecting bx1,by1,bz1,bx2,by2,bz2
                    samples['bx1'].append(values[0])
                    samples['by1'].append(values[1])
                    samples['bz1'].append(values[2])
                    samples['bx2'].append(values[3])
                    samples['by2'].append(values[4])
                    samples['bz2'].append(values[5])
                    
                    if (i+1) % 10 == 0:
                        print('.', end='', flush=True)
                
                time.sleep(sample_delay)
                    
            except (ValueError, IndexError):
                continue  # Skip bad readings
        
        # Calculate offsets
        offsets = {k: np.mean(v) for k, v in samples.items()}
        
        print("\nCalibration complete!")
        print("Calculated offsets:")
        for axis, offset in offsets.items():
            print(f"{axis}: {offset:.4f}")
            
        return offsets
        
    except Exception as e:
        print(f"\nCalibration failed: {str(e)}")
        return None
        
    finally:
        if ser and ser.is_open:
            ser.close()

def parse_serial_line(line):
    """Parse serial line like 'bx,by,bz,bx2,by2,bz2'"""
    try:
        parts = list(map(float, line.strip().split(',')))
        if len(parts) == 6:
            return parts
    except (ValueError, AttributeError):
        pass
    return None

def predict_force(bx, by, bz, bx2, by2, bz2):
    """Normalize inputs and predict force using time-lagged features"""
    # Add current sample to history buffers
    bx_history.append(bx)
    by_history.append(by)
    bz_history.append(bz)
    bx2_history.append(bx2)
    by2_history.append(by2)
    bz2_history.append(bz2)
    
    # Only predict when we have enough history
    if len(bx_history) < HISTORY_BUFFER_SIZE:
        return np.zeros(3)  # Return zeros until we have enough data
    
    # Create feature vector with current + lagged samples
    features = []
    for i in range(HISTORY_BUFFER_SIZE):
        features.extend([
            bx_history[i], by_history[i], bz_history[i],
            bx2_history[i], by2_history[i], bz2_history[i]
        ])
    
    X_new = np.array([features], dtype=np.float32)
    X_norm = (X_new - X_mean) / X_std
    
    with torch.no_grad():
        Y_norm = model(torch.from_numpy(X_norm)).numpy()[0]
    
    return (Y_norm * Y_std) + Y_mean  # [Fx, Fy, Fz]

def main():
    # Initialize serial connection
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
    print(f"Connected to {SERIAL_PORT}, waiting for data...")
    
    offsets = calibrate_force_sensor()
    if offsets is None:
        print("Failed to calibrate, exiting...")
        return
    
    try:
        while True:
            line = ser.readline().decode('ascii', errors='ignore')
            data = parse_serial_line(line)
            
            if data is not None:
                # Apply offsets
                bx = data[0] - offsets['bx1']
                by = data[1] - offsets['by1']
                bz = data[2] - offsets['bz1']
                bx2 = data[3] - offsets['bx2']
                by2 = data[4] - offsets['by2']
                bz2 = data[5] - offsets['bz2']
                
                # Apply smoothing (optional)
                bx_buffer.append(bx)
                by_buffer.append(by)
                bz_buffer.append(bz)
                bx2_buffer.append(bx2)
                by2_buffer.append(by2)
                bz2_buffer.append(bz2)
                
                smoothed_bx = np.mean(bx_buffer) if bx_buffer else bx
                smoothed_by = np.mean(by_buffer) if by_buffer else by
                smoothed_bz = np.mean(bz_buffer) if bz_buffer else bz
                smoothed_bx2 = np.mean(bx2_buffer) if bx2_buffer else bx2
                smoothed_by2 = np.mean(by2_buffer) if by2_buffer else by2
                smoothed_bz2 = np.mean(bz2_buffer) if bz2_buffer else bz2
                
                # Predict force
                raw_force = predict_force(
                    smoothed_bx, smoothed_by, smoothed_bz,
                    smoothed_bx2, smoothed_by2, smoothed_bz2
                )
                kf.predict()
                filtered_force = kf.update(raw_force)
                # Print results
                print(f"\rFx: {filtered_force[0]:.4f} N | Fy: {filtered_force[1]:.4f} N | Fz: {filtered_force[2]:.4f} N", 
                end='', flush=True)
                
            time.sleep(0.01)  # Small delay to prevent CPU overload
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        ser.close()

if __name__ == "__main__":
    main()
