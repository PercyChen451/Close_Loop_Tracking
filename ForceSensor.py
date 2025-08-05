import numpy as np
import torch
import serial
import time
from collections import deque

class SensorConnect(serial.Serial):
    def __init__(self, port, baud):
        serial.Serial.__init__(self, port=port, baudrate=baud, timeout=0.1)

class SensorComm:
    def __init__(self, chain):
        self.comm = chain
        
        # Load normalization parameters
        norm_params = np.load('normalization_params.npy', allow_pickle=True).item()
        self.X_median = norm_params['X_median'].astype(np.float32)
        self.X_iqr = norm_params['X_iqr'].astype(np.float32)
        self.Y_median = norm_params['Y_median'].astype(np.float32)
        self.Y_iqr = norm_params['Y_iqr'].astype(np.float32)
        self.n_lags = norm_params['n_lags']
        
        # Initialize buffers
        self.HISTORY_BUFFER_SIZE = self.n_lags + 1
        self.history = deque(maxlen=self.HISTORY_BUFFER_SIZE)
        self.SMOOTHING_WINDOW = 5
        self.bx_buffer = deque(maxlen=self.SMOOTHING_WINDOW)
        self.by_buffer = deque(maxlen=self.SMOOTHING_WINDOW)
        self.bz_buffer = deque(maxlen=self.SMOOTHING_WINDOW)
        self.bx2_buffer = deque(maxlen=self.SMOOTHING_WINDOW)
        self.by2_buffer = deque(maxlen=self.SMOOTHING_WINDOW)
        self.bz2_buffer = deque(maxlen=self.SMOOTHING_WINDOW)

        # Load model
        self.model = torch.jit.load('force_calibration_model_optimized.pt')
        self.model.eval()
        
        self.baseForce = np.zeros(3)
        self.current_data = np.zeros(3)
        self.initial_samples = 0

    def receive_data(self):
        while self.comm.in_waiting > 0:
            line = self.comm.readline().decode('ascii', errors='ignore').strip()
            if line and line.count(',') == 5:
                try:
                    bx, by, bz, bx2, by2, bz2 = map(float, line.split(','))
                    
                    # Apply smoothing
                    self.bx_buffer.append(bx)
                    self.by_buffer.append(by)
                    self.bz_buffer.append(bz)
                    self.bx2_buffer.append(bx2)
                    self.by2_buffer.append(by2)
                    self.bz2_buffer.append(bz2)
                    
                    smoothed = [
                        np.mean(self.bx_buffer) if self.bx_buffer else bx,
                        np.mean(self.by_buffer) if self.by_buffer else by,
                        np.mean(self.bz_buffer) if self.bz_buffer else bz,
                        np.mean(self.bx2_buffer) if self.bx2_buffer else bx2,
                        np.mean(self.by2_buffer) if self.by2_buffer else by2,
                        np.mean(self.bz2_buffer) if self.bz2_buffer else bz2
                    ]
                    
                    # Predict force
                    force = self.predict_force(*smoothed)
                    self.current_data = force - self.baseForce
                    
                except ValueError:
                    continue
        
        return self.current_data

    def read_valid_line(self):
        """Read until getting a valid data line"""
        while self.comm.in_waiting > 0:
            line = self.comm.readline().decode('utf-8', errors='ignore').strip()
            if line and not any(x in line for x in ['Initializing', 'sensor', 'ready']):
                if len(line.split(',')) == 6:
                    return line
        return None

    def predict_force(self, bx, by, bz, bx2, by2, bz2):
        # Calculate derived features
        b_mag = np.sqrt(bx**2 + by**2 + bz**2)
        b2_mag = np.sqrt(bx2**2 + by2**2 + bz2**2)
        
        # Create current feature vector
        current_features = np.array([bx, by, bz, bx2, by2, bz2, b_mag, b2_mag], dtype=np.float32)
        
        # Add to history
        self.history.append(current_features)
        self.initial_samples += 1
        
        # Wait until buffer fills
        if len(self.history) < self.HISTORY_BUFFER_SIZE:
            return np.zeros(3)
        
        # Create input vector
        features = np.concatenate(self.history).reshape(1, -1)
        X_norm = (features - self.X_median) / (self.X_iqr + 1e-8)
        
        with torch.no_grad():
            Y_norm = self.model(torch.from_numpy(X_norm)).numpy()[0]
        
        return Y_norm * self.Y_iqr + self.Y_median

    def read_valid_line(self):
        """Read until getting a valid data line"""
        while self.comm.in_waiting > 0:
            line = self.comm.readline().decode('utf-8', errors='ignore').strip()
            if line and not any(x in line for x in ['Initializing', 'sensor', 'ready']):
                if len(line.split(',')) == 6:
                    return line
        return None

    def calibrate_sensor(self, duration=3.0):
        print("Calibrating...")
        samples = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            line = self.read_valid_line()
            if line:
                try:
                    values = list(map(float, line.split(',')))
                    if len(values) == 6:
                        samples.append(values)
                except ValueError:
                    continue
        
        if len(samples) >= self.HISTORY_BUFFER_SIZE:
            # Process calibration samples
            features = []
            for i in range(len(samples) - self.HISTORY_BUFFER_SIZE + 1):
                time_series = []
                for j in range(self.HISTORY_BUFFER_SIZE):
                    bx, by, bz, bx2, by2, bz2 = samples[i+j]
                    b_mag = np.sqrt(bx**2 + by**2 + bz**2)
                    b2_mag = np.sqrt(bx2**2 + by2**2 + bz2**2)
                    time_series.extend([bx, by, bz, bx2, by2, bz2, b_mag, b2_mag])
                features.append(time_series)
            
            X_cal = np.array(features, dtype=np.float32)
            X_norm = (X_cal - self.X_median) / (self.X_iqr + 1e-8)
            
            with torch.no_grad():
                forces = self.model(torch.from_numpy(X_norm)).numpy()
                self.baseForce = np.mean(forces, axis=0)
            
            print(f"Calibration complete. Base force: {self.baseForce}")
        else:
            print("Insufficient calibration data")
            self.baseForce = np.zeros(3)
        
        return self.baseForce

def main():
    # Initialize serial connection
    ser = SensorConnect('/dev/ttyUSB0', 115200)
    sensor = SensorComm(ser)
    
    # Calibrate sensor
    sensor.calibrate_sensor()
    
    # Main loop
    try:
        while True:
            force = sensor.receive_data()
            print(f"Fx: {force[0]:.2f} N, Fy: {force[1]:.2f} N, Fz: {force[2]:.2f} N")
            time.sleep(0.01)
    except KeyboardInterrupt:
        ser.close()

if __name__ == "__main__":
    main()
