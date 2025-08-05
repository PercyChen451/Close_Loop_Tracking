# Import relevant libraries
import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import numpy as np
import torch
import serial
import time
from collections import deque
# from LivePlot import Plotting

# Global pressure vector in N/mm^2
kinPressures = np.array([5, 2, 1])/1000
goalCoords = np.array( [0, 20, 20] )

class SensorConnect(serial.Serial):

    #Initialize host computer serial communication settings
    def __init__(self, port, baud):
        serial.Serial.__init__(self, port=port, baudrate=baud, bytesize=serial.EIGHTBITS,
                               stopbits=serial.STOPBITS_ONE, parity=serial.PARITY_NONE, timeout=3)


class SensorComm:

    def __init__(self, chain):
        self.comm = chain
    
        # Serial port configuration
        self.PORT = 'COM13'  # Adjust to your Teensy port
        self.BAUD_RATE = 115200
        self.TIMEOUT = 0.1
        self.current_data = [0 ,0, 0, 0, 0, 0]
        norm_params = np.load('normalization_params.npy', allow_pickle=True).item()

        self.X_median = norm_params['X_median'].astype(np.float32)
        self.X_iqr = norm_params['X_iqr'].astype(np.float32)
        self.Y_median = norm_params['Y_median'].astype(np.float32)
        self.Y_iqr = norm_params['Y_iqr'].astype(np.float32)
        self.n_lags = norm_params['n_lags']
        self.HISTORY_BUFFER_SIZE = self.n_lags + 1  # This needs to be defined before any methods try to use it
        self.history = deque(maxlen = self.HISTORY_BUFFER_SIZE)
        
        # Initialize buffers
        self.SMOOTHING_WINDOW = 5
        self.bx_buffer = deque(maxlen=self.SMOOTHING_WINDOW)
        self.by_buffer = deque(maxlen=self.SMOOTHING_WINDOW)
        self.bz_buffer = deque(maxlen=self.SMOOTHING_WINDOW)
        self.bx2_buffer = deque(maxlen=self.SMOOTHING_WINDOW)
        self.by2_buffer = deque(maxlen=self.SMOOTHING_WINDOW)
        self.bz2_buffer = deque(maxlen=self.SMOOTHING_WINDOW)

        self.model = torch.jit.load('force_calibration_model_optimized.pt')
        self.model.eval()
        self.model.float()  # Ensure model uses float32

        self.baseForce = np.array([2.390, -1.545, 15.920])



    def clear_initial_messages(self):
        """Clear any initialization messages from the sensor"""
        start_time = time.time()
        while time.time() - start_time < 2.0:  # Wait up to 2 seconds
            if self.comm.in_waiting > 0:
                line = self.comm.readline().decode('utf-8', errors='ignore').strip()
                if line and 'X1' in line:  # Look for data header
                    break
            time.sleep(0.1)


    def read_valid_line(self):
        """Read until getting a valid data line"""
        while self.comm.in_waiting > 0:
            line = self.comm.readline().decode('utf-8', errors='ignore').strip()
            if line and not any(x in line for x in ['Initializing', 'sensor', 'ready']):
                if len(line.split(',')) == 6:
                    return line
        return None

    def receive_data(self):
        # clear data buffer, eventually change to queue with pop fucntion 
        data_buffer = []
        

        while self.comm.in_waiting > 0:
            line = self.comm.readline().decode('utf-8').strip()
            line_v = line.split(",")

            try:
                bx, by, bz, bx2, by2, bz2 = [float(val) for val in line_v]

                self.bx_buffer.append(bx)
                self.by_buffer.append(by)
                self.bz_buffer.append(bz)
                self.bx2_buffer.append(bx2)
                self.by2_buffer.append(by2)
                self.bz2_buffer.append(bz2)

                smoothed_bx = np.mean(self.bx_buffer) if self.bx_buffer else bx
                smoothed_by = np.mean(self.by_buffer) if self.by_buffer else by
                smoothed_bz = np.mean(self.bz_buffer) if self.bz_buffer else bz
                smoothed_bx2 = np.mean(self.bx2_buffer) if self.bx2_buffer else bx2
                smoothed_by2 = np.mean(self.by2_buffer) if self.by2_buffer else by2
                smoothed_bz2 = np.mean(self.bz2_buffer) if self.bz2_buffer else bz2

                # Predict force
                force_vec = self.predict_force(
                    smoothed_bx, smoothed_by, smoothed_bz,
                    smoothed_bx2, smoothed_by2, smoothed_bz2
                )
                data_buffer.append(force_vec - self.baseForce)
                
            except ValueError:
                print("Invalid data received:", line)
        
        # if the buffer is empty, return the most current data value
        if len(data_buffer) == 0:
            # print("No data received, sending latest value instead")
            return self.current_data
        
        # otherwise, update the current data value and return the buffer
        self.current_data = data_buffer[-1]

        return self.current_data

    def normalize_input(self, x, median, iqr):
        """Normalize using robust scaling"""
        return (x - median) / (iqr + 1e-8)
    
    def denormalize_output(self, y, median, iqr):
        """Denormalize predictions"""
    
        return y * iqr + median
    
    def predict_force(self, bx, by, bz, bx2, by2, bz2):
        """Predict with proper feature construction"""
        # Calculate derived features (must match training)
        b_mag = np.sqrt(bx**2 + by**2 + bz**2)
        b2_mag = np.sqrt(bx2**2 + by2**2 + bz2**2)
        
        # Create current feature vector (8 elements)
        current_features = np.array([
            bx, by, bz, bx2, by2, bz2, b_mag, b2_mag
        ], dtype=np.float32)
        
        # Add to history
        self.history.append(current_features)
        
        # Wait until we have enough history
        if len(self.history) < self.history.maxlen:
            return np.zeros(3)  # Return zeros until buffer fills
        
        # Create flattened input vector
        features = np.concatenate(self.history)  # Shape: (n_lags+1)*8
        
        # Reshape to what model expects (1, features)
        X_new = features.reshape(1, -1).astype(np.float32)
        
        # Verify shape matches normalization params
        if X_new.shape[1] != len(self.X_median):
            raise ValueError(
                f"Feature dimension mismatch! "
                f"Got {X_new.shape[1]}, expected {len(self.X_median)}. "
                f"Check n_lags (currently {self.n_lags})"
            )
        
        # Normalize and predict
        X_norm = (X_new - self.X_median) / (self.X_iqr + 1e-8)
        with torch.no_grad():
            Y_norm = self.model(torch.from_numpy(X_norm)).numpy()[0]
        
        return Y_norm * self.Y_iqr + self.Y_median

    def calibrate_sensor(self, duration=3.0):
        """Calibrate with proper feature construction"""
        print("Starting calibration...")
        samples = []
        start_time = time.time()
        while time.time() - start_time < duration:
            line = self.read_valid_line()
            if line:
                try:
                    bx, by, bz, bx2, by2, bz2 = map(float, line.split(','))
                    # Calculate derived features
                    b_mag = np.sqrt(bx**2 + by**2 + bz**2)
                    b2_mag = np.sqrt(bx2**2 + by2**2 + bz2**2)
                    samples.append([bx, by, bz, bx2, by2, bz2, b_mag, b2_mag])
                except ValueError:
                    continue
        if len(samples) >= self.HISTORY_BUFFER_SIZE:
            # Build proper time-lagged features
            features = []
            for i in range(len(samples) - self.HISTORY_BUFFER_SIZE + 1):
                time_series = []
                for j in range(self.HISTORY_BUFFER_SIZE):
                    time_series.extend(samples[i+j])
                features.append(time_series)
            X_cal = np.array(features, dtype=np.float32)
            X_norm = self.normalize_input(X_cal, self.X_median, self.X_iqr)
            with torch.no_grad():
                forces = self.model(torch.from_numpy(X_norm)).numpy()
                self.baseForce = np.mean(forces, axis=0)
            print(f"Calibration complete. Base force: {self.baseForce}")
        else:
            print("Insufficient calibration data")
            self.baseForce = np.zeros(3)
        return self.baseForce

    def send_data(self, pressures, commands):
        # Format desired values as strings with 3 decimal places
        serial_str = f"{pressures[0]:.3f},{pressures[1]:.3f},{pressures[2]:.3f},{commands[0]:.1f},{commands[1]:.1f},{commands[2]:.1f},{commands[3]:.1f}\n"  

        # print(serial_str)
        # Send serial command to Teensy over serial
        self.comm.write(serial_str.encode())  

    def closePort(self):
        self.comm.close()



def main():
    norm_params = np.load('normalization_params.npy', allow_pickle=True).item()
    print(norm_params['X_median'].shape)  # Should be (6,)
    print(norm_params['X_median'])

    np.set_printoptions(precision=2, suppress=True)

    
    sen_con = SensorConnect('/dev/ttyUSB0', 115200)
    sen = SensorComm(sen_con)
    time.sleep(.1)
    #sen.calibrate_sensor()
    time_start = time.time()
    time_prev = time_start
    while True:
        time_now = time.time()
        dt = time_prev - time_now
        time_prev = time_now

        force = sen.receive_data()
        print(force, dt)
        time.sleep(0.01)

if __name__ == "__main__":
    main()
