(venv) cardio@cardio-PC:~/Documents/camera_tracking/cali_tracking/collocated_CLIK/0723$ /home/cardio/Documents/camera_tracking/venv/bin/python /home/cardio/Documents/camera_tracking/cali_tracking/collocated_CLIK/0723/ForceSensor.py
(32,)
Starting calibration - keep sensor at rest...
Traceback (most recent call last):
  File "/home/cardio/Documents/camera_tracking/cali_tracking/collocated_CLIK/0723/ForceSensor.py", line 215, in <module>
    main()
  File "/home/cardio/Documents/camera_tracking/cali_tracking/collocated_CLIK/0723/ForceSensor.py", line 202, in main
    sen.calibrate_sensor()
  File "/home/cardio/Documents/camera_tracking/cali_tracking/collocated_CLIK/0723/ForceSensor.py", line 169, in calibrate_sensor
    X_norm = self.normalize_input(X_new, self.X_median, self.X_iqr)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/cardio/Documents/camera_tracking/cali_tracking/collocated_CLIK/0723/ForceSensor.py", line 129, in normalize_input
    return (x - median) / (iqr + 1e-8)
            ~~^~~~~~~~
ValueError: operands could not be broadcast together with shapes (1,81,6) (32,) 
(venv) cardio@cardio-PC:~/Documents/camera_tracking/cali_tracking/collocated_CLIK/0723$ /home/cardio/Documents/camera_tracking/venv/bin/python /home/cardio/Documents/camera_tracking/cali_tracking/collocated_CLIK/0723/ForceSensor.py
(32,)
[  -28.         -1165.         -7774.          -832.
    81.         -8927.          7862.06232486  8965.85665174
   -28.         -1165.         -7774.          -832.
    81.         -8927.          7862.06232486  8965.85665174
   -28.         -1165.         -7774.          -832.
    81.         -8927.          7862.06324701  8965.85665174
   -28.         -1165.         -7774.          -832.
    81.         -8927.          7862.06547289  8965.83124981]
Starting calibration - keep sensor at rest...
Traceback (most recent call last):
  File "/home/cardio/Documents/camera_tracking/cali_tracking/collocated_CLIK/0723/ForceSensor.py", line 216, in <module>
    main()
  File "/home/cardio/Documents/camera_tracking/cali_tracking/collocated_CLIK/0723/ForceSensor.py", line 203, in main
    sen.calibrate_sensor()
  File "/home/cardio/Documents/camera_tracking/cali_tracking/collocated_CLIK/0723/ForceSensor.py", line 169, in calibrate_sensor
    X_norm = self.normalize_input(X_new, self.X_median, self.X_iqr)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/cardio/Documents/camera_tracking/cali_tracking/collocated_CLIK/0723/ForceSensor.py", line 129, in normalize_input
    return (x - median) / (iqr + 1e-8)
            ~~^~~~~~~~
ValueError: operands could not be broadcast together with shapes (1,81,6) (32,) 
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
        """Normalize inputs and predict force"""
        X_new = np.array([[bx, by, bz, bx2, by2, bz2]], dtype=np.float32)
        #X_norm = (X_new - self.X_median) / self.X_std
        #X_new = np.array([features], dtype=np.float32)
        X_norm = self.normalize_input(X_new, self.X_median, self.X_iqr)
        with torch.no_grad():
            Y_norm = self.model(torch.from_numpy(X_norm)).numpy()[0]
        
        return self.denormalize_output(Y_norm, self.Y_median, self.Y_iqr)  # [Fx, Fy, Fz]


    def calibrate_sensor(self, duration=3.0):
        """Perform sensor calibration"""
        print("Starting calibration - keep sensor at rest...")
        samples = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            line = self.read_valid_line()
            if line:
                try:
                    values = [float(x) for x in line.split(',')]
                    if len(values) == 6:
                        samples.append(values)
                except ValueError:
                    continue
        
        if samples:
            #X_cal = np.array(samples)
            #X_norm = (X_cal - self.X_median) / self.X_std
            #X_new = np.array([[bx, by, bz, bx2, by2, bz2]], dtype=np.float32)
            X_new = np.array([samples], dtype=np.float32)
            X_norm = self.normalize_input(X_new, self.X_median, self.X_iqr)
            with torch.no_grad():
                Y_norm = self.model(torch.from_numpy(X_norm)).numpy()[0]
            self.baseForce = np.mean(Y_norm * self.Y_std + self.Y_mean, axis=0)
            print(f"Calibration complete. Base force: {self.baseForce}")
        else:
            print("Warning: No valid calibration data collected")
            self.baseForce = np.array([0, 0, 0], dtype=np.float32)
        
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
    sen.calibrate_sensor()
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
