    main()
  File "/home/cardio/Documents/camera_tracking/cali_tracking/collocated_CLIK/0723/ForceSensor.py", line 151, in main
    force = sen.receive_data()
            ^^^^^^^^^^^^^^^^^^
  File "/home/cardio/Documents/camera_tracking/cali_tracking/collocated_CLIK/0723/ForceSensor.py", line 67, in receive_data
    line = self.comm.readline().decode('utf-8').strip()
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xf4 in position 12: invalid continuation byte
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

    def normalize_input(x, median, iqr):
        """Normalize using robust scaling"""
        return (x - median) / (iqr + 1e-8)

    def predict_force(self, bx, by, bz, bx2, by2, bz2):
        """Normalize inputs and predict force"""
        X_new = np.array([[bx, by, bz, bx2, by2, bz2]], dtype=np.float32)
        #X_norm = (X_new - self.X_median) / self.X_std
        #X_new = np.array([features], dtype=np.float32)
        X_norm = self.normalize_input(X_new, self.X_median, self.X_iqr)
        with torch.no_grad():
            Y_norm = self.model(torch.from_numpy(X_norm)).numpy()[0]
        
        return (Y_norm * self.Y_std) + self.Y_median  # [Fx, Fy, Fz]


    def send_data(self, pressures, commands):
        # Format desired values as strings with 3 decimal places
        serial_str = f"{pressures[0]:.3f},{pressures[1]:.3f},{pressures[2]:.3f},{commands[0]:.1f},{commands[1]:.1f},{commands[2]:.1f},{commands[3]:.1f}\n"  

        # print(serial_str)
        # Send serial command to Teensy over serial
        self.comm.write(serial_str.encode())  

    def closePort(self):
        self.comm.close()



def main():
    np.set_printoptions(precision=2, suppress=True)

    
    sen_con = SensorConnect('/dev/ttyUSB0', 115200)
    sen = SensorComm(sen_con)
    time.sleep(.1)

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
