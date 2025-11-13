# Import relevant libraries
import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time as time

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class ArduinoConnect(serial.Serial):
    #Initialize host computer serial communication settings
    def __init__(self, port, baud):
        serial.Serial.__init__(self, port=port, baudrate=baud, bytesize=serial.EIGHTBITS,
                               stopbits=serial.STOPBITS_ONE, parity=serial.PARITY_NONE, timeout=3)


class FoceSensor:
    def __init__(self, chain, model_path, n_steps, input_size, hidden_dim=256, device=None):
        self.comm = chain
    
        # Serial port configuration
        self.PORT = 'COM13'  # Adjust to your Teensy port
        self.BAUD_RATE = 115200
        self.TIMEOUT = 0.1
        self.current_force = np.zeros((1,3))
        self.mag_buffer = []
        self.baseline_pred = np.zeros((1,6))
        self.n_vals = 3
        self.force_q = [0*3]

        # Model params
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_steps = n_steps

       # Build model
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # 3 forces + 3 torques
        ).to(self.device)

        # Load model weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Model weights loaded from {model_path}")

        # Load scalers
        self.scaler_X = StandardScaler()
        self.scaler_X.mean_ = np.load("model_params/X_mean.npy")
        self.scaler_X.scale_ = np.load("model_params/X_std.npy")

        self.scaler_y = StandardScaler()
        self.scaler_y.mean_ = np.load("model_params/y_mean.npy")
        self.scaler_y.scale_ = np.load("model_params/y_std.npy")

        # cali 11-11
        self.baseline_mag = np.array([-865.0,-273.0,6310.0,-69.0,-9.0,6453.0,-803.0,1154.0,8518.0], dtype=np.float32)
        # 8-25_1
        # self.baseline_mag = np.array([-730.0, -478.0, 6615.0, 10.0, -43.0, 7036.0, -803.0, 1142.0, 8962.0], dtype=np.float32)
        # percy 86
        # self.baseline_mag = np.array([-811.0, -429.0, 6634.0, 43.0, -127.0, 7178.0, -973.0, 1209.0, 8699.0], dtype=np.float32)

    def _destandardize(self, y):
        return y * self.scaler_y.scale_ + self.scaler_y.mean_

    def predict_force(self, mag_data_raw):
        # zero magnetometer data
        mag_data = mag_data_raw - self.baseline_mag

        # Standardize input first (per original 9 features)
        mag_data_std = (mag_data - self.scaler_X.mean_) / self.scaler_X.scale_

        # Build sequences
        X_seq = []
        X_seq.append(mag_data_std.flatten())
        X_seq = np.array(X_seq, dtype=np.float32)

        # Predict
        with torch.no_grad():
            X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
            y_pred_std = self.model(X_tensor).cpu().numpy()

        # Destandardize predictions
        y_pred = self._destandardize(y_pred_std) - self.baseline_pred

        return y_pred[0]
    
    def calibrate(self, num_vals = 100):
        init_values = [] 
        while len(init_values) < num_vals:
            if self.comm.in_waiting > 0:
                line = self.comm.readline().decode('utf-8').strip()
                line_v = line.split(",")

                try:
                    values = [float(val) for val in line_v]
                    init_values.append(values)
                    # print(values)
                except ValueError:
                    print("Invalid data received:", line)


        self.baseline_mag = np.mean(init_values, axis=0)
        baseline_reading = np.tile(self.baseline_mag, (self.n_steps,1))
        self.baseline_pred = self.predict_force(baseline_reading)

        return self.baseline_mag, self.baseline_pred
    
    def receive_data(self):

        while self.comm.in_waiting > 0:
            line = self.comm.readline().decode('utf-8').strip()
            line_v = line.split(",")

            try:
                self.mag_buffer.append([float(val) for val in line_v])
            except ValueError:
                print("Invalid data received:", line)
        
        # if the buffer has enough values, compute force
        if len(self.mag_buffer ) >= self.n_steps:
            # Use only the last n steps based on the model requirements
            self.mag_buffer  = self.mag_buffer[-self.n_steps:]
            mag_array = np.array(self.mag_buffer)
            pred_force = self.predict_force(mag_array)
            self.current_force = pred_force

            
            # # add force value to force queue
            # self.force_q.append(pred_force)
            # self.force_q = self.force_q[-self.n_vals:]

            # # compute mean filter for past n forces
            # mean_force = np.mean(self.force_q)
            # self.current_force = mean_force

            return self.current_force
        
        # otherwise, return the latest predcted value
        return self.current_force

    def send_data(self, pressures, commands):
        # Format desired values as strings with 3 decimal places
        serial_str = f"{pressures[0]:.3f},{pressures[1]:.3f},{pressures[2]:.3f},{commands[0]:.1f},{commands[1]:.1f},{commands[2]:.1f},{commands[3]:.1f}\n"  

        # print(serial_str)
        # Send serial command to Teensy over serial
        self.comm.write(serial_str.encode())  

    def closePort(self):
        self.comm.close()

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    comm = ArduinoConnect("COM5",250000)
    sensor = FoceSensor(
        comm,
        model_path="model_params/force_torque_model.pth",
        n_steps=5,
        input_size=5*9,  # n_steps * 9 magnetometer values
        hidden_dim=512
    )
    time.sleep(.2)
    print( sensor.calibrate(100) )

    # time.sleep(1)
    while True:
        force = sensor.receive_data()
        print("force:", force, flush=True)
        # print()
        time.sleep(.05)

    # test_seq =  np.array(  [[-812.0, -424.0, 6630.0, 43.0, -126.0, 7176.0, -974.0, 1215.0, 8695.0],
    #             [-809.0, -428.0, 6620.0, 35.0, -122.0, 7189.0, -974.0, 1215.0, 8700.0],
    #             [-811.0, -418.0, 6633.0, 44.0, -133.0, 7191.0, -972.0, 1207.0, 8706.0],
    #             [-818.0, -424.0, 6633.0, 53.0, -119.0, 7178.0, -971.0, 1210.0, 8701.0],
    #             [-818.0, -432.0, 6633.0, 43.0, -123.0, 7177.0, -974.0, 1210.0, 8694.0]] )
    
    # force = sensor.predict_force(test_seq)
    # print("predicted_force:", force)

    