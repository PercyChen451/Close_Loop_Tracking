def main():
    np.set_printoptions(precision=3, suppress=True)
    
    # Initialize sensor first with longer timeout
    print("Initializing force sensor...")
    sen_connect = SensorConnect('/dev/ttyUSB0', 115200)
    sensor = SensorComm(sen_connect)
    
    # Wait for sensor to stabilize
    time.sleep(2)
    
    # Perform calibration
    print("Calibrating force sensor...")
    sensor.calibrate_sensor()
    
    # Initialize other components
    print("Initializing Arduino...")
    connect = ArduinoConnect('/dev/ttyACM0', 250000)
    arduino = ArduinoComm(connect)
    time.sleep(1)
    
    print("Initializing camera tracker...")
    tracker = Cam_Tracker()
    dyn = CollocatedDynamics()
    
    # Setup trajectory
    path_coords = dyn.generate_circle(15, 30, 5)
    
    # Main control loop
    try:
        while True:
            # Get sensor data
            force = sensor.receive_data()
            
            # Get robot data
            robot_data = arduino.receive_data()
            
            # Update tracker
            tracker.update(rigid_pred)
            t_tip, R_tip = tracker.get_pose()
            
            # Run control algorithm
            (u_volumes, commands, backbone_pred, 
             rigid_pred, tip_pred, dt) = dyn.CLIK(
                robot_data, t_tip*1000, R_tip, force, path_coords
            )
            
            # Send commands
            arduino.send_data(u_volumes, commands)
            
            time.sleep(0.005)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Cleanup
        arduino.send_data(np.zeros(3), [0, 0, 0, 0])
        sensor.close()
        connect.close()
        print("Resources released")


import numpy as np
import torch
import time
from collections import deque
import serial

class SensorComm:
    def __init__(self, chain):
        self.comm = chain
        self.current_data = np.array([0, 0, 0], dtype=np.float32)
        self.baseForce = np.array([0, 0, 0], dtype=np.float32)
        
        # Load model and normalization parameters
        self.load_model()
        
        # Initialize buffers
        self.init_buffers()
        
        # Clear any initial messages
        self.clear_initial_messages()

    def load_model(self):
        """Load the trained model and normalization parameters"""
        try:
            self.norm_params = np.load('normalization_params.npy', allow_pickle=True).item()
            self.X_mean = self.norm_params['mean'].astype(np.float32)
            self.X_std = self.norm_params['std'].astype(np.float32)
            self.Y_mean = self.norm_params.get('Y_mean', 0).astype(np.float32)
            self.Y_std = self.norm_params.get('Y_std', 1).astype(np.float32)
            
            self.model = torch.jit.load('force_calibration_model_optimized.pt')
            self.model.eval()
            self.model.float()  # Ensure model uses float32
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def init_buffers(self):
        """Initialize data smoothing buffers"""
        self.SMOOTHING_WINDOW = 5
        self.bx_buffer = deque(maxlen=self.SMOOTHING_WINDOW)
        self.by_buffer = deque(maxlen=self.SMOOTHING_WINDOW)
        self.bz_buffer = deque(maxlen=self.SMOOTHING_WINDOW)
        self.bx2_buffer = deque(maxlen=self.SMOOTHING_WINDOW)
        self.by2_buffer = deque(maxlen=self.SMOOTHING_WINDOW)
        self.bz2_buffer = deque(maxlen=self.SMOOTHING_WINDOW)

    def clear_initial_messages(self):
        """Clear any initialization messages from the sensor"""
        start_time = time.time()
        while time.time() - start_time < 2.0:  # Wait up to 2 seconds
            if self.comm.in_waiting > 0:
                line = self.comm.readline().decode('utf-8', errors='ignore').strip()
                if line and 'X1' in line:  # Look for data header
                    break
            time.sleep(0.1)

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
            X_cal = np.array(samples)
            X_norm = (X_cal - self.X_mean) / self.X_std
            with torch.no_grad():
                Y_norm = self.model(torch.from_numpy(X_norm).float()).numpy()
            self.baseForce = np.mean(Y_norm * self.Y_std + self.Y_mean, axis=0)
            print(f"Calibration complete. Base force: {self.baseForce}")
        else:
            print("Warning: No valid calibration data collected")
            self.baseForce = np.array([0, 0, 0], dtype=np.float32)
        
        return self.baseForce

    def read_valid_line(self):
        """Read until getting a valid data line"""
        while self.comm.in_waiting > 0:
            line = self.comm.readline().decode('utf-8', errors='ignore').strip()
            if line and not any(x in line for x in ['Initializing', 'sensor', 'ready']):
                if len(line.split(',')) == 6:
                    return line
        return None

    def receive_data(self):
        """Get the latest force data with proper error handling"""
        line = self.read_valid_line()
        if not line:
            return self.current_data
        
        try:
            bx, by, bz, bx2, by2, bz2 = [float(x) for x in line.split(',')]
            
            # Update buffers
            self.bx_buffer.append(bx)
            self.by_buffer.append(by)
            self.bz_buffer.append(bz)
            self.bx2_buffer.append(bx2)
            self.by2_buffer.append(by2)
            self.bz2_buffer.append(bz2)
            
            # Get smoothed values
            smoothed = [
                np.mean(self.bx_buffer) if self.bx_buffer else bx,
                np.mean(self.by_buffer) if self.by_buffer else by,
                np.mean(self.bz_buffer) if self.bz_buffer else bz,
                np.mean(self.bx2_buffer) if self.bx2_buffer else bx2,
                np.mean(self.by2_buffer) if self.by2_buffer else by2,
                np.mean(self.bz2_buffer) if self.bz2_buffer else bz2
            ]
            
            # Predict force
            X_new = np.array([smoothed], dtype=np.float32)
            X_norm = (X_new - self.X_mean) / self.X_std
            with torch.no_grad():
                Y_norm = self.model(torch.from_numpy(X_norm)).numpy()[0]
            force_vec = (Y_norm * self.Y_std) + self.Y_mean
            
            self.current_data = force_vec - self.baseForce
            return self.current_data
            
        except Exception as e:
            print(f"Data processing error: {e}")
            return self.current_data

    def close(self):
        """Close the serial connection"""
        self.comm.close()
