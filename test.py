(venv) cardio@cardio-PC:~/Documents/camera_tracking$ /home/cardio/Documents/camera_tracking/venv/bin/python /home/cardio/Documents/Force_sensor/Force_Sensor_Cali/RealTimeWithMCU.py
Connected to /dev/ttyUSB0, waiting for data...
Starting calibration - keep sensor at rest...
Collecting 100 samples..........
Calibration complete!
Calculated offsets:
bx1: -103.7320
by1: -1019.4742
bz1: -7457.3093
bx2: -980.2577
by2: -172.6082
bz2: -8689.3402
N
Traceback (most recent call last):
  File "/home/cardio/Documents/Force_sensor/Force_Sensor_Cali/RealTimeWithMCU.py", line 164, in <module>
    main()
  File "/home/cardio/Documents/Force_sensor/Force_Sensor_Cali/RealTimeWithMCU.py", line 148, in main
    Fx, Fy, Fz = predict_force(
                 ^^^^^^^^^^^^^^
  File "/home/cardio/Documents/Force_sensor/Force_Sensor_Cali/RealTimeWithMCU.py", line 104, in predict_force
    X_norm = (X_new - X_mean) / X_std
              ~~~~~~^~~~~~~~
ValueError: operands could not be broadcast together with shapes (1,6) (24,) 

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

model = torch.jit.load('force_calibration_model_optimized.pt')
model.eval()

# Serial port configuration
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 115200
TIMEOUT = 1

# Data smoothing (optional)
SMOOTHING_WINDOW = 5
bx_buffer = deque(maxlen=SMOOTHING_WINDOW)
by_buffer = deque(maxlen=SMOOTHING_WINDOW)
bz_buffer = deque(maxlen=SMOOTHING_WINDOW)
bx2_buffer = deque(maxlen=SMOOTHING_WINDOW)
by2_buffer = deque(maxlen=SMOOTHING_WINDOW)
bz2_buffer = deque(maxlen=SMOOTHING_WINDOW)


def calibrate_force_sensor(sample_delay=0.01, serial_port='/dev/ttyUSB0', baud_rate=115200):
    """
    Collects 100 samples from serial-connected force sensor and returns average offsets.
    """
    import serial  # Local import to avoid dependency if not using serial
    
    print("Starting calibration - keep sensor at rest...")
    print("Collecting 100 samples", end='', flush=True)
    
    # Initialize data storage
    samples = defaultdict(list)
    ser = None
    
    try:
        # Initialize serial connection
        ser = serial.Serial(serial_port, baud_rate, timeout=1)
        time.sleep(2)  # Allow time for connection
        
        # Collect samples
        for i in range(100):
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
    """Normalize inputs and predict force"""
    X_new = np.array([[bx, by, bz, bx2, by2, bz2]], dtype=np.float32)
    X_norm = (X_new - X_mean) / X_std
    
    with torch.no_grad():
        Y_norm = model(torch.from_numpy(X_norm)).numpy()[0]
    
    return (Y_norm * Y_std) + Y_mean  # [Fx, Fy, Fz]

def main():
    # Initialize serial connection
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
    print(f"Connected to {SERIAL_PORT}, waiting for data...")
    

    offsets = calibrate_force_sensor()
    try:
        while True:
            line = ser.readline().decode('ascii', errors='ignore')
            data = parse_serial_line(line)
            if ser is not None:
                bx = data[0] - offsets['bx1']
                by = data[1] - offsets['by1']
                bz = data[2] - offsets['bz1']
                bx2 = data[3] - offsets['bx2']
                by2 = data[4] - offsets['by2']
                bz2 = data[5] - offsets['bz2']
                print("N")
                bx, by, bz, bx2, by2, bz2 = data
                
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
                Fx, Fy, Fz = predict_force(
                    smoothed_bx, smoothed_by, smoothed_bz,
                    smoothed_bx2, smoothed_by2, smoothed_bz2
                )
                
                # Print results (customize as needed)
                print(f"\rFx: {Fx:.4f} N | Fy: {Fy:.4f} N | Fz: {Fz:.4f} N", end='', flush=True)
                
            time.sleep(0.01)  # Small delay to prevent CPU overload
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        ser.close()

if __name__ == "__main__":
    main()
