import numpy as np
import time
from Collocated_Dynamics_0713 import CollocatedDynamics
from Cam_tracker_class import Cam_Tracker
from Arduino import ArduinoComm, ArduinoConnect
from ForceSensor import SensorComm, SensorConnect

def main():
    np.set_printoptions(precision=3, suppress=True)
    
    # Initialize hardware with error handling
    try:
        # 1. First initialize sensor with longer timeout
        print("Initializing force sensor...")
        sen_connect = SensorConnect('/dev/ttyUSB0', 115200)
        sensor = SensorComm(sen_connect)
        
        # Wait for sensor to initialize and clear buffers
        time.sleep(2)
        while sen_connect.in_waiting > 0:
            sen_connect.readline()
        
        # Perform calibration
        print("Calibrating force sensor...")
        sensor.baseForce = sensor.calibrate_sensor()
        
        # 2. Then initialize Arduino
        print("Initializing Arduino...")
        connect = ArduinoConnect('/dev/ttyACM0', 250000)
        arduino = ArduinoComm(connect)
        time.sleep(1)  # Allow Arduino to initialize
        
        # 3. Initialize camera tracker
        print("Initializing camera tracker...")
        tracker = Cam_Tracker()
        
        # 4. Initialize dynamics model
        dyn = CollocatedDynamics()
        
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    # Setup trajectory
    path_coords = dyn.generate_circle(15, 30, 5)
    # path_coords = np.array([[0], [-10], [15]])

    # Initialize state variables
    robot_data = np.array([0, 0, 0])
    t_tip = np.array([0, 0, 5]) / 1000  # Convert to meters
    rigid_pred = np.zeros((3, 4))
    last_print_time = time.time()

    try:
        while True:
            current_time = time.time()
            
            # 1. Get sensor data with error handling
            try:
                force = sensor.receive_data()
                if time.time() - last_print_time > 1.0:  # Throttle printing
                    print(f"Force: {force}")
                    last_print_time = time.time()
            except Exception as e:
                print(f"Force sensor error: {e}")
                force = np.array([0, 0, 0])  # Default to zero force on error
            
            # 2. Get robot data
            try:
                robot_data = arduino.receive_data()
            except Exception as e:
                print(f"Arduino comm error: {e}")
                continue
            
            # 3. Update tracker
            try:
                tracker.update(rigid_pred)
                t_tip, R_tip = tracker.get_pose()
            except Exception as e:
                print(f"Tracker error: {e}")
                continue
            
            # 4. Run control loop
            try:
                (u_volumes, commands, backbone_pred, 
                 rigid_pred, tip_pred, dt) = dyn.CLIK(
                    robot_data, t_tip*1000, R_tip, force, path_coords
                )
                
                # 5. Send commands
                arduino.send_data(u_volumes, commands)
                
            except Exception as e:
                print(f"Control loop error: {e}")
                # Send stop command if control fails
                arduino.send_data(np.array([0, 0, 0]), [0, 0, 0, 0])
                continue
            
            # Optional: Add small delay to prevent CPU overload
            time.sleep(0.005)
            
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    finally:
        # Cleanup
        arduino.send_data(np.array([0, 0, 0]), [0, 0, 0, 0])  # Stop robot
        sen_connect.close()
        connect.close()
        print("Resources released")

if __name__ == "__main__":
    main()



Calibration error: could not convert string to float: 'Initializing MLX90394 sensors...'
Calibrating force sensor...
Starting calibration...
Calibration error: The following operation failed in the TorchScript interpreter.
Traceback of TorchScript, serialized code (most recent call last):
  File "code/__torch__.py", line 18, in forward
    relu1 = self.relu1
    fc1 = self.fc1
    _0 = (relu1).forward((fc1).forward(x, ), )
                          ~~~~~~~~~~~~ <--- HERE
    _1 = (relu2).forward((fc2).forward(_0, ), )
    return (fc3).forward(_1, )
  File "code/__torch__/torch/nn/modules/linear.py", line 12, in forward
    bias = self.bias
    weight = self.weight
    return torch.linear(x, weight, bias)
           ~~~~~~~~~~~~ <--- HERE

Traceback of TorchScript, original code (most recent call last):
/home/cardio/Documents/opencv/venv/lib/python3.12/site-packages/torch/nn/modules/linear.py(125): forward
/home/cardio/Documents/opencv/venv/lib/python3.12/site-packages/torch/nn/modules/module.py(1741): _slow_forward
/home/cardio/Documents/opencv/venv/lib/python3.12/site-packages/torch/nn/modules/module.py(1762): _call_impl
/home/cardio/Documents/opencv/venv/lib/python3.12/site-packages/torch/nn/modules/module.py(1751): _wrapped_call_impl
/home/cardio/Documents/Force_sensor/Force_Sensor_Cali/NNtraining.py(100): forward
/home/cardio/Documents/opencv/venv/lib/python3.12/site-packages/torch/nn/modules/module.py(1741): _slow_forward
/home/cardio/Documents/opencv/venv/lib/python3.12/site-packages/torch/nn/modules/module.py(1762): _call_impl
/home/cardio/Documents/opencv/venv/lib/python3.12/site-packages/torch/nn/modules/module.py(1751): _wrapped_call_impl
/home/cardio/Documents/opencv/venv/lib/python3.12/site-packages/torch/jit/_trace.py(1279): trace_module
/home/cardio/Documents/opencv/venv/lib/python3.12/site-packages/torch/jit/_trace.py(696): _trace_impl
/home/cardio/Documents/opencv/venv/lib/python3.12/site-packages/torch/jit/_trace.py(1002): trace
/home/cardio/Documents/Force_sensor/Force_Sensor_Cali/NNtraining.py(222): <module>
RuntimeError: mat1 and mat2 must have the same dtype, but got Double and Float
