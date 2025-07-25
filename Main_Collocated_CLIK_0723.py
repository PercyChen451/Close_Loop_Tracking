import numpy as np
import time
import csv
from datetime import datetime
from Collocated_Dynamics_0713 import CollocatedDynamics
from Cam_tracker_class import Cam_Tracker
from Arduino import ArduinoComm, ArduinoConnect
from ForceSensor import SensorComm, SensorConnect

def main():
    np.set_printoptions(precision=4, suppress=True)

    # Create CSV filename with current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"sensor_data_{timestamp}.csv"
    
    # Initialize CSV file and write header
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'timestamp', 'force_x', 'force_y', 'force_z',
            'robot_data_1', 'robot_data_2', 'robot_data_3',
            't_tip_x', 't_tip_y', 't_tip_z',
            'u_volumes_1', 'u_volumes_2', 'u_volumes_3',
            'commands_1', 'commands_2', 'commands_3', 'commands_4',
            'backbone_pred_1', 'backbone_pred_2', 'backbone_pred_3',
            'rigid_pred_1', 'rigid_pred_2', 'rigid_pred_3',
            'tip_pred_x', 'tip_pred_y', 'tip_pred_z',
            'dt'
        ])

    # Initialize hardware
    sen_con = SensorConnect('/dev/ttyUSB0', 115200)
    sensor = SensorComm(sen_con)
    time.sleep(.1)
    sensor.calibrate_sensor()
    
    connect = ArduinoConnect('/dev/ttyACM0', 250000)
    time.sleep(1)
    arduino = ArduinoComm(connect)
    
    tracker = Cam_Tracker()
    dyn = CollocatedDynamics()
    path_coords = dyn.generate_circle(15, 30, 5)

    # Initialize state variables
    robot_data = np.array([0, 0, 0])
    t_tip = np.array([0, 0, 5])/1000
    rigid_pred = np.zeros((3, 4))
    time_start = time.time()
    time_prev = time_start

    while True:
        try:
            # Get current timestamp
            current_time = time.time() - time_start
            
            # Read data
            robot_data = arduino.receive_data()
            force = sensor.receive_data()
            
            # Update tracker
            tracker.update(rigid_pred)
            t_tip, R_tip = tracker.get_pose()
            
            # Run control algorithm
            u_volumes, commands, backbone_pred, rigid_pred, tip_pred, dt = dyn.CLIK(
                robot_data, t_tip*1000, R_tip, force, path_coords
            )
            
            # Send commands
            arduino.send_data(u_volumes, commands)
            
            # Flatten arrays for CSV storage
            rigid_pred_flat = rigid_pred.flatten()[:3]  # Only store first 3 elements if needed
            backbone_pred_flat = backbone_pred.flatten()[:3] if backbone_pred is not None else [0, 0, 0]
            
            # Write data to CSV
            with open(csv_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    current_time,
                    force[0], force[1], force[2],
                    robot_data[0], robot_data[1], robot_data[2],
                    t_tip[0], t_tip[1], t_tip[2],
                    u_volumes[0], u_volumes[1], u_volumes[2],
                    commands[0], commands[1], commands[2], commands[3],
                    backbone_pred_flat[0], backbone_pred_flat[1], backbone_pred_flat[2],
                    rigid_pred_flat[0], rigid_pred_flat[1], rigid_pred_flat[2],
                    tip_pred[0], tip_pred[1], tip_pred[2],
                    dt
                ])
            
            # Print status
            print(f"Force: {force} | Tip pos: {t_tip*1000} | Time: {current_time:.2f}s")
            
            time.sleep(0.01)
            
        except KeyboardInterrupt:
            print("\nData collection stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    main()
