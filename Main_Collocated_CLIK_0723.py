import numpy as np
import time
from Collocated_Dynamics_0713 import CollocatedDynamics
from Cam_tracker_class import Cam_Tracker
from Arduino import ArduinoComm, ArduinoConnect
from ForceSensor import SensorComm, SensorConnect

def main():
    np.set_printoptions(precision=4, suppress=True)

    
    sen_con = SensorConnect('/dev/ttyUSB0', 115200)
    sensor = SensorComm(sen_con)
    time.sleep(.1)
    sensor.calibrate_sensor()
    
    # Initialize other components
    print("Initializing Arduino...")
    connect = ArduinoConnect('/dev/ttyACM0', 250000)
    time.sleep(1)
    arduino = ArduinoComm(connect)
    
    tracker = Cam_Tracker()


    

    dyn = CollocatedDynamics()  # Initialize kinematics model
    

    path_coords = dyn.generate_circle(15, 30, 5)
    # path_coords = np.array([[0],
    #                         [-10],
    #                         [15]])

    robot_data = np.array([0,0,0])
    t_tip = np.array([0,0,5])/1000
    rigid_pred =  np.zeros((3,4))

    time_start = time.time()
    time_prev = time_start
    while True:
        robot_data = arduino.receive_data()
        force = sensor.receive_data()
        print(force)
        # force = np.array([0,0,0])
        tracker.update(rigid_pred)
        t_tip, R_tip = tracker.get_pose()
        

        u_volumes, commands, backbone_pred, rigid_pred, tip_pred, dt = dyn.CLIK(robot_data, t_tip*1000, R_tip, force, path_coords)
        

        arduino.send_data(u_volumes, commands)

        # print(t_tip)
        #print(u_volumes, tip_pred, t_tip*1000)

        # TODO: record data



if __name__ == "__main__":
    main()