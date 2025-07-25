import numpy as np
import time
import numpy as np
import time
from Collocated_Dynamics_0713 import CollocatedDynamics
from Cam_tracker_class import Cam_Tracker
from Arduino import ArduinoComm
from Arduino import ArduinoConnect
from ForceSensor import SensorComm
from ForceSensor import SensorConnect


def main():
    np.set_printoptions(precision=3, suppress=True)
    connect = ArduinoConnect('/dev/ttyACM0', 250000)  # Change COM6 if needed
    arduino = ArduinoComm(connect)

    sen_connect = SensorConnect('/dev/ttyUSB0', 115200)
    sensor = SensorComm(sen_connect)

    tracker = Cam_Tracker()


    

    dyn = CollocatedDynamics()  # Initialize kinematics model
    

    path_coords = dyn.generate_circle(15, 30, 5)
    # path_coords = np.array([[0],
    #                         [-10],
    #                         [15]])

    robot_data = np.array([0,0,0])
    t_tip = np.array([0,0,5])/1000
    rigid_pred =  np.zeros((3,4))

    while True:
        robot_data = arduino.receive_data()
        force = sensor.receive_data()
        print(force)
        tracker.update(rigid_pred)
        t_tip, R_tip = tracker.get_pose()
        

        u_volumes, commands, backbone_pred, rigid_pred, tip_pred, dt = dyn.CLIK(robot_data, t_tip*1000, R_tip, force, path_coords)
        

        arduino.send_data(u_volumes, commands)

        # robot_data = u_volumes
        # t_tip = tip_pred/1000


        # print(t_tip)
        #print(u_volumes, tip_pred, t_tip*1000)

        # TODO: record data




if __name__ == "__main__":
    main()
