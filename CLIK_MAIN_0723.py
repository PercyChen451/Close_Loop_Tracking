import cv2
import numpy as np
import time
from Rigid_Kinematics import RPPR_Kinematics
from Arduino import ArduinoComm
from Arduino import ArduinoConnect
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import csv
from datetime import datetime
from Kinematics_Dynamics_0618 import Kinematics
import json
import torch
from collections import deque
import serial

# ========================== Force Sensor Settings ========================
norm_params = np.load('normalization_params.npy', allow_pickle=True).item()
X_mean = norm_params['mean'].astype(np.float32)
X_std = norm_params['std'].astype(np.float32)
Y_mean = norm_params.get('Y_mean', 0).astype(np.float32)
Y_std = norm_params.get('Y_std', 1).astype(np.float32)


model = torch.jit.load('force_calibration_model_optimized.pt')
model.eval()

# Data smoothing (optional)
SMOOTHING_WINDOW = 5
bx_buffer = deque(maxlen=SMOOTHING_WINDOW)
by_buffer = deque(maxlen=SMOOTHING_WINDOW)
bz_buffer = deque(maxlen=SMOOTHING_WINDOW)
bx2_buffer = deque(maxlen=SMOOTHING_WINDOW)
by2_buffer = deque(maxlen=SMOOTHING_WINDOW)
bz2_buffer = deque(maxlen=SMOOTHING_WINDOW)

# ========================== SBA and Camera Settings ========================
CAM_IDS = [0, 2, 4] # Adjust as needed
VID_PATH = ['output_cam0.avi','output_cam1.avi','output_cam2.avi',]
IMAGE_SIZE = (640, 480)
np.set_printoptions(precision=2, suppress=True)

i_goal = 0
tip_prev = np.zeros(3)
goal_prev = np.zeros(3)


# LOWER_COLOR = np.array([0, 120, 100]) # Red HSV lower
# UPPER_COLOR = np.array([10, 255, 255]) # Red HSV upper
# COLOR_RANGE = (LOWER_COLOR, UPPER_COLOR)

import numpy as np
import time
from collections import defaultdict

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

def load_params(filename):
    data = np.load(filename)
    return data['K'], data['D'], data['R'], data['T']

def find_marker_center(frame, n=2, min_pixels=50):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Color thresholding
    if n == 0:  # Yellow
        lower = np.array([13, 100, 80])
        upper = np.array([56, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
    elif n == 1:  # Blue
        lower = np.array([100, 90, 70])
        upper = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
    else:  # Red (n==2)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([4, 255, 255])
        lower_red2 = np.array([155, 50, 50])
        upper_red2 = np.array([179, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
    # Count matching pixels
    pixel_count = cv2.countNonZero(mask)

    # Debug visualization
    # cv2.imshow("mask debug", mask)
    # cv2.putText(frame, f"Pixels: {pixel_count}", (10, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # # Skip if too few pixels
    if pixel_count < min_pixels:
        # print(f"Warning: Only {pixel_count} pixels detected (minimum {min_pixels} required)")
        return None
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return np.array([cx, cy])
    return None



def backproject_pixel(K, R, T, pt):
    uv1 = np.array([pt[0], pt[1], 1.0])
    K_inv = np.linalg.inv(K)
    ray_cam = K_inv @ uv1
    d = R.T @ ray_cam
    d /= np.linalg.norm(d)
    origin = -R.T @ T
    return origin, d

def triangulate_rays(origins, directions):
    A, b = [], []
    for o, d in zip(origins, directions):
        I = np.eye(3)
        A.append(I - np.outer(d, d))
        b.append((I - np.outer(d, d)) @ o)
    A = np.sum(A, axis=0)
    b = np.sum(b, axis=0)
    return np.linalg.lstsq(A, b, rcond=None)[0]


def estimate_pose(marker_world, marker_ref):
    """
    Estimate rigid transformation from marker_ref to marker_world.
    Returns rotation matrix R and translation vector t.
    """
    assert marker_world.shape == marker_ref.shape == (3, 3)

    # Center the points
    centroid_world = marker_world.mean(axis=0)
    centroid_ref = marker_ref.mean(axis=0)

    P = marker_ref - centroid_ref
    Q = marker_world - centroid_world

    # Compute optimal rotation using SVD
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    R_opt = Vt.T @ U.T

    # Handle reflection case
    if np.linalg.det(R_opt) < 0:
        Vt[2, :] *= -1
        R_opt = Vt.T @ U.T

    t_opt = centroid_world - R_opt @ centroid_ref

    return R_opt, t_opt

def draw_pose(params, frames, R_tip_0, t_tip_0, T_b0, backbone):
    '''Draw pose on frames using camera parameters and rot/ trans'''

    R_b0 = T_b0[0:3,0:3]
    p_b0 = T_b0[0:3,3]
    for (K, D, R, T), frame in zip(params, frames):
        imgpts, _ = cv2.projectPoints(t_tip_0, R, T, K, D)
        imgpts_int = tuple(map(int, imgpts.ravel())) # flatten to 1D array, then convert to int tupple

        # Define length and axes in the tip frame
        axis_length = 0.005  # 2 cm
        origin = np.array([[0, 0, 0]], dtype=np.float32)
        axes = np.array([
            [axis_length, 0, 0],   # X axis
            [0, axis_length, 0],   # Y axis
            [0, 0, axis_length]    # Z axis
        ], dtype=np.float32)

        # Transform to global frame using T_tip_0 or T_tip_base
        # R_tip, t_tip = T_tip_base[:3, :3], T_tip_base[:3, 3]
        R_tip = R_tip_0 @ R_b0.T
        t_tip = t_tip_0 
        axes_world = (R_tip @ axes.T).T + t_tip  # shape (3, 3)
        points_to_project = np.vstack((t_tip, axes_world))  # (4, 3)

        # print(p_b0, np.shape(p_b0))
        backbone_world = (backbone.T/1000 - p_b0.T) @ R_b0
        # print(backbone_world*1000, np.shape(backbone_world))
        # print(points_to_project*1000, np.shape(points_to_project))
        # print(T_b0, np.shape(T_b0))
        # backbone = backbone.T/1000 + np.array(p_b0.T)
        

        # Project to image and draw
        imgpts, _ = cv2.projectPoints(points_to_project, R, T, K, D)
        imgpts = imgpts.reshape(-1, 2).astype(int)
        origin_2d = tuple(imgpts[0])

        # project backbone in meters
        img_backbone, _ = cv2.projectPoints(backbone_world, R, T, K, D) 
        img_backbone = img_backbone.reshape(-1, 2).astype(int)
        # print(imgpts, img_backbone)

        # print(img_backbone, np.shape(img_backbone))
        # print(imgpts, np.shape(imgpts))


        cv2.circle(frame, imgpts_int, 8, (0, 255, 255), -1)               # Tip position in yellow
        cv2.line(frame, origin_2d, tuple(imgpts[1]), (0, 0, 255), 2)  # X axis in red
        cv2.line(frame, origin_2d, tuple(imgpts[2]), (0, 255, 0), 2)  # Y axis in green
        cv2.line(frame, origin_2d, tuple(imgpts[3]), (255, 0, 0), 2)  # Z axis in blue
        cv2.polylines(frame, [img_backbone], False, (255, 255, 0), 2)   # Backbone in cyan
    return

def update_control_loop(tipCoords, R_tip, kin, PathCoords, base_height, arduino, sensor, dt, ser, offsets):
    """
    Update the control loop for the soft robot.
    
    Returns:
        tuple: Contains multiple values for data recording:
            - softPts (np.array): Soft body points coordinates
            - skelPts (np.array): Skeleton points coordinates
            - q_next (np.array): Current configuration variables (including actuated and unactuated)
            - q_ref (np.array): Configuration under no external load
            - q_a_est (np.array): Estimated actuated variables (chamber lengths)
            - q_u_est (np.array): Estimated unactuated variables
            - du (np.array): Delta u (control input)
            - volumes (np.array): Current chamber volumes
            - pressures (np.array): Current chamber pressures
            - u_target (np.array): Target chamber lengths
            - new_vol (np.array): New target volumes
    """
    global i_goal
    global tip_prev, goal_prev
    global t_elapsed
    global softPts, skelPts
    global q_ref_prev    # print('U_ells: {}' .format(u_ells))
    global q_prev, Ei_prev
    global integral
    # print('q_next: {}' .format(q_next))

    # print('softTip: {}'. format(softPts[:,-1]))
    # print('hardTip: {}'. format(skelPts[:,-1]))

    t = time.time()
    _, numGoals = PathCoords.shape
    # len_e = 3

    #--------------get arduino data----------
    arduino_data = arduino.receive_data()
    pressures = np.maximum(np.array(arduino_data)[0:3] / 1000, -12/1000)   # get pressures and convert to N/mm^2 for old Jacobian, keep within sensor ranges
    volumes = np.maximum(np.array(arduino_data[3:6])/1000, 0)              # get vol from arduino in uL, convert to mL for model, keep within real volumes
    V1, V2, V3 = volumes[0], volumes[1] + 0.00001, volumes[2] + 0.0000015
    P1, P2, P3 = pressures[0], pressures[1] + 0.0000001, pressures[2] + 0.000000015
    l1, l2, l3 = kin.volume_to_length(V1, V2, V3)
    u_ells = np.array([l1, l2, l3]) + base_height
    # print(volumes, u_ells, np.array([l1, l2]) , base_height)
    '''at  = 0, SBA should always be at base height, regardless of external force'''
    # K = np.diag([15,15,0.5,15,15,0.5,15,15,0.5])

    scale = np.sum(u_ells)/40
    K = np.diag([15*scale, 15*scale, 0.5*scale, 15*scale, 15*scale, 0.5*scale, 15*scale, 15*scale, 0.5*scale])/2.5/3

    D = np.diag([5,5,5,5,5,5,5,5,5])

    if integral != 0:
        I = np.diag([1.6,2.4,1,1.6,2.4,1,1.6,2.4,1])
    else:
        I = np.diag([0,0,0,0,0,0,0,0,0])


    # print(I)

    _, numGoals = PathCoords.shape

    #=============== get force ============
    # ================ In update_control_loop function ================
# Replace the entire force sensor section with this non-blocking version:

if sensor is not None and ser is not None and ser.in_waiting > 0:
    try:
        line = ser.readline().decode('ascii', errors='ignore').strip()
        data = parse_serial_line(line)
        if data:
            bx = data[0] - offsets['bx1']
            by = data[1] - offsets['by1']
            bz = data[2] - offsets['bz1']
            bx2 = data[3] - offsets['bx2']
            by2 = data[4] - offsets['by2']
            bz2 = data[5] - offsets['bz2']
            
            # Apply smoothing
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
            # Create force function
            f_ext_fun = lambda t: [Fx, Fy, Fz]
    except Exception as e:
        print(f"Force sensor error: {e}")
        f_ext_fun = lambda t: [0, 0, 0]
else:
    f_ext_fun = lambda t: [0, 0, 0]

    #=============== get q (config) ===============
    # compute q_0, config variables under no external load
    q_ref = kin.q_no_load(u_ells)

    q_ref_dot = (q_ref - q_ref_prev)/dt

    # compute next configuration using dynamics
    q_next = kin.q_dynamics_new(q_prev, q_ref, q_ref_dot, Ei_prev, f_ext_fun, t_elapsed, dt,K,D,I)

    # Compute soft curve and skeleton points
    softPts, skelPts, _ = kin.Compute_Soft_Curve(q_next)
    pred_tip = kin.Compute_actual_tip(q_next)
    skelPts = np.append(skelPts, pred_tip.reshape(-1,1), axis=1)

    goalCoords = PathCoords[:,i_goal]
    error = goalCoords - tipCoords
    tip_vel = (tipCoords - tip_prev)/dt
    # print('U_ells: {}' .format(u_ells))
    # print('q_next: {}' .format(q_next))

    # print('softTip: {}'. format(softPts[:,-1]))
    # print('hardTip: {}'. format(skelPts[:,-1]))

    # Initialize delta_u
    delta_u = np.array([0,0,0])
    commands = [0,0,0,0,0]

    # Check if robot reached current goal
    if np.linalg.norm(error) < 0.8:
        # print("reached goal with final coords:", tipCoords)
        i_goal = ((i_goal+1) % numGoals)

    else:
        try:
            # Calculate Jacobian with regularization
            J_ac = kin.Actuated_Jacobian(q_next)
            U, S, Vh = np.linalg.svd(J_ac, full_matrices=False)
            V = Vh.T
            sigma_0 = 0.01*max(S)
            nu = 50
            h = (S**3 + nu*S**2 + 2*S + 2*sigma_0)/(S**2 + nu*S + 2)

            H_inv = np.diag(1.0/h)
            J_ac_inv = V @ H_inv @ U.T

            # error_norm = np.linalg.norm(error)
            # rate_fb = 0.23 * (1.0  + max(0.0, 2.0 - error_norm))
            rate_fb = 0.4
            rate_ff = .001

            delta_u = J_ac_inv @ (rate_fb * error + rate_ff * (-tip_vel))

        except np.linalg.LinAlgError:
            print("SVD failed to converge - using previous command")
            delta_u = np.zeros(3)

    # Calculate target lengths and volumes
    u_target = np.round(np.clip(u_ells + kin.gain*delta_u, 0, 50), 3)
    new_vol = np.clip(kin.lengths_to_volumes(u_target, base_height), 0, 500)

    # print(tipCoords, new_vol)

    # Update previous values
    error_current = kin.q_diff(q_next, q_ref)*dt 
    Ei_prev += kin.q_diff(q_next, q_ref) * dt

    # print('Ei_prev: {}' .format(Ei_prev))
    # print('Error_current: {}' .format(error_current))
    tip_prev = tipCoords
    goal_prev = goalCoords
    q_ref_prev = q_ref
    q_prev = q_next
    
    if True:
        omega = 2*np.pi*0.04
        sentVolumes = 400 + 100 * np.array([np.sin(omega * t ), 
                                      np.sin(omega * t + 2/3* np.pi ),
                                      np.sin(omega * t + 4/3* np.pi )])

    # Send commands to arduino
    arduino.send_data(sentVolumes, commands)

    # Extract actuated and unactuated variables from q_next
    # Assuming q_next structure: [q_a (actuated), q_u (unactuated)]
    q_a_est = q_next[:3]  # First 3 elements are actuated variables
    q_u_est = q_next[3:]  # Remaining elements are unactuated variables

    return (
        softPts,        # body coordinates
        skelPts,        # Skeleton coordinates
        q_next,         # Current configuration(q_est)
        q_ref,          # no load config
        q_a_est,        # Estimated chamber lengths
        q_u_est,        # Estimated unactuated variable
        delta_u,        # Control input (du)
        volumes,        # Current chamber volumes
        pressures,      # Current chamber pressures
        u_target,       # Target chamber lengths
        # new_vol,         # New target volumes
        pred_tip
    )




#=============== Force Sensor Functions ===============
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
    global tip_prev, goal_prev, keys_pressed, t_elapsed
    global softPts, skelPts
    global q_ref_prev
    global q_prev, Ei_prev
    global integral
    global i_goal
    global ser

    # ----------------Arduino connection----------------------------------
    connect = ArduinoConnect('/dev/ttyACM0', 250000)  # Change COM6 if needed
    arduino = ArduinoComm(connect)
    #SERIAL_PORT = '/dev/ttyUSB0'
    #BAUD_RATE = 115200
    #TIMEOUT = 1
    #ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
    #print(f"Connected to {SERIAL_PORT}, waiting for data...")
    # sensor_connect = ArduinoConnect('/dev/ttyUSB0', 115200)  # Change COM6 if needed
    # sensor = ArduinoComm(sensor_connect)
    sensor = None
    offsets = None
    ser = None
    if sensor is not None:
        print("Calibrating force sensor...")
        offsets = calibrate_force_sensor()
        if offsets is not None:
            ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.1)  # Non-blocking
    # --------------Call and define CC kinematics class---------------
    kin = Kinematics()
    kin2  = RPPR_Kinematics()

    base_height = np.array([5,5,5])

    integral = 0  #Switch for integral

    arduino_data = arduino.receive_data()
    volumes = np.maximum(np.array(arduino_data[3:6])/1000, 0)              # get vol from arduino in uL, convert to mL for model, keep within real volumes
    V1, V2, V3 = volumes[0], volumes[1] + 0.00001, volumes[2] + 0.0000015
    l1, l2, l3 = kin.volume_to_length(V1, V2, V3)
    u_ells = np.array([l1, l2, l3]) + base_height


    q_ref_prev = kin.q_no_load(u_ells)
    q_prev     = q_ref_prev.copy()
    Ei_prev    = np.zeros(3*kin.n)

    pathCoords = kin2.generate_circle(10, 18, 100)
    pathCoords = kin2.rotate_axis('y', pathCoords, -10)
    pathCoords = kin2.rotate_axis('x', pathCoords, 10)

    # pathCoords = np.array([[-8], [-8], [28]])
    # pathCoords = np.array([[0.1], [0.1], [8]])

    # pathCoords = np.array([[0.1], [-8], [30]])

    # pathCoords = np.array([[0.1, 0.1, 0.1, 0.1],
    #                        [0.1, 0.1, 0.1, 0.1],
    #                        [30, 32, 34, 36]])


    f_pathcoord = open('pathcoords.csv', 'w')
    if f_pathcoord.tell() == 0:
        f_pathcoord.write(f"pathcoords\n")
        _,num_pts = pathCoords.shape
    for i in range(num_pts):
        f_pathcoord.write(f"{[pathCoords[0,i].item(), pathCoords[1,i].item(), pathCoords[2,i].item()]}\n")
    f_pathcoord.close()


    tip_prev = np.array([0,0,8]) # eventually use first frame of opencv for initial pose
    goal_prev = pathCoords[:,0]


    # ================ Marker Kinematics===============
    # Start Kinematics
    base_mark_height = 0.010 # meters
    # sba base frame wrt to global, eventually repplace with three marker reference
    T_b0 = np.eye(4)
    R_b0 = np.array([[0, 0, 1],
                    [-1, 0, 0],
                    [0, -1, 0]])
    p_b0 = -np.array([0.1623366, -0.0030681 - 0.00063, -0.005048 - base_mark_height])
    T_b0[0:3,0:3] = R_b0
    T_b0[0:3,3] = p_b0

    # marker_ref_b = np.array([
    #     [ 0.0,   -6.729,    -.5],  # red
    #     [-5.827, -3.265,    -.5],  # yellow
    #     [-5.827, 3.265,     -.5],  # blue
    #     ]) / 1000 *1.05
    
    # mass_h = 6.75 
    mass_h = 0
    marker_ref_b = np.array([
        [ 0.0,   -6.729,    4 + mass_h],  # red
        [-5.827, -3.265,    4 + mass_h],  # yellow
        [-5.827, 3.265,     4 + mass_h],  # blue
        ]) / 1000 *1.05

    marker_ref_0 = marker_ref_b @ R_b0

    print(marker_ref_b)
    print(marker_ref_0)

    # ===================== Load camera params ========================
    params = [load_params(f'cam{i+1}.npz') for i in range(3)]

    # ======================= Setup cameras =========================
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # caps = [cv2.VideoCapture(cam_id) for cam_id in VID_PATH]
    caps = [cv2.VideoCapture(cam_id) for cam_id in CAM_IDS]
    outs = []
    for i, cap in enumerate(caps):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_SIZE[1])
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        outs.append( cv2.VideoWriter( 'output_cam'+str(i)+'.avi', fourcc, 30, IMAGE_SIZE) ) # FIX by adding toggle for recording

    # ========================= Begin time and file recording =========================
    time_start = time.time()
    time_prev = time_start - 0.1
    file  =  open('tracking_data_rigid.csv', 'w')

    while True:
        time_now = time.time()
        t_elapsed = time_now - time_start
        dt = time_now - time_prev
        time_prev = time_now
        
        frames = [cap.read()[1] for cap in caps]
        if any(f is None for f in frames):
            print("Frame capture failed")
            break
        
        pts_r = [find_marker_center(f, 2) for f in frames]
        pts_y = [find_marker_center(f, 0) for f in frames]
        pts_b = [find_marker_center(f, 1) for f in frames]
        pts = [pts_r, pts_y, pts_b]

        for i, f in enumerate(frames):
            outs[i].write(f)

        i_views = [[],[],[]]
        for j, pts_c in enumerate(pts):
            for i, pt in enumerate(pts_c):
                if pt is not None:
                    cv2.circle(frames[i], tuple(pt.astype(int)), 5, (255, 255, 0), -1)
                    i_views[j].append(i)

        # i_views = [0, 1]
        pts3d_b = np.zeros((3,3))
        pts3d = np.zeros((3,3))
        for j, pts_c in enumerate(pts):
            if len(i_views[j]) >= 2: # triangualte pt if seen by at least two cams
                params_seen, pts_seen = [params[i] for i in i_views[j] ], [pts_c[i] for i in i_views[j]]
                origins, directions = [], []
                for (K, D, R, T), pt in zip(params_seen, pts_seen):
                    o, d = backproject_pixel(K, R, T, pt)
                    origins.append(o)
                    directions.append(d)

                pt3d = triangulate_rays(origins, directions).ravel() 
                pts3d[j,:] = pt3d

                # the following transforms the color 3d pts to the base frame
                pt3d_b = T_b0 @ np.append(pt3d, 1) # transform to base frame
                pts3d_b[j,:] = pt3d_b[0:3] # update all colored pts
                
                # x_sen, y_sen, z_sen, scale = pt3d_b*1000 # convert to mm

                # cv2.putText(frames[i_views[j][-1]], f"({x_sen:.2f}, {y_sen:.2f}, {z_sen:.2f}) mm", (pts_c[i_views[j][-1]][0]+10, pts_c[i_views[j][-1]][1]),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # print(f"3D Position: x={x_sen:.3f}, y={y_sen:.3f}, z={z_sen:.3f}")

        T_tip_0 = np.eye(4)  
        if pts3d is not None: #  extract tip pose for 3 makers in camera frame
            R_tip_0, t_tip_0 = estimate_pose(pts3d, marker_ref_0)
            T_tip_0[0:3,0:3] = R_tip_0
            T_tip_0[0:3,3] = t_tip_0

            T_tip_base =  T_b0 @ T_tip_0

            R_tip_b, t_tip_b = estimate_pose(pts3d_b, marker_ref_b)
            # print(R_tip_b)
            # print(T_tip_base[0:3,0:3])

            # print('Tip position (base):', t_tip_b*1000)
            # print('Tip position (global):', T_tip_base[0:3,3]*1000)
            t_tip_mm = t_tip_b*1000

            sensor = 1
            # Get current position and other data
            (pcc_coords, rigid_coords, q_next, q_ref, q_a_est, q_u_est, delta_u, vols, pressures, u_target, pred_tip) = update_control_loop(
             t_tip_mm, R_tip_b, kin, pathCoords, base_height, arduino, sensor, dt, ser, offsets)

            draw_pose(params, frames, R_tip_0, t_tip_0, T_b0, rigid_coords)

            # Record data to CSV
            if file.tell() == 0:  # Write header if file is empty
                file.write(f"time;est_x;est_y;est_z;sen_x;sen_y;sen_z;path_idx;V1;V2;V3\n")
            file.write(f"{t_elapsed:.2f};{pred_tip[0]:.2f};{pred_tip[1]:.2f};{pred_tip[2]:.2f};{t_tip_mm[0]:.2f};{t_tip_mm[1]:.2f};{t_tip_mm[2]:.2f};{i_goal};{vols[0]:.4f};{vols[1]:.4f};{vols[2]:.4f}\n")

                

        for i, f in enumerate(frames):
            cv2.imshow(f"Camera {i+1}", f)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for i, cap in enumerate(caps):
        cap.release()
        outs[i].release()
    cv2.destroyAllWindows()
    file.close()


if __name__ == "__main__":
    main()
