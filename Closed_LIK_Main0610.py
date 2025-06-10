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
from kine_dyna import Kinematics


i_goal = 0
tip_prev = np.zeros(3)
goal_prev = np.zeros(3)
isManualMode = True
keys_pressed = set()
clicked_points = []

def click_event(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"Selected point: ({x}, {y})")

def choose_video_source(width = 1280, height= 720):
    print("Choose input source:")
    print("1. Webcam")
    print("2. Video file")
    choice = input("Enter 1 or 2: ").strip()

    
    # width, height = 1920, 1080

    cap, cap2 = [], []
    if choice == "1":
        cap = cv2.VideoCapture(0) # choose webcam here
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FPS, 30)


        cap2 = cv2.VideoCapture(2) # choose webcam here
        cap2.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap2.set(cv2.CAP_PROP_FPS, 30)  # print('v1: {}, v2: {}, v3: {}' .format(V1,V2,V3))

    # print('l1: {}, l2: {}, l3: {}' .format(l1,l2,l3))


    elif choice == "2":
        path = input("Enter path to video file: ").strip()
        cap = cv2.VideoCapture(path)

        path = input("Enter path to second video file: ").strip()
        cap2 = cv2.VideoCapture(path)


    if not cap.isOpened() or not cap2.isOpened():
        print("Error: Could not open video source.")
        exit()

    return cap, cap2

def select_calibration_points(frame, msg, n=2):
    global clicked_points
    clicked_points = []
    clone = frame.copy()
    cv2.putText(clone, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.imshow("Calibration", clone)
    cv2.setMouseCallback("Calibration", click_event)

    while len(clicked_points) < n:
        display = clone.copy()
        for pt in clicked_points:
            cv2.circle(display, pt, 5, (0, 255, 0), -1)
        cv2.imshow("Calibration", display)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyWindow("Calibration")
    return clicked_points

def calculate_scale(p1, p2, real_distance_mm):
    pixel_dist = np.linalg.norm(np.array(p1) - np.array(p2))
    return real_distance_mm / pixel_dist

def detect_red_marker(frame, n=1):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # print('v1: {}, v2: {}, v3: {}' .format(V1,V2,V3))

    # print('l1: {}, l2: {}, l3: {}' .format(l1,l2,l3))

    # hsv single range
    lower_red = np.array([153, 140, 0])
    upper_red = np.array([179, 255, 255])

    if n == 2:
        lower_red = np.array([0, 107, 143])
        upper_red = np.array([179, 255, 255])

    mask = cv2.inRange(hsv, lower_red, upper_red)

    if n==3:
        lower_red1 = np.array([0, 95, 95])
        upper_red1 = np.array([40, 255, 255])

        lower_red2 = np.array([155, 95, 95])
        upper_red2 = np.array([179, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)


    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.imshow("Mask Debug", mask)

    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)

def transform_to_global(pt, origin, x_axis_vec, scale, n=1):
    vec = np.array(pt) - np.array(origin)

    x_unit = x_axis_vec / np.linalg.norm(x_axis_vec)
    y_unit = np.array([x_unit[1], x_unit[0]])  # perpendicular in 2D


    if n == 2:
        x_unit = x_axis_vec / np.linalg.norm(x_axis_vec)
        y_unit = np.array([x_unit[1], -x_unit[0]])  # perpendicular in 2D


    x_mm = np.dot(vec, x_unit) * scale
    y_mm = np.dot(vec, y_unit) * scale

    # 2D rotation matrix for this angle

    rotation_matrix = np.array([
        [x_unit[0], y_unit[0]],  # x components
        [x_unit[1], y_unit[1]]   # y components
    ])

    return x_mm, y_mm, rotation_matrix
    
def format_4sf(number):
    #4 significant figures
    if isinstance(number, (list, np.ndarray)):
        return [float(f"{x:.4g}") for x in number]
    return float(f"{number:.4g}")
    
def render_matplotlib_overlay(origin,tip_pos, image, pcc_coords, rigid_coords, goal_coords, scale, rotation_matrix, width, height, plane):
    height, width = image.shape[:2]
    fig = Figure(figsize=(width/100, height/100), dpi=100, facecolor='none')
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)

    if tip_pos is not None and origin is not None:
        try:
            # Plot single thick line from origin to current position
            # print(origin, tip_pos)


            # print("pcc_y: {},  pcc_z: {}".format(pcc_coords[1,:], pcc_coords[2,:]))
            # print("Rigid_X: {}, Rigid_Y: {}, Rigid_Z: {}".format(rigid_coords[0,:], rigid_coords[1,:], rigid_coords[2,:]))


            rigid_coords_x, rigid_coords_y  =   -1*rigid_coords[1,:] / scale + origin[0],   -1*rigid_coords[2,:]/ scale + origin[1]
            pcc_coords_x, pcc_coords_y      =   -1*pcc_coords[1,:] / scale + origin[0],     -1*pcc_coords[2,:]/ scale + origin[1]
            goal_coords_x, goal_coords_y    =   -1*goal_coords[1,:] / scale + origin[0],    -1*goal_coords[2,:]/ scale + origin[1]



            if plane=='xz':
                rigid_coords_x, rigid_coords_y  =   1*rigid_coords[0,:] / scale + origin[0],    -1*rigid_coords[2,:]/ scale + origin[1]
                pcc_coords_x, pcc_coords_y      =   1*pcc_coords[0,:] / scale + origin[0],      -1*pcc_coords[2,:]/ scale + origin[1]
                goal_coords_x, goal_coords_y    =   1*goal_coords[0,:] / scale + origin[0],     -1*goal_coords[2,:]/ scale + origin[1]

            
            ax.plot(goal_coords_x, goal_coords_y, "-o", color="darkmagenta", linewidth=3, alpha = 1)
            ax.plot(rigid_coords_x,  rigid_coords_y, "-", color="tab:blue", linewidth=6, alpha = 1)
            ax.plot(pcc_coords_x, pcc_coords_y, "-", color="tab:orange", linewidth=6, alpha = 1)
            



            ax.plot([origin[0], tip_pos[0]], [origin[1], tip_pos[1]], '-o', markersize=10, alpha = 1)

        except (TypeError, IndexError):
            pass  # Skip plotting if invalid
    # Set correct limits based on image size
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Inverted Y axis
    ax.axis("off")
    fig.tight_layout(pad=0)
    canvas.draw()
    buf = canvas.buffer_rgba()
    overlay = np.asarray(buf)
    # Convert to BGR and resize to match input exactly
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGR)
    return cv2.resize(overlay_bgr, (width, height))


def remove_white_background_hsv(image, h_low=0, h_high=179, s_low=0, s_high=32, v_low=159, v_high=255):

    h_low=0
    h_high=179
    s_low=10
    s_high=255
    v_low=10
    v_high=255

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([h_low, s_low, v_low])
    upper_white = np.array([h_high, s_high, v_high])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    # mask_inv = cv2.bitwise_not(mask)
    masked_frame = cv2.bitwise_and(image, image, mask=mask)

    #result = cv2.add(fg, bg)
    #return result
    return masked_frame

def update_control_loop(tipCoords, kin, PathCoords, arduino, dt):
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
    global isManualMode, keys_pressed, t_elapsed
    global softPts, skelPts
    global q_ref_prev    # print('U_ells: {}' .format(u_ells))
    global q_prev
    # print('q_next: {}' .format(q_next))

    # print('softTip: {}'. format(softPts[:,-1]))
    # print('hardTip: {}'. format(skelPts[:,-1]))


    _, numGoals = PathCoords.shape

    base_height = np.array([8,8,8])
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
    K = np.diag([15,15,0.5,15,15,0.5,15,15,0.5])
    D = np.diag([5,5,5,5,5,5,5,5,5])

    _, numGoals = PathCoords.shape

    f_ext_fun = lambda t: [0, 0, 0]

    # compute q_0, config variables under no external load
    q_ref = kin.q_no_load(u_ells)

    q_ref_dot = (q_ref - q_ref_prev)/dt

    # compute next configuration using dynamics
    q_next = kin.q_dynamics_new(q_prev, q_ref, q_ref_dot, f_ext_fun, t_elapsed, dt, K, D)

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
        print("reached goal with final coords:", tipCoords)
        i_goal = ((i_goal+1) % numGoals)

    if 'm' in keys_pressed: # if 'm' key is pressed, toggle manual mode on/off
        print("Manual Mode: ", isManualMode)
        commands = [1, -1, -1, -1, -1]
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
            rate_fb = 0.5
            rate_ff = 5e-3

            delta_u = J_ac_inv @ (rate_fb * error + rate_ff * (-tip_vel))

        except np.linalg.LinAlgError:
            print("SVD failed to converge - using previous command")
            delta_u = np.zeros(3)

    # Calculate target lengths and volumes
    u_target = np.round(np.clip(u_ells + kin.gain*delta_u, 0, 50), 3)
    new_vol = np.clip(kin.lengths_to_volumes(u_target, base_height), 0, 500)

    # Update previous values
    tip_prev = tipCoords
    goal_prev = goalCoords
    q_ref_prev = q_ref
    q_prev = q_next

    # Send commands to arduino
    arduino.send_data(np.array([new_vol[0], new_vol[1], new_vol[2]]), commands)

    # Extract actuated and unactuated variables from q_next
    # Assuming q_next structure: [q_a (actuated), q_u (unactuated)]
    q_a_est = q_next[:3]  # First 3 elements are actuated variables
    q_u_est = q_next[3:]  # Remaining elements are unactuated variables

    return (
        softPts,        # body coordinates
        skelPts,        # Skeleton coordinates
        q_next,         # Current configuration(q_est)
        q_ref,    # no load config
        q_a_est,        # Estimated chamber lengths
        q_u_est,        # Estimated unactuated variable
        delta_u,        # Control input (du)
        volumes,        # Current chamber volumes
        pressures,      # Current chamber pressures
        u_target,       # Target chamber lengths
        new_vol         # New target volumes
    )


def main():
    global tip_prev, goal_prev, keys_pressed, t_elapsed
    global softPts, skelPts
    global q_ref_prev
    global q_prev

    # ----------------Arduino connection-------------
    connect = ArduinoConnect('/dev/ttyACM0', 250000)  # Change COM6 if needed
    arduino = ArduinoComm(connect)

    # Call and define CC kinematics class
    kin = Kinematics()

    base_height = np.array([8,8,8])

    arduino_data = arduino.receive_data()
    volumes = np.maximum(np.array(arduino_data[3:6])/1000, 0)              # get vol from arduino in uL, convert to mL for model, keep within real volumes
    V1, V2, V3 = volumes[0], volumes[1] + 0.00001, volumes[2] + 0.0000015
    l1, l2, l3 = kin.volume_to_length(V1, V2, V3)
    u_ells = np.array([l1, l2, l3]) + base_height


    q_ref_prev = kin.q_no_load(u_ells)

    q_prev     = q_ref_prev.copy()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"tracking_data_{timestamp}.csv"
    
    # Define CSV headers
    headers = [
        'time',
        'real_x', 'real_y', 'real_z',
        'est_tip_x', 'est_tip_y', 'est_tip_z',
        *[f'q_next_{i}' for i in range(3)],  # q_next has 3 elements?
        *[f'q_a_est_{i}' for i in range(3)],
        *[f'q_u_est_{i}' for i in range(3)],
        *[f'du_{i}' for i in range(3)],
        *[f'vol_{i}' for i in range(3)],
        *[f'press_{i}' for i in range(3)],
        *[f'target_vol_{i}' for i in range(3)],
        'notes'
    ]

    time_start = time.time()
    time_prev = time_start - 0.1

    # PathCoords = np.array([[   8.00,    7.3068,    5.3644,    2.5576,   -0.5576,   -3.3644,   -5.3068,   -6.0000,   -5.3068,   -3.3644,   -0.5576,    2.5576,    5.3644,    7.3068,    8.0000],
    #                        [   0.00,    3.0372,    5.4728,    6.8245,    6.8245,    5.4728,    3.0372,    0.0000,   -3.0372,   -5.4728,   -6.8245,   -6.8245,   -5.4728,   -3.0372,   -0.0000],
    #                        [10.0000,    9.8019,    9.2470,    8.4450,    7.5550,    6.7530,    6.1981,    6.0000,    6.1981,    6.7530,    7.5550,    8.4450,    9.2470,    9.8019,   10.0000]])

    # pathCoords = np.array([[   0,   0, 0, 0, 0],
    #                         [   0.01,   4, 8, 10, 12],
    #                        [    35,    24, 33, 22, 31]])
    

    pathCoords = np.array([[     1,   2.4,   3.1,  3.9,   4.4,    5,    5.9,    7.2,   7.9,     7.9,      7.2,     5.9,     5,    5.5,   3.9,     3.1,    2.4,  1],
                           [   0.01, -15,   -13,   -11,    -9,    -7,    -5,    -3,    -1,     1,     3,     5,     7,     9,    11,    13,    15, 8],
                           [    25, 20.4772, 24.2736,   26.5758,   28.1909,   29.3527,   30.1658,   30.6844,   30.9374,   30.9374,   30.6844,   30.1658,    29.3527,   28.1909,   26.5758,    24.2736,    20.4772, 20]])

    # pathCoords = np.array([[   -0.01,   -4, -8, -10, -12],
    #                         [   0,   0, 0, 0, 0],
    #                        [    25,    24, 23, 22, 21]])

    # pathCoords = np.array([[      0,   0,         0,  0,         0,    0,    0,    0,     0,     0,      0,     0,     0,    0,     0,     0,    0,  0],
    #                         [   0.01, -15,   -13,   -11,    -9,    -7,    -5,    -3,    -1,     1,     3,     5,     7,     9,    11,    13,    15, 8],
    #                        [    25, 20.4772,    24.2736,   26.5758,   28.1909,   29.3527,   30.1658,   30.6844,   30.9374,   30.9374,   30.6844,   30.1658,    29.3527,   28.1909,   26.5758,    24.2736,    20.4772, 20]])

    # pathCoords = np.array([[ 0], [10], [15]])

    tip_prev = np.array([0,0,8]) # eventually use first frame of opencv for initial pose
    goal_prev = pathCoords[:,0]

    # ---------------------Begin OpenCV calibrations-------------------
    # width, height = 1920, 1080
    width, height = 1280, 720
    cap, cap2 = choose_video_source(width, height)

    # --------------Read first frame, first camera-------------
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        exit()

    # Step 1: Calibration
    # print("Select two points for scale calibration (real world distance)")
    # cal_pts_yz = select_calibration_points(frame, "Click 2 points for scale calibration")
    # real_dist_yz = float(input("Enter real-world distance in mm between these points: "))
    # scale_yz = calculate_scale(cal_pts_yz[0], cal_pts_yz[1], real_dist_yz)

    # # Step 2: Origin and X-axis direction
    # print("Select origin point")
    # origin_yz = select_calibration_points(frame, "Click origin point", n=1)[0]
    # print("Select a second point along the desired global X axis")
    # y_axis_pt = select_calibration_points(frame, "Click point on X-axis", n=1)[0]
    # y_axis_vec = np.array(y_axis_pt) - np.array(origin_yz)

    # 1080p values
    # cal_pts_yz = [[312, 715],[603, 720]]
    # real_dist_yz = 19.9
    # scale_yz = calculate_scale(cal_pts_yz[0], cal_pts_yz[1], real_dist_yz)

    # origin_yz = [960, 848]
    # y_axis_pt = [693, 839]
    # y_axis_vec = np.array(y_axis_pt) - np.array(origin_yz)


    # 720p values
    cal_pts_yz = [[57,554],[319, 560]]
    real_dist_yz = 19.9
    scale_yz = calculate_scale(cal_pts_yz[0], cal_pts_yz[1], real_dist_yz)

    origin_yz = [675, 678] 
    y_axis_pt = [389, 672]
    y_axis_vec = np.array(y_axis_pt) - np.array(origin_yz)


    # -------------Read first frame: Second camera------------
    ret2, frame2 = cap2.read()
    if not ret2:
        print("Error: Could not read frame.")
        exit()

    # Step 1: Calibration
    # print("Select two points for scale calibration (real world distance)")
    # cal_pts_xz = select_calibration_points(frame2, "Click 2 points for scale calibration")
    # real_dist_xz = float(input("Enter real-world distance in mm between these points: "))
    # scale_xz = calculate_scale(cal_pts_xz[0], cal_pts_xz[1], real_dist_xz)

    # # # Step 2: Origin and X-axis direction
    # print("Select origin point")
    # origin_xz = select_calibration_points(frame2, "Click origin point", n=1)[0]
    # print("Select a second point along the desired global X axis")
    # x_axis_pt = select_calibration_points(frame2, "Click point on X-axis", n=1)[0]
    # x_axis_vec = np.array(x_axis_pt) - np.array(origin_xz)

    # 1080p values
    # cal_pts_xz = [[1111,745],[1386, 738]]
    # real_dist_xz = 19.9
    # scale_xz = calculate_scale(cal_pts_xz[0], cal_pts_xz[1], real_dist_xz)

    # origin_xz = [827, 851]
    # x_axis_pt = [1074, 852]
    # x_axis_vec = np.array(x_axis_pt) - np.array(origin_xz)


    # 720p values
    cal_pts_xz = [[941, 585],[1202, 574]]
    real_dist_xz = 19.9
    scale_xz = calculate_scale(cal_pts_xz[0], cal_pts_xz[1], real_dist_xz)

    origin_xz = [666, 691]
    x_axis_pt = [873, 689]
    x_axis_vec = np.array(x_axis_pt) - np.array(origin_xz)

    # Tracking loop
    positions = []
    print("Tracking started. Press 'q' to quit.")

    # f = float
    # open file
    # Pure CC, RPPR original, RPPR modern
    # Open loop and close loop
    # Vary weights at tip

    # Record the following
        # Estimated (tip) position from model
        # Real position from tracker (opencv)
        # backbone coordinates 
        # record q_est (q_a_est adn q_u_est)
        # du (volumes in each chamber)
        # Time (from python)
        # any other  notes like DNF
        # f.write(f"{time:.2f},\n")

    time_start = time.time()
    time_prev = time_start - 0.1
try:
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            
            while cap.isOpened():
                time_now = time.time()
                t_elapsed = format_4sf(time_now - time_start)
                dt = format_4sf(time_now - time_prev)
                time_prev = time_now

                ret, frame = cap.read()
                ret2, frame2 = cap2.read()
                if not ret or not ret2:
                    break

                red_pos_yz = detect_red_marker(frame, 3)
                red_pos_xz = detect_red_marker(frame2, 3)

                if red_pos_yz and red_pos_xz:
                    y_yz, z_yz, rotation_matrix_yz = transform_to_global(red_pos_yz, origin_yz, y_axis_vec, scale_yz, 1)
                    x_xz, z_xz, rotation_matrix_xz = transform_to_global(red_pos_xz, origin_xz, x_axis_vec, scale_xz, 2)

                    # Get control loop data
                    (pcc_coords, rigid_coords, q_next, q_ref, q_a_est, q_u_est, 
                     delta_u, volumes, pressures, u_target, new_vol) = update_control_loop(
                        np.array([x_xz, y_yz, z_yz]), kin, pathCoords, arduino, dt)

                    # Format all numerical data to 4 significant figures
                    formatted_data = [
                        t_elapsed,  # time
                        format_4sf(x_xz), format_4sf(y_yz), format_4sf(z_yz),  # real position
                        *[format_4sf(x) for x in pcc_coords[:,-1]],  # estimated tip
                        *[format_4sf(x) for x in q_next],  # q_est
                        *[format_4sf(x) for x in q_a_est],  # q_a_est
                        *[format_4sf(x) for x in q_u_est],  # q_u_est
                        *[format_4sf(x) for x in delta_u],  # du
                        *[format_4sf(x) for x in volumes],  # volumes
                        *[format_4sf(x) for x in pressures],  # pressures
                        *[format_4sf(x) for x in new_vol],  # target volumes
                        ''  # notes
                    ]

                    writer.writerow(formatted_data)

                    # Visualization (unchanged)
                    cv2.circle(frame, red_pos_yz, 5, (0, 255, 255), -1)
                    cv2.putText(frame, f"({y_yz:.4g}, {z_yz:.4g}) mm", 
                               (red_pos_yz[0]+10, red_pos_yz[1]),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
                    
                    cv2.circle(frame2, red_pos_xz, 5, (0, 255, 255), -1)
                    cv2.putText(frame2, f"({x_xz:.4g}, {z_xz:.4g}) mm", 
                               (red_pos_xz[0]+10, red_pos_xz[1]),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
                    

                    overlay = render_matplotlib_overlay(origin_yz, red_pos_yz, frame, pcc_coords, rigid_coords, pathCoords, scale_yz, rotation_matrix_yz, width, height,'yz')
                    overlay_clean = remove_white_background_hsv(overlay)
                    if overlay is not None:
                        frame = cv2.addWeighted(frame, 1, overlay_clean, 0.9, 0)


                    overlay2 = render_matplotlib_overlay(origin_xz, red_pos_xz, frame2, pcc_coords, rigid_coords, pathCoords, scale_xz, rotation_matrix_xz, width, height, 'xz')
                    overlay2_clean = remove_white_background_hsv(overlay2)
                    if overlay2 is not None:
                        frame2 = cv2.addWeighted(frame2, 1, overlay2_clean, 0.9, 0)



            # print(f"{x_mm:.2f}, {y_mm:.2f}, {dt:.3f}")

        
            cv2.imshow("Tracking y-z plane", frame)
            cv2.imshow("Tracking x-z plane", frame2)
                # print(f"{x_mm:.2f}, {y_mm:.2f}, {dt:.3f}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error during tracking: {e}")
    finally:
        # GUARANTEED CLEANUP - executes whether loop completes or errors occur
        csv_file.close()  # Ensures all data is flushed to disk
        cap.release()
        cv2.destroyAllWindows()
        print(f"Data successfully saved to {csv_filename}")

if __name__ == "__main__":
    main()
    # Output result
    # print("Tracked Positions (mm):")
    # for pos in positions:
    #     print(f"{pos[0]:.2f}, {pos[1]:.2f}")

