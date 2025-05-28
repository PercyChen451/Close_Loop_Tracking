import cv2
import numpy as np
import time
from Rigid_Kinematics import RPPR_Kinematics
from Arduino import ArduinoComm
from Arduino import ArduinoConnect
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


i_goal = 0
tip_prev = np.zeros(2)
goal_prev = np.zeros(2)
isManualMode = True
keys_pressed = set()
clicked_points = []

def click_event(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"Selected point: ({x}, {y})")

def choose_video_source():
    print("Choose input source:")
    print("1. Webcam")
    print("2. Video file")
    choice = input("Enter 1 or 2: ").strip()
    if choice == "2":
        path = input("Enter path to video file: ").strip()
        cap = cv2.VideoCapture(path)
    else:
        cap = cv2.VideoCapture(0)# choose webcam here
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        exit()
    return cap

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

def detect_red_marker(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #lower_red = np.array([153, 140, 0])
    #upper_red = np.array([179, 255, 255])


    # lower_red2 = np.array([160, 100, 100])
    # upper_red2 = np.array([179, 255, 255])

    # mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    # mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    # mask = cv2.bitwise_or(mask1, mask2)

    # upper end hsv
    lower_red = np.array([95, 81, 0])
    upper_red = np.array([179, 255, 255])

    # lower_red = np.array([0, 142, 168])
    # upper_red = np.array([44, 255, 255])

    mask = cv2.inRange(hsv, lower_red, upper_red)

    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.imshow("Mask Debug", mask)

    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)

def transform_to_global(pt, origin, x_axis_vec, scale):
    vec = np.array(pt) - np.array(origin)
    x_unit = x_axis_vec / np.linalg.norm(x_axis_vec)
    y_unit = -1 * np.array([-x_unit[1], x_unit[0]])  # perpendicular in 2D
    x_mm = np.dot(vec, x_unit) * scale
    y_mm = np.dot(vec, y_unit) * scale
    angle = np.arccos(np.clip(np.dot(x_axis_vec, x_unit), -1.0, 1.0))
    
    # 2D rotation matrix for this angle
    rotation_matrix = np.array([
        [x_unit[0], y_unit[0]],  # x components
        [x_unit[1], y_unit[1]]   # y components
    ])
    return x_mm, y_mm, rotation_matrix

def render_matplotlib_overlay(origin, tip_pos, image, pcc_coords, rigid_coords, scale, rotation_matrix):
    height, width = image.shape[:2]
    fig = Figure(figsize=(width/100, height/100), dpi=100, facecolor='none')
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    
    if tip_pos is not None and origin is not None:
        try:
            # Plot single thick line from origin to current position

            # print(pcc_coords[0,:], pcc_coords[1,:])
            rigid_coords_x1, rigid_coords_y1= rigid_coords[0,:] / scale + origin[0], -1*rigid_coords[1,:]/ scale + origin[1]
            # rigid_coords_1 = [rigid_coords[0,:], -1*rigid_coords[1,:]]
            # print(rigid_coords_1)

            rigid_coords_result = rigid_coords.T @ rotation_matrix
            #print(rigid_coords_result)
            rigid_coords_result = rigid_coords_result.T
            rigid_coords_x, rigid_coords_y = rigid_coords_result[0] / scale + origin[0], rigid_coords_result[1]/ scale + origin[1]

            pcc_coords_2d = np.vstack((pcc_coords[0,:],pcc_coords[2,:]))
            pcc_coords_result = pcc_coords_2d.T @ rotation_matrix
            pcc_coords_result = pcc_coords_result.T

            pcc_coords_x, pcc_coords_y = pcc_coords_result [0,:] / scale + origin[0], pcc_coords_result [1,:]/ scale + origin[1]
            pcc_coords_x1, pcc_coords_y1 = pcc_coords[0,:] / scale + origin[0], -1*pcc_coords[2,:]/ scale + origin[1]
            print(pcc_coords_x, pcc_coords_y)
            #rigid_mat = np.matrix([rigid_coords_x], [rigid_coords_y])
            #rigid_result = np.matmul(rigid_mat,rotation_matrix)
            # rigid_result = rigid_mat@rotation_matrix
            # print(rigid_result, rigid_mat)q
            # #ax.plot(rigid_result[0],rigid_result[1], "-", color="tab:blue", linewidth=6, alpha = 1)

            # print (pcc_coords)
            
            # rot correctly
            ax.plot(rigid_coords_x,  rigid_coords_y, "-", color="tab:blue", linewidth=6, alpha = 1)
            ax.plot(pcc_coords_x, pcc_coords_y, "-", color="tab:red", linewidth=6, alpha = 1)

            # original
            # ax.plot(rigid_coords_x1, rigid_coords_y1, "-", color="tab:green", linewidth=6, alpha = 1)
            # ax.plot(pcc_coords_x1, pcc_coords_y1, "-", color="tab:orange", linewidth=6, alpha = 1)
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

def update_control_loop(tipCoords, num_pts, kin, circleCoords, arduino, dt):
    global i_goal
    global tip_prev, goal_prev
    global isManualMode, keys_pressed

    base_height = np.array([8,8])
    len_e = 3

    #--------------get arduino data----------
    arduino_data = arduino.receive_data()
    pressures = np.maximum(np.array(arduino_data)[0:3] / 1000, -12/1000)   # get pressures and convert to N/mm^2 for old Jacobian, keep within sensor ranges
    volumes = np.maximum( np.array(arduino_data[3:6])/ 1000, 0)          # get vol from arduino in  uL, convert to mL for model, keep within real volumes
    V1, V2, V3 = volumes[0], volumes[1] + 0.00001, volumes[2] + 0.0000015
    P1, P2, P3 = pressures[0], pressures[1] + 0.0000001, pressures[2] + 0.000000015
    l1, l2, l3 = kin.volume_to_length(V1, V2, V3)
    u_ells = np.array([l1, l2]) + base_height
    # print(volumes, u_ells, np.array([l1, l2]) , base_height)
    '''at  = 0, SBA should always be at base height, regardless of external force'''
    K = np.diag([(l1+l2)/2 / 30, (l1+l2)/2 / 30, (l1+l2)/2 / 30, 15, 15, 15])
    # compute q_0, config variables under  no external load)
    q0 =  kin.q_no_load(u_ells)
    
    #Force = np.array([0.0,-.5])
    Force = np.array([0.0,0])
    #Force = np.array([0.0,-.4])
    # Force = external_force(time_now, [0, -.1], .5)
    omega = np.array([0,0,0])

    # compute q (config variables under external load)
    links_now, beta_now = kin.compute_q_dynamics(K, q0[0:3], q0[3:], Force, omega, len_e)
    # print("links:", links_now)
    # print("betas:", beta_now)

    revoluteVecDamp = kin.get_revolute_backbone(beta_now, np.maximum(links_now, 8/3), len_e)
    pcc_coords = kin.get_PCC_backbone(revoluteVecDamp[:,0:4], num_pts, True)

    predTipCoords = revoluteVecDamp[:,-1]
    goalCoords = circleCoords[:,i_goal]

    # goalCoords = np.array([-10 + 5*np.sin(time_now), 10+5*np.sin(time_now)])
    tip_vel = (tipCoords - tip_prev)/dt
    goal_vel = (goalCoords - goal_prev)/dt
    error = goalCoords - tipCoords

    tip_prev = tipCoords
    goal_prev = goalCoords

    # print("tip_vel:",tip_vel)
    # print("tipCoords:", tipCoords)
    # print("goal_coords:", goalCoords)
    
    # -----------prepare arduino inputs---------------
    delta_u = np.array([0,0])
    commands = [0,0,0,0,0]

    # if the robot has reached the first point along the circle move on to the next
    if np.linalg.norm(error) < 0.8:
        print("reached goal with final coords:", tipCoords)
        if i_goal < len(circleCoords[0,:])-1:
            i_goal += 1
        else:
            i_goal = 0

    if 'm' in keys_pressed: # if 'm' key is pressed, toggle manual mode on/off
        print("Manual Mode: ", isManualMode)
        commands = [1, -1, -1, -1, -1]
        # time.sleep(0.2)
    else:
        try:
            # Calculate Jacobian with regularization
            J = kin.get_jacobian_actuated(u_ells, links_now, beta_now, len_e)
            # Add small regularization to prevent singular matrices
            regularization = 1e-6 * np.eye(J.shape[1])
            Ja_inv = np.linalg.pinv(J.T @ J + regularization) @ J.T
            rate_base = 0.5
            rate_fb = rate_base if np.linalg.norm(error) > 2 else rate_base + rate_base*(2 - np.linalg.norm(error) )
            rate_ff = 0.0
            delta_u = np.real(rate_fb*Ja_inv.dot(error.T)) + np.real( rate_ff*Ja_inv.dot(goal_vel-tip_vel.T) )

        except np.linalg.LinAlgError:
            # If SVD fails
            print("SVD failed to converge - using previous command")
            delta_u = np.zeros(2)  # Or np.random.normal(0, 0.01, 2)

    u_target = np.clip( u_ells + delta_u, 0, 50)

    ''' This will need some work, going from u_ells = q_a -> Tau, 
        will need to consider external forces, material properties and underactuated vars'''
    new_vol = np.clip( kin.lengths_to_volumes(u_target, base_height), 0, 500 )
    
    arduino.send_data(np.array([new_vol[0], new_vol[1], new_vol[1]]), commands) # send new set pressure and commands to arduino

    # print(u_ells, delta_u, u_target, new_vol)
    #print(tipCoords, goalCoords, np.linalg.norm(error), volumes)
    return (pcc_coords, revoluteVecDamp)

def main():
    global tip_prev, goal_prev, keys_pressed
    # ----------------Arduino connection-------------
    connect = ArduinoConnect('/dev/ttyACM1', 250000)  # Change COM6 if needed
    arduino = ArduinoComm(connect)
    # Call and define CC kinematics class
    kin = RPPR_Kinematics()
    # Define backbone discretization
    num_pts = 20
    # pathCoords = np.array([[   0.01, -15,   -13,   -11,    -9,    -7,    -5,    -3,    -1,     1,     3,     5,     7,     9,    11,    13,    15],
    #                        [    10, 5.4772,    9.2736,   11.5758,   13.1909,   14.3527,   15.1658,   15.6844,   15.9374,   15.9374,   15.6844,   15.1658,    14.3527,   13.1909,   11.5758,    9.2736,    5.4772]])

    # pathCoords = np.array([[   0.01,   -13,   -11,    -9,    -7,    -5,    -3,    -1,     1,     3,     5,     7,     9,    11,    13],
    #                        [    25,    24.2736,   26.5758,   28.1909,   29.3527,   30.1658,   30.6844,   30.9374,   30.9374,   30.6844,   30.1658,    29.3527,   28.1909,   26.5758,    24.2736]])
    
    # pathCoords = np.array([[   0.01,   4, 8, 10, 12],
    #                        [    20,    20, 18, 16, 15]])
    pathCoords = np.array([[   0.01, -15,   -13,   -11,    -9,    -7,    -5,    -3,    -1,     1,     3,     5,     7,     9,    11,    13,    15, 13, 10, 8],
                           [    25, 20.4772,    24.2736,   26.5758,   28.1909,   29.3527,   30.1658,   30.6844,   30.9374,   30.9374,   30.6844,   30.1658,    29.3527,   28.1909,   26.5758,    24.2736,    20.4772, 20, 20, 20]])
    
    tip_prev = np.array([0,5]) # eventually use first frame of opencv for initial pose
    goal_prev = pathCoords[:,0]
    cap = choose_video_source()
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        exit()
    
    # Step 1: Calibration
    print("Select two points for scale calibration (real world distance)")
    cal_pts = select_calibration_points(frame, "Click 2 points for scale calibration")
    real_dist = float(input("Enter real-world distance in mm between these points: "))
    scale = calculate_scale(cal_pts[0], cal_pts[1], real_dist)

    # Step 2: Origin and X-axis direction
    print("Select origin point")
    origin = select_calibration_points(frame, "Click origin point", n=1)[0]
    print("Select a second point along the desired global X axis")
    x_axis_pt = select_calibration_points(frame, "Click point on X-axis", n=1)[0]
    x_axis_vec = np.array(x_axis_pt) - np.array(origin)
    # Tracking loop
    positions = []
    print("Tracking started. Press 'q' to quit.")

    time_start = time.time()
    time_prev = time_start - 0.1

    while cap.isOpened():
        time_now = time.time()
        dt = time_now - time_prev
        time_prev = time_now

        ret, frame = cap.read()
        if not ret:
            break

        red_pos = detect_red_marker(frame)
        if red_pos:
            x_mm, y_mm, rotation_matrix = transform_to_global(red_pos, origin, x_axis_vec, scale)
            positions.append((x_mm, y_mm))
            cv2.circle(frame, red_pos, 5, (0, 255, 255), -1)
            cv2.putText(frame, f"({x_mm:.2f}, {y_mm:.2f}) mm", (red_pos[0]+10, red_pos[1]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            keys_pressed = set()
            if cv2.waitKey(1) & 0xFF == ord('m'):
                keys_pressed.add('m')
                print("manual mode")
            # print(x_mm, y_mm)
            (pcc_coords, rigid_coords) = update_control_loop(np.array([x_mm, y_mm]), num_pts, kin, pathCoords, arduino, dt)

            overlay = render_matplotlib_overlay(origin, red_pos, frame, pcc_coords, rigid_coords, scale,rotation_matrix)
            if overlay is not None:
                frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

            print(f"{x_mm:.2f}, {y_mm:.2f}, {dt:.3f}")

        cv2.imshow("Tracking", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Output result
    # print("Tracked Positions (mm):")
    # for pos in positions:
    #     print(f"{pos[0]:.2f}, {pos[1]:.2f}")

if __name__ == "__main__":
    main()
