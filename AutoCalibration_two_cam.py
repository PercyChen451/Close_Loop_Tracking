import cv2
import numpy as np
import time
from Rigid_Kinematics import RPPR_Kinematics
from Arduino import ArduinoComm
from Arduino import ArduinoConnect
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.stats import multivariate_normal
import serial


class GaussianKalmanFilter:
    def __init__(self, initial_pos, initial_uncertainty=100.0, process_noise=None, measurement_noise=None):
        """
        Initialize Gaussian Kalman Filter
        Args:
            initial_pos: (x, y) starting position (tuple or list)
            initial_uncertainty: Initial covariance scaling factor
            process_noise: Optional 4x4 process noise matrix (Q)
            measurement_noise: Optional 2x2 measurement noise matrix (R)
        """
        # Convert initial position to numpy array
        initial_pos = np.array(initial_pos, dtype=np.float32)
        # State vector: [x, y, vx, vy]
        self.state = np.array([initial_pos[0], initial_pos[1], 0, 0], dtype=np.float32)
        # State covariance matrix
        self.covariance = np.eye(4, dtype=np.float32) * initial_uncertainty
        self.covariance = self.covariance + 1e-6 * np.eye(4)  # Prevent numerical issues
        # Process noise matrix (Q)      [10.0, 10.0, 50.0, 50.0]
        #self.Q = process_noise if process_noise is not None else np.diag([1.0, 1.0, 5.0, 5.0]).astype(np.float32)
        self.Q = process_noise if process_noise is not None else np.diag([10.0, 10.0, 50.0, 50.0]).astype(np.float32)
        # Measurement noise matrix (R)
        self.R = measurement_noise if measurement_noise is not None else np.eye(2, dtype=np.float32) * 10.0
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        # Measurement matrix (observing position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        # Initialize belief distribution
        self.belief = None
        self.update_belief()
        # For velocity boosting
        #self.velocity_boost_factor = 0.3  # 30% velocity lookahead
        self.velocity_boost_factor = 0.5
        # For adaptive measurement trust
        self.last_measurement = None
        self.last_prediction = None
    def update_belief(self):
        """Update Gaussian belief about current state"""
        self.belief = multivariate_normal(
            mean=self.state[:2],
            cov=self.covariance[:2, :2]
        )
    def predict(self):
        """Predict next state using state transition"""
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        self.last_prediction = self.state.copy()
        self.update_belief()
        return self.state[:2]
    def update(self, measurement):
        """
        Update state with new measurement using Bayesian update
            Filtered (x, y) position
        """
        if measurement is None:
            return self.predict()
        measurement = np.array(measurement, dtype=np.float32)
        self.last_measurement = measurement.copy()
        # Prediction step
        self.predict()
        # Measurement residual (innovation)
        y = measurement - self.H @ self.state
        S = self.H @ self.covariance @ self.H.T + self.R
        # Kalman gain
        K = self.covariance @ self.H.T @ np.linalg.inv(S)
        print(f"Kalman gain:\n{K}")  # Should NOT be near-zero!
        # State update
        self.state = self.state + K @ y
        self.covariance = (np.eye(4) - K @ self.H) @ self.covariance
        self.update_belief()
        return self.state[:2]
    def probability(self, position):
        """
        Get probability of given position under current belief
            position: (x, y) position to evaluate
            Probability density at given position
        """
        if self.belief is None:
            return 0.0
        return self.belief.pdf(position)
    def get_boosted_position(self):
        """Get position with velocity boost applied"""
        if len(self.state) < 4:
            return self.state[:2]
        return self.state[:2] + self.velocity_boost_factor * self.state[2:4]
    def adaptive_update(self, measurement, movement_threshold=30, prob_threshold=0.01):
        """
        Smart update that adapts to movement speed and measurement probability
        Args:
            measurement: (x, y) position or None
            movement_threshold: Speed (pixels/frame) for fast movement
            prob_threshold: Minimum probability for full trust
        Returns:
            Filtered position, possibly with velocity boost
        """
        if measurement is None:
            return self.predict()
        measurement = np.array(measurement, dtype=np.float32)
        # Calculate movement characteristics
        movement_speed = 0.0
        if self.last_measurement is not None:
            movement_speed = np.linalg.norm(measurement - self.last_measurement)
        # Get probability of current measurement
        prob = self.probability(measurement)
        # Adaptive logic
        if movement_speed > movement_threshold or prob > prob_threshold:
            # High confidence update - apply velocity boost
            self.update(measurement)
            return self.get_boosted_position()
        else:
            # Normal update
            return self.update(measurement)

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
    
    cap, cap2 = [], []
    if choice == "1":
        cap = cv2.VideoCapture(0) # choose webcam here
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FPS, 30)


        cap2 = cv2.VideoCapture(2) # choose webcam here
        cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap2.set(cv2.CAP_PROP_FPS, 30)

        
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
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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

def initialize_serial(port='/dev/ttyACM0', baudrate = 250000):
    global ser
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)  # Wait for connection to establish
        print(f"Connected to {port} at {baudrate} baud")
    except Exception as e:
        print(f"Error opening serial port: {e}")
        ser = None

def write_to_serial(x, y, z):
    if ser is not None:
        command = f"{x},{y},{z}\n"
        ser.write(command.encode())
        print(f"Sent to serial: {command.strip()}")
    else:
        print("Serial port not initialized. Cannot send command.")

def get_marker_positions(cap1, cap2):
    """Get marker positions from both cameras"""
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    if not ret1 or not ret2:
        print("Error reading frames")
        return None, None
    
    marker1 = detect_red_marker(frame1)
    marker2 = detect_red_marker(frame2, n=2)  # Using different HSV range for second camera
    
    # Display frames with markers
    if marker1:
        cv2.circle(frame1, marker1, 10, (0, 255, 0), 2)
    if marker2:
        cv2.circle(frame2, marker2, 10, (0, 255, 0), 2)
    
    cv2.imshow("Camera 1", frame1)
    cv2.imshow("Camera 2", frame2)
    cv2.waitKey(100)
    
    return marker1, marker2

def triangulate_3d_point(p1, p2, P1, P2):
    """Calculate 3D position using two camera views"""
    A = np.zeros((4, 4))
    A[0] = p1[0] * P1[2] - P1[0]
    A[1] = p1[1] * P1[2] - P1[1]
    A[2] = p2[0] * P2[2] - P2[0]
    A[3] = p2[1] * P2[2] - P2[1]
    
    _, _, V = np.linalg.svd(A)
    point_3d = V[-1, :3] / V[-1, 3]
    return point_3d

def calculate_camera_matrices(origin1, moved1, origin2, moved2):
    """
    Calculate projection matrices for both cameras and scaling factor
    Returns P1, P2, scale_factor
    """
    x = moved1[0] - origin1[0]
    y1 = moved1[1] - origin1[1]
    z = moved2[0] - origin2[0]
    y2 = moved2[1] - origin2[1]
    scale_factor = 0 
    #scale_factor = y1 / y2
    return x, y1, y2, z, scale_factor 
"""
    # Calculate fundamental matrix (simplified approach)
    pts1 = np.array([origin1, moved1], dtype=np.float32)
    pts2 = np.array([origin2, moved2], dtype=np.float32)
    
    # Calculate scaling factor (distance should be same in real world)
    dist1 = np.linalg.norm(pts1[1] - pts1[0])
    dist2 = np.linalg.norm(pts2[1] - pts2[0])
    scale_factor = dist1 / dist2 if dist2 != 0 else 1.0
    
    # Normalize points from camera 2
    pts2_scaled = pts2 * scale_factor
    
    # Estimate fundamental matrix
    F, _ = cv2.findFundamentalMat(pts1, pts2_scaled, cv2.FM_8POINT)
    
    # Camera 1 matrix (assuming identity for first camera)
    P1 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]])
    
    # Calculate camera 2 origin1,
    #e2 = np.linalg.svd(F)[2][-1]  
    e2 = np. array([1,1,1], dtype = np.float32)
    e2_skew = np.array([[0, -e2[2], e2[1]],
                        [e2[2], 0, -e2[0]],
                        [-e2[1], e2[0], 0]], dtype = np.float32)
    term1 = np.dot(e2_skew, F)
    term2 = np.outer(e2, np.array([1, 1, 1], dtype=np.float32))
    combined = term1 + term2
    P2 = np.hstack((combined, e2.reshape(3, 1)))
    # Scale camera 2 matrix to match real-world units
    P2[:, :3] = P2[:, :3] * scale_factor
    
    return P1, P2, scale_factor
"""
def main():
    global ser
    
    # Initialize video sources
    cap1, cap2 = choose_video_source()
    initialize_serial()
    write_to_serial(0, 0, 0)  # Send move command
    ret, frame = cap1.read()
    ret2, frame2 = cap2.read()
    red_pos_yz = detect_red_marker(frame, 3)
    red_pos_xz = detect_red_marker(frame2, 3)
    # Step 1: Get origin position in both cameras
    print("=== STEP 1: Select origin position ===")
    input("Place the robot at origin position and press Enter...")
    origin1 = red_pos_xz
    origin2 = red_pos_yz
    #origin1, origin2 = get_marker_positions(cap1, cap2)
    if origin1 is None or origin2 is None:
        print("Could not detect marker in one or both cameras")
        return
    
    print(f"Origin positions - Cam1: {origin1}, Cam2: {origin2}")
    
    # Step 2: Move robot up and get new position
    print("\nMoving robot up")
    input("Press Enter to move robot up...")
    write_to_serial(200, 200, 200)  # Send move command
    time.sleep(2)  # Wait for movement to complete
    red_pos_yz = detect_red_marker(frame, 3)
    red_pos_xz = detect_red_marker(frame2, 3)
    #moved1, moved2 = get_marker_positions(cap1, cap2)
    moved1, moved2  = red_pos_xz, red_pos_yz
    if moved1 is None or moved2 is None:
        print("Could not detect marker after movement")
        return
    if moved1 is None or moved2 is None:
        print("Could not detect marker after movement")
        return
    time.sleep(2)  # Send move command
    print(f"Moved positions - Cam1: {moved1}, Cam2: {moved2}")
    write_to_serial(0, 0, 0) 
    # Step 3: Calculate camera matrices and scaling
    print("\n=== STEP 3: Calculating camera matrices ===")
    print (origin1, moved1, origin2, moved2)
    x, y1, y2, z, scale_factor = calculate_camera_matrices(origin1, moved1, origin2, moved2)
    #P1, P2, scale_factor = calculate_camera_matrices(origin1, moved1, origin2, moved2)
    print (x, y1, y2, z, scale_factor )
    #print(f"Scale factor (cam2 to cam1): {scale_factor}")
    print("Camera 1 projection matrix:")
    #print(P1)
    print("Camera 2 projection matrix (scaled):")
    #print(P2)
    
    # Step 4: Continuous 3D position tracking
    print("\n=== STEP 4: 3D Position Tracking ===")
    print("Press ESC to exit...")
    
    while True:
        # Get current marker positions
        marker1, marker2 = get_marker_positions(cap1, cap2)
        if marker1 is None or marker2 is None:
            print("Could not detect markers")
            continue
        
        # Apply scale to camera2 points
        marker2_scaled = (int(marker2[0] * scale_factor), 
                          int(marker2[1] * scale_factor))
        
        # Triangulate 3D position
        #point_3d = triangulate_3d_point(marker1, marker2_scaled, P1, P2)
        
        #print(f"3D Position (mm): X={point_3d[0]:.1f}, Y={point_3d[1]:.1f}, Z={point_3d[2]:.1f}")
        
        if cv2.waitKey(100) & 0xFF == 27:
            break
    
    # Cleanup
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
    if ser is not None:
        ser.close()

if __name__ == "__main__":
    main()