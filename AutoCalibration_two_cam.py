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

def calculate_camera_matrices(origin1, moved1, origin2, moved2):
    """
    Calculate the transformation between cameras using known movement
    Returns:
    - scale_factor: scaling from cam2 pixels to cam1 pixels
    - translation: translation vector between camera coordinates
    - rotation: rotation angle between camera views (simplified)
    """
    # Calculate movement vectors in each camera
    vec1 = np.array(moved1) - np.array(origin1)
    vec2 = np.array(moved2) - np.array(origin2)
    
    # Calculate scaling factor (assuming vertical movement)
    scale_factor = vec1[1] / vec2[1] if vec2[1] != 0 else 1.0
    
    # Calculate rotation angle (simplified 2D rotation)
    angle = np.arctan2(vec2[1], vec2[0]) - np.arctan2(vec1[1], vec1[0])
    
    # Calculate translation (origin points should match after scaling and rotation)
    rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]])
    translated_origin2 = np.dot(rot_matrix, np.array(origin2) * scale_factor
    translation = np.array(origin1) - translated_origin2
    
    return scale_factor, translation, angle

def triangulate_3d_position(point1, point2, scale_factor, translation, angle):
    #Calculate 3D position from two camera views
    #point1: (x,y) from camera 1 (YZ plane)
    #point2: (x,y) from camera 2 (XZ plane)
    #Returns: (X, Y, Z) in mm
    # Apply scaling and rotation to camera2 point
    rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]])
    point2_transformed = np.dot(rot_matrix, np.array(point2)) * scale_factor + translation
    
    # Simple triangulation:
    # Camera 1 sees (Y,Z)
    # Camera 2 sees (X,Z)
    X = point2_transformed[0]
    Y = point1[0]
    
    # Average Z from both cameras (after transformation)
    Z_cam1 = point1[1]
    Z_cam2 = point2_transformed[1]
    Z = (Z_cam1 + Z_cam2) / 2
    
    return X, Y, Z

def main():
    global ser
    
    # Initialize video sources and serial
    cap1, cap2 = choose_video_source()
    initialize_serial()
    
    # Move to origin position
    write_to_serial(0, 0, 0)
    time.sleep(2)
    
    # Get origin positions
    print("=== STEP 1: Get origin position ===")
    origin1 = detect_red_marker(cap1.read()[1], 3)  # Camera 1 sees YZ plane
    origin2 = detect_red_marker(cap2.read()[1], 3)  # Camera 2 sees XZ plane
    
    if origin1 is None or origin2 is None:
        print("Could not detect markers at origin")
        return
    
    print(f"Origin positions - Cam1 (YZ): {origin1}, Cam2 (XZ): {origin2}")
    
    # Move up by 200mm
    print("\n=== STEP 2: Moving up 200mm ===")
    write_to_serial(200, 200, 200)
    time.sleep(2)
    
    # Get moved positions
    moved1 = detect_red_marker(cap1.read()[1], 3)
    moved2 = detect_red_marker(cap2.read()[1], 3)
    
    if moved1 is None or moved2 is None:
        print("Could not detect markers after movement")
        return
    
    print(f"Moved positions - Cam1: {moved1}, Cam2: {moved2}")
    
    # Calculate calibration parameters
    print("\n=== STEP 3: Calculate calibration ===")
    scale_factor, translation, angle = calculate_camera_matrices(
        origin1, moved1, origin2, moved2
    )
    
    print(f"Scale factor (cam2 to cam1): {scale_factor}")
    print(f"Translation: {translation}")
    print(f"Rotation angle (rad): {angle}")
    
    # Continuous 3D tracking
    print("\n=== STEP 4: 3D Tracking ===")
    print("Press ESC to exit...")
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            continue
            
        point1 = detect_red_marker(frame1, 3)
        point2 = detect_red_marker(frame2, 3)
        
        if point1 and point2:
            # Calculate 3D position
            X, Y, Z = triangulate_3d_position(point1, point2, scale_factor, translation, angle)
            
            # Display results
            print(f"3D Position: X={X:.1f}mm, Y={Y:.1f}mm, Z={Z:.1f}mm")
            
            # Draw markers
            cv2.circle(frame1, point1, 10, (0,255,0), 2)
            cv2.circle(frame2, point2, 10, (0,255,0), 2)
            
        cv2.imshow("Camera 1 (YZ)", frame1)
        cv2.imshow("Camera 2 (XZ)", frame2)
        
        if cv2.waitKey(100) & 0xFF == 27:
            break
    
    # Cleanup
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
    if ser:
        ser.close()

if __name__ == "__main__":
    main()
