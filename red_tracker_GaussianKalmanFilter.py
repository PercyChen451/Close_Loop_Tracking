import cv2
import numpy as np
from scipy.stats import multivariate_normal
#   e:/MBL/OpenCV/BigWeight.mp4
clicked_points = []
import numpy as np
from scipy.stats import multivariate_normal

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
        
        # Process noise matrix (Q)
        self.Q = process_noise if process_noise is not None else np.diag([0.1, 0.1, 1.0, 1.0]).astype(np.float32)
        
        # Measurement noise matrix (R)
        self.R = measurement_noise if measurement_noise is not None else np.eye(2, dtype=np.float32) * 5.0
        
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
        self.velocity_boost_factor = 0.3  # 30% velocity lookahead
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
"""
class ResponsiveKalmanFilter(KalmanFilter):
    def update(self, z):
        # Detect rapid movements
        if z is not None and hasattr(self, 'last_z'):
            movement = np.linalg.norm(z - self.last_z)
            if movement > 10:  # Pixels/frame threshold for "fast motion"
                self.Q[:2,:2] = np.eye(2) * 0.5  # Temporary high process noise
        self.last_z = z
        return super().update(z)
    
    def update_belief(self):
        #Update our Gaussian belief about the state
        self.belief = multivariate_normal(
            mean=self.state[:2],  # Only position components
            cov=self.covariance[:2, :2]  # Position covariance
        )
    
    def predict(self):
        #Predict next state using Gaussian propagation
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        self.update_belief()
    
    def update(self, measurement):
        if measurement is None:
            self.predict()
            return self.state[:2]
            
        measurement = np.array(measurement)
        
        # Prediction step
        self.predict()
        
        # Measurement residual
        y = measurement - self.H @ self.state
        S = self.H @ self.covariance @ self.H.T + self.R
        K = self.covariance @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        
        # Update state and covariance
        self.state = self.state + K @ y
        self.covariance = (np.eye(4) - K @ self.H) @ self.covariance
        
        self.update_belief()
        return self.state[:2]
    
    def probability(self, position):
        #Return the probability of a given position under current belief
        return self.belief.pdf(position)     
"""
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
        cap = cv2.VideoCapture(0)
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
    # lower_red1 = np.array([0, 62, 38])
    # upper_red1 = np.array([52, 255, 255])

    # lower_red2 = np.array([160, 100, 100])
    # upper_red2 = np.array([179, 255, 255])

    # mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    # mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    # mask = cv2.bitwise_or(mask1, mask2)

    # upper end hsv
    # lower_red = np.array([153, 140, 0])
    # upper_red = np.array([179, 255, 255])

    lower_red = np.array([95, 81, 0])
    upper_red = np.array([179, 255, 255])

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
    return x_mm, y_mm

def main():
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
    red_pos = detect_red_marker(frame)
    MOVEMENT_THRESHOLD = 30  # pixels/frame
    last_raw_pos = np.array(origin)
    initial_pos = red_pos
    kf = GaussianKalmanFilter(
        initial_pos=initial_pos,
        initial_uncertainty=200.0,
        process_noise=np.diag([0.5, 0.5, 2.0, 2.0]),
        measurement_noise=np.eye(2)*2.0
    )
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Get detection
        raw_pos_tuple = detect_red_marker(frame)
        raw_pos = np.array(raw_pos_tuple) if raw_pos_tuple is not None else None
        #kf = ResponsiveKalmanFilter(raw_pos)
        # Tracking logic
        if raw_pos is not None:
            # Calculate movement speed
            movement_speed = np.linalg.norm(raw_pos - last_raw_pos)
            # Get probability of current measurement
            prob = kf.probability(raw_pos) if hasattr(kf, 'belief') else 1.0
            #trust measurements more when:
            # 1. Moving fast, or 2. Measurement is very likely
            if movement_speed > MOVEMENT_THRESHOLD or prob > 0.02:
                filtered_pos = kf.update(raw_pos)
                # Add velocity boost
                filtered_pos = filtered_pos + 0.2 * kf.state[2:]  # 20% velocity
            else:
                filtered_pos = kf.update(raw_pos)
            
            last_raw_pos = raw_pos
            
            # Visualization
            display_pos = tuple(map(int, np.round(filtered_pos)))
            x_mm, y_mm = transform_to_global(filtered_pos, origin, x_axis_vec, scale)
            
            cv2.circle(frame, display_pos, 5, (0, 255, 0), -1)
            cv2.putText(frame, f"({x_mm:.1f}, {y_mm:.1f}) mm | P={prob:.3f}",
                (display_pos[0]+10, display_pos[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            #print (raw_pos, filtered_pos )
            print (raw_pos[0]-filtered_pos[0], raw_pos[1]-filtered_pos[1])
            # Show raw detection in red
            cv2.circle(frame, tuple(map(int, raw_pos)), 3, (0, 0, 255), -1)
        
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Output result
    print("Tracked Positions (mm):")
    for pos in positions:
        print(f"{pos[0]:.2f}, {pos[1]:.2f}")

if __name__ == "__main__":
    main()
