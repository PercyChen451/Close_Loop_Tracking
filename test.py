import cv2
import numpy as np
import time
from Arduino import ArduinoComm, ArduinoConnect

class RobotCalibration:
    def __init__(self):
        # Initialize Arduino connection
        self.connect = ArduinoConnect('/dev/ttyACM0', 115200)  # Update port as needed
        self.arduino = ArduinoComm(self.connect)
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Calibration parameters
        self.calibration_positions = [
            [200, 200, 200],  # Target position
            [0, 0, 0]         # Home position
        ]
        self.current_position = [0, 0, 0]
        self.tracked_positions = []
        
    def detect_red_marker(self, frame):
        """Detect red marker using HSV color space"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define red color ranges
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Clean up the mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
            
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)
    
    def move_robot(self, target_position):
        """Send movement command to Arduino"""
        print(f"Moving to: {target_position}")
        
        # Convert position to pressure commands (adjust based on your robot's requirements)
        pressures = [
            target_position[0] / 100.0,  # Scale down for pressure command
            target_position[1] / 100.0,
            target_position[2] / 100.0
        ]
        
        # Command format: [enable, mode, direction, etc.]
        commands = [1, 1, 1, 1]  # Adjust based on your Arduino command protocol
        
        # Send command
        self.arduino.send_data(pressures, commands)
        self.current_position = target_position
        
        # Wait for movement to complete (adjust time as needed)
        time.sleep(3)
    
    def track_position(self, duration=5):
        """Track the red marker position for specified duration"""
        print(f"Tracking for {duration} seconds...")
        start_time = time.time()
        positions = []
        
        while time.time() - start_time < duration:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            marker_pos = self.detect_red_marker(frame)
            if marker_pos:
                positions.append(marker_pos)
                cv2.circle(frame, marker_pos, 10, (0, 255, 0), 2)
                cv2.putText(frame, f"Position: {marker_pos}", (20, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyWindow("Tracking")
        return positions
    
    def run_calibration_sequence(self):
        """Execute the full calibration sequence"""
        try:
            # 1. Move to first calibration position
            self.move_robot(self.calibration_positions[0])
            
            # 2. Track position at first point
            pos1 = self.track_position()
            if pos1:
                avg_pos1 = np.mean(pos1, axis=0)
                print(f"Average position at {self.calibration_positions[0]}: {avg_pos1}")
                self.tracked_positions.append(avg_pos1)
            
            # 3. Move to home position
            self.move_robot(self.calibration_positions[1])
            
            # 4. Track position at home
            pos2 = self.track_position()
            if pos2:
                avg_pos2 = np.mean(pos2, axis=0)
                print(f"Average position at {self.calibration_positions[1]}: {avg_pos2}")
                self.tracked_positions.append(avg_pos2)
            
            # Calculate scale factor if we have two points
            if len(self.tracked_positions) == 2:
                pixel_dist = np.linalg.norm(self.tracked_positions[0] - self.tracked_positions[1])
                real_dist = np.linalg.norm(np.array(self.calibration_positions[0]) - np.array(self.calibration_positions[1]))
                scale = real_dist / pixel_dist
                print(f"Calculated scale factor: {scale:.4f} mm/pixel")
            
        except KeyboardInterrupt:
            print("Calibration interrupted")
        finally:
            # Return to home position
            self.move_robot([0, 0, 0])
            self.cap.release()
            cv2.destroyAllWindows()
            self.connect.closePort()

if __name__ == "__main__":
    calibrator = RobotCalibration()
    print("Starting calibration sequence...")
    calibrator.run_calibration_sequence()
    print("Calibration complete!")