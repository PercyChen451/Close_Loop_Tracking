import cv2
import numpy as np
import serial
import time

# Global variables
clicked_points = []
ser = None

# Your existing functions would be placed here (click_event, choose_video_source, select_calibration_points, calculate_scale, detect_red_marker)

def initialize_serial(port='COM3', baudrate=9600):
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

def calibrate_cameras(cap1, cap2):
    # Capture frames from both cameras
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    if not ret1 or not ret2:
        print("Error reading frames for calibration")
        return None, None
    
    # Get calibration points from camera 1
    points1 = select_calibration_points(frame1, "Select 2 points for calibration (Camera 1)", 2)
    if len(points1) < 2:
        print("Not enough points selected for Camera 1")
        return None, None
    
    # Get calibration points from camera 2
    points2 = select_calibration_points(frame2, "Select same 2 points for calibration (Camera 2)", 2)
    if len(points2) < 2:
        print("Not enough points selected for Camera 2")
        return None, None
    
    # Calculate transformation between cameras
    pts1 = np.array(points1, dtype=np.float32)
    pts2 = np.array(points2, dtype=np.float32)
    
    # Calculate homography matrix
    H, _ = cv2.findHomography(pts2, pts1)
    print("Homography matrix calculated:")
    print(H)
    
    return H, None  # Returning homography and None for scale (you can add scale calculation if needed)

def main():
    global ser
    
    # Initialize video sources
    cap1, cap2 = choose_video_source()
    
    # Initialize serial
    initialize_serial()
    
    # Calibrate cameras
    H, scale = calibrate_cameras(cap1, cap2)
    if H is None:
        print("Camera calibration failed")
        return
    
    # Main loop
    while True:
        # Read frames from both cameras
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            print("Error reading frames")
            break
        
        # Detect red marker in both cameras
        marker1 = detect_red_marker(frame1)
        marker2 = detect_red_marker(frame2, n=2)  # Using different HSV range for second camera
        
        # Draw markers if found
        if marker1:
            cv2.circle(frame1, marker1, 10, (0, 255, 0), 2)
            print(f"Camera 1 marker: {marker1}")
        
        if marker2:
            cv2.circle(frame2, marker2, 10, (0, 255, 0), 2)
            print(f"Camera 2 marker: {marker2}")
            
            # Transform marker2 coordinates to camera1's coordinate system
            if H is not None:
                # Convert to homogeneous coordinates
                point = np.array([[marker2[0], marker2[1], 1]], dtype=np.float32)
                transformed = np.dot(H, point.T).T
                transformed = (transformed / transformed[0, 2])  # Normalize
                transformed_point = (int(transformed[0, 0]), int(transformed[0, 1]))
                print(f"Transformed Camera 2 marker to Camera 1 space: {transformed_point}")
        
        # Display frames
        cv2.imshow("Camera 1", frame1)
        cv2.imshow("Camera 2", frame2)
        
        # Send coordinates to serial
        write_to_serial(200, 200, 200)
        
        # Exit on ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    # Cleanup
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
    if ser is not None:
        ser.close()

if __name__ == "__main__":
    main()
