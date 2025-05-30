import cv2
import numpy as np
import serial
import time

# Global variables
clicked_points = []
ser = None

# Your existing functions would be placed here (click_event, choose_video_source, 
# select_calibration_points, calculate_scale, detect_red_marker)

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

def get_marker_positions(cap1, cap2):
    """Get marker positions from both cameras"""
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    if not ret1 or not ret2:
        print("Error reading frames")
        return None, None
    
    marker1 = detect_red_marker(frame1)
    marker2 = detect_red_marker(frame2, n=2)  # Using different HSV range for second camera
    
    return marker1, marker2

def calculate_calibration(origin1, moved1, origin2, moved2):
    """
    Calculate transformation between cameras using origin and moved points
    Returns homography matrix
    """
    # Create point arrays (origin and moved point for each camera)
    pts_cam1 = np.array([origin1, moved1], dtype=np.float32)
    pts_cam2 = np.array([origin2, moved2], dtype=np.float32)
    
    # Calculate homography (minimum 4 points needed, but we only have 2)
    # For proper calibration you'd want more points, but with just two points
    # we can calculate a simple affine transformation (translation + scaling)
    if len(pts_cam1) >= 2 and len(pts_cam2) >= 2:
        # Calculate translation vector
        t = pts_cam1[0] - pts_cam2[0]
        
        # Calculate scale (distance between points)
        dist1 = np.linalg.norm(pts_cam1[1] - pts_cam1[0])
        dist2 = np.linalg.norm(pts_cam2[1] - pts_cam2[0])
        scale = dist1 / dist2 if dist2 != 0 else 1.0
        
        print(f"Calculated translation: {t}, scale: {scale}")
        
        # Create simple affine transformation matrix
        # This is a simplified approach - proper calibration would need more points
        H = np.array([
            [scale, 0, t[0]],
            [0, scale, t[1]],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return H
    return None

def main():
    global ser
    
    # Initialize video sources
    cap1, cap2 = choose_video_source()
    
    # Initialize serial
    initialize_serial()
    
    # Step 1: Get origin position in both cameras
    print("=== STEP 1: Select origin position ===")
    input("Place the robot at origin position and press Enter...")
    
    origin1, origin2 = get_marker_positions(cap1, cap2)
    if origin1 is None or origin2 is None:
        print("Could not detect marker in one or both cameras")
        return
    
    print(f"Origin positions - Cam1: {origin1}, Cam2: {origin2}")
    
    # Step 2: Move robot up and get new position
    print("\n=== STEP 2: Moving robot up ===")
    input("Press Enter to move robot up...")
    write_to_serial(200, 200, 200)  # Send move command
    time.sleep(2)  # Wait for movement to complete
    
    moved1, moved2 = get_marker_positions(cap1, cap2)
    if moved1 is None or moved2 is None:
        print("Could not detect marker after movement")
        return
    
    print(f"Moved positions - Cam1: {moved1}, Cam2: {moved2}")
    
    # Step 3: Calculate calibration
    print("\n=== STEP 3: Calculating calibration ===")
    H = calculate_calibration(origin1, moved1, origin2, moved2)
    if H is None:
        print("Calibration failed")
        return
    
    print("Calibration matrix (Homography):")
    print(H)
    
    # Step 4: Verification
    print("\n=== STEP 4: Verification ===")
    while True:
        # Get current marker positions
        marker1, marker2 = get_marker_positions(cap1, cap2)
        if marker1 is None or marker2 is None:
            print("Could not detect markers")
            continue
        
        # Transform camera2 point to camera1 space
        point_cam2 = np.array([marker2[0], marker2[1], 1], dtype=np.float32)
        transformed = np.dot(H, point_cam2)
        transformed = (transformed / transformed[2])[:2]  # Normalize and take first two components
        transformed_point = (int(transformed[0]), int(transformed[1]))
        
        print(f"Camera1: {marker1}, Camera2: {marker2} -> Transformed: {transformed_point}")
        
        # Display frames
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if ret1 and ret2:
            cv2.circle(frame1, marker1, 10, (0, 255, 0), 2)
            cv2.circle(frame1, transformed_point, 10, (0, 0, 255), 2)
            
            cv2.circle(frame2, marker2, 10, (0, 255, 0), 2)
            
            cv2.imshow("Camera 1", frame1)
            cv2.imshow("Camera 2", frame2)
        
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
