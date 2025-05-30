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
    
    # Calculate camera 2 matrix from fundamental matrix
    e2 = np.linalg.svd(F)[2][-1]  # Epipole
    e2_skew = np.array([[0, -e2[2], e2[1]],
                        [e2[2], 0, -e2[0]],
                        [-e2[1], e2[0], 0]])
    P2 = np.hstack((e2_skew.dot(F) + np.outer(e2, [1, 1, 1]), e2.reshape(3, 1)))
    
    # Scale camera 2 matrix to match real-world units
    P2[:, :3] = P2[:, :3] * scale_factor
    
    return P1, P2, scale_factor

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
    
    # Step 3: Calculate camera matrices and scaling
    print("\n=== STEP 3: Calculating camera matrices ===")
    P1, P2, scale_factor = calculate_camera_matrices(origin1, moved1, origin2, moved2)
    
    print(f"Scale factor (cam2 to cam1): {scale_factor}")
    print("Camera 1 projection matrix:")
    print(P1)
    print("Camera 2 projection matrix (scaled):")
    print(P2)
    
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
        point_3d = triangulate_3d_point(
            marker1, marker2_scaled, P1, P2
        )
        
        print(f"3D Position (mm): X={point_3d[0]:.1f}, Y={point_3d[1]:.1f}, Z={point_3d[2]:.1f}")
        
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
