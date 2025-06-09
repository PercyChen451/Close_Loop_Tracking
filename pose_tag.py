import cv2
import numpy as np
import apriltag

# === CAMERA CALIBRATION ===
with np.load('calibration_data_xz_480.npz') as data:
    camera_matrix = data['K']
    dist_coeffs = data['D']

# === APRILTAG SETTINGS ===
tag_size = 0.005  # Tag side length in meters (e.g., 5mm)

# Initialize webcam
cap = cv2.VideoCapture(2)

# Set resolution (optional)
# width, height = 1280, 720
width, height = 640, 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, 30)
# Good opencv commands
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) # turn the autofocus off
# cap.set(cv2.CAP_PROP_FOCUS, focusvalue) 

# Initialize AprilTag detector
options = apriltag.DetectorOptions(families='tag16h5')
detector = apriltag.Detector(options)
TARGET_IDS = {0, 1, 2, 3}

print("[INFO] Starting AprilTag pose detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for AprilTag
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect tags
    detections = detector.detect(gray)

    for det in detections:
        if det.tag_id not in TARGET_IDS:
            continue # Skip irrelevant tags
    
        # Pose estimation
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]

        pose, e0, e1 = detector.detection_pose(det, (fx, fy, cx, cy), tag_size)

        # Decompose pose
        R, t = pose[:3, :3], pose[:3, 3]

        # Print tag ID and pose
        print(f"[Tag {det.tag_id}] Position (mm): {t*1000}")
        # print(f"[Tag {det.tag_id}] Rotation matrix:\n{R}")

        # Draw axes
        rvec, _ = cv2.Rodrigues(R)
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, t, tag_size / 2)

        # Draw the tag outline
        for i in range(4):
            pt1 = tuple(det.corners[i].astype(int))
            pt2 = tuple(det.corners[(i + 1) % 4].astype(int))
            cv2.line(frame, pt1, pt2, (255, 255, 0), 2)

        # Draw center
        center = tuple(det.center.astype(int))
        cv2.circle(frame, center, 5, (0, 0, 255), -1)

    # Show result
    cv2.imshow("AprilTag Pose Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
