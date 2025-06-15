import cv2
import numpy as np
import util.dodecaBoard as dodecaBoard
from util.pose_estimation import pose_estimation
from scipy.spatial.transform import Rotation as R
import pyigtl

PLOTTING = False

# Replace the live stream source with the path to your saved video
video_path = "/home/george/MultiCameraTracking/Tracking/video/T1/T1_01/1.mp4"  # Update this to the path of your video
cap = cv2.VideoCapture(video_path)

server = pyigtl.OpenIGTLinkServer(port=18955)
poseEstimator = pose_estimation(framerate=60, plotting=PLOTTING)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
arucoParams = cv2.aruco.DetectorParameters()

# Update parameters to match the working example exactly
arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
arucoParams.cornerRefinementMaxIterations = 1000
arucoParams.cornerRefinementMinAccuracy = 0.001
arucoParams.adaptiveThreshWinSizeStep = 2
arucoParams.adaptiveThreshWinSizeMax = 15
arucoParams.adaptiveThreshConstant = 8

# Create the ArUco detector
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, arucoParams)

# Camera calibration parameters
camera_matrix = np.array([[1.56842921e+03, 0, 2.89275503e+02],
                           [0, 1.57214434e+03, 2.21092150e+02],
                           [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([2.28769970e-02, -4.54632281e+00, -3.04424079e-03, -2.06207084e-03, 9.30400565e+01], dtype=np.float32)

if not cap.isOpened():
    print("Cannot open video")
    exit()

while True:
    ret, frame = cap.read()  # Read a frame from the video
    if not ret:
        print("End of video or cannot read frame. Exiting ...")
        break

    # Detect ArUco markers
    corners, ids, rejected = aruco_detector.detectMarkers(frame)

    # Draw markers and overlay them on the frame
    overlayImg = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    # Display the frame with overlay
    cv2.imshow('frame', overlayImg)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
server.stop()

if PLOTTING:
    poseEstimator.plot()
    avgPos = np.average(poseEstimator.total_distance, axis=0)
    print(f'Avg X: {avgPos[0]}, Y: {avgPos[1]}, Z: {avgPos[2]}')
