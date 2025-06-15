import util.dodecaBoard as dodecaBoard
from util.pose_estimation import pose_estimation
from util.kalman import KalmanFilterCV
from scipy.spatial.transform import Rotation as R
import cv2
import numpy as np
import multiprocessing as mp
from multiprocessing import Manager
import time
import pyigtl
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import itertools
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import contextlib
import json

### PRESS Q ON ANY CAMERA WINDOW TO QUIT ###

TIME_RANGE = (0, 999) # Video-specific time range for data collection, in seconds
RATE = 10
OFFSET = True
MARKER_MAPPER = True
PRERECORDED = True
STEREO_REPROJECT = False
STEREO_TRIANGULATE_ALL = True
KALMAN = False
ADAPTIVE = False
PLOTTING = False
PLOTTING_3D = False
CROP = True
CROP_PADDING = 100
IGT = False
IGT_Port = 18955

toolOffset = (0.59176578, -1.00066888, -241.33763016) # T33_minpos_int_percam
calibFile = '../Calibration/calib_files/camera/T33_minpos_int_percam_cam1fixed.json'
calibFile_target = '../Calibration/calib_files/tool/T33_minpos_int_percam_cam1fixed_target.txt'
calibFile_ref = '../Calibration/calib_files/tool/T33_minpos_int_percam_cam1fixed_ref.txt'
mapperFile_target = '../Calibration/calib_files/marker_mapper/target_map_T33_3.yml'
mapperFile_ref = '../Calibration/calib_files/marker_mapper/ref_map_T33_3.yml'
RFile_mono = '../Calibration/calib_files/filter/R_30x30_mono.txt'
QFile_stereo = '../Calibration/calib_files/filter/Q_matrix_stereo.txt'
video_folder= '../Registration/Landmark_Registration_Trials/GT01_init' # Directory that contains a list of of folders with recordings of each point of interest
videoExt = 'mp4' # extension of provided video files
frame_height, frame_width = 1088, 1456

top_config = [  
    [1, 2, 3, 4, 5],
    [1, 2, 4, 5],
    [1, 4, 5],
    [1, 5],
    [1]
]

left_config = [list(range(1, n + 1)) for n in range(5, 0, -1)]
right_config = list(reversed([list(range(5, 5 - n, -1)) for n in range(1, 6)]))

camera_configs_dict = {
    "Top": top_config,
    "Left": left_config,
    "Right": right_config
}

fusion_options = {
    "STEREO": [False, True],
    # "KALMAN": [False, True],
    # "ADAPTIVE": [False, True]
}

fusion_combinations = [
    dict(zip(fusion_options, values))
    for values in itertools.product(*fusion_options.values())
]

# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5) # Works with marker mapper
poseEstimator = pose_estimation(framerate=RATE, plotting=False, aruco_dict=aruco_dict)

arucoParams = cv2.aruco.DetectorParameters()
arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG  # Default CORNER_REFINE_NONE
arucoParams.cornerRefinementMaxIterations = 1000  # Default 30
arucoParams.cornerRefinementMinAccuracy = 0.001  # Default 0.1
arucoParams.adaptiveThreshWinSizeStep = 2  # Default 10
arucoParams.adaptiveThreshWinSizeMax = 15  # Default 23
arucoParams.adaptiveThreshConstant = 8  # Default 7
# arucoParams.polygonalApproxAccuracyRate = 0.01  # Default 0.03

detector = cv2.aruco.ArucoDetector(aruco_dict, arucoParams)

# m = 33.2/2 # half of marker length (currently in mm)

# # Single marker board
# board_points = np.array([[[-m, m, 0],[m, m, 0],[m, -m, 0],[-m, -m, 0]]],dtype=np.float32)

# ref_board = cv2.aruco.Board(board_points, aruco_dict, np.array([0]))
# ref_board = cv2.aruco.Board(board_points, aruco_dict, np.array([1]))

# Dodecahedron board
target_marker_size = 24  # dodecahedron edge length in mm
target_pentagon_size = 27.5
ref_marker_size = 33  # dodecahedron edge length in mm
ref_pentagon_size = 40

# Function to suppress noise profile prints coming from Adaptive Kalman
@contextlib.contextmanager
def suppress_print():
    # Redirect standard output to devnull to suppress prints
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        sys.stdout = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# Function to get fusion config description
def fusion_desc(settings):
    if not settings["STEREO"]:
        description = "Mono"
    else:
        description = "Stereo"
    
    return description


# Combined function to read and parse the marker map YAML file
def load_markerMapper(file_path, offset=(0.0, 0.0, 0.0)):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Skip lines starting with '%', in this case the directive
    yaml_content = ''.join(line for line in lines if not line.strip().startswith('%'))
    data = yaml.safe_load(yaml_content)
    
    markers = data.get('aruco_bc_markers', [])
    
    if not markers:
        raise ValueError("No markers found in the YAML file.")
    
    # Extract the top marker's corners
    top_marker = markers[0]
    corners_top_marker = np.array(top_marker['corners'], dtype=np.float32)
    center_top_marker = np.mean(corners_top_marker, axis=0)
    
    # Calculate the normal vector of the top marker plane
    v1 = corners_top_marker[1] - corners_top_marker[0]
    v2 = corners_top_marker[2] - corners_top_marker[0]
    normal_top_marker = np.cross(v1, v2)
    normal_top_marker /= np.linalg.norm(normal_top_marker)
    normal_top_marker *= -1 # Invert normal so that is faces outwards from board center

    # Create a rotation matrix to align the normal with the z-axis
    z_axis = np.array([0, 0, 1], dtype=np.float32)
    axis = np.cross(normal_top_marker, z_axis)
    angle = np.arccos(np.clip(np.dot(normal_top_marker, z_axis), -1.0, 1.0))

    # Check if the normal is already aligned with the z-axis
    if np.linalg.norm(axis) != 0:
        axis /= np.linalg.norm(axis)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ], dtype=np.float32)
        
        Rot = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    else:
        Rot = np.eye(3)

    parsed_markers = []
    for marker in markers:
        if 'corners' in marker:
            corners = np.array(marker['corners'], dtype=np.float32)
            # Translate and rotate corners
            translated_corners = corners - center_top_marker
            rotated_corners = np.dot(translated_corners, Rot.T)  # Apply rotation
            rotated_corners = rotated_corners[[2, 3, 0, 1], :] * 1000 # Flip marker orientation and go from m to mm
            offset_corners = rotated_corners - np.array(offset, dtype=np.float32) # Apply offset
            parsed_markers.append(offset_corners)

    # Convert parsed_markers to a numpy array of the same type
    return np.array(parsed_markers, dtype=np.float32)

def terminate_processes(processes):
    for process in processes:
        process.terminate()
    # for process in processes:
    #     process.join()

def runCam(cam, cameraMatrix, distCoeffs, video_path):
    local_corners_dict = {}
    local_ids_dict = {}

    cap = cv2.VideoCapture(f'{video_path}/{cam}.{videoExt}')
    if not cap.isOpened():
        print(f"Cannot open camera {cam}.")
        return local_corners_dict, local_ids_dict

    frame_idx = 0
    if CROP: found = False
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= 30:  # Stop if no more frames or after collecting 30 frames
            break
         
        frame = cv2.undistort(frame, cameraMatrix, distCoeffs)
        
        if CROP and found:
            ids = []
            corners = []
            frame_target = frame[ymin_target:ymax_target, xmin_target:xmax_target]
            frame_ref = frame[ymin_ref:ymax_ref, xmin_ref:xmax_ref]
            
            corners_target, ids_target, _ = detector.detectMarkers(frame_target)
            if ids_target is not None:
                corners_target = np.array(corners_target) + (xmin_target, ymin_target)
                corners.append(corners_target)
                ids.append(ids_target)
                
            corners_ref, ids_ref, _ = detector.detectMarkers(frame_ref)
            if ids_ref is not None:
                corners_ref = np.array(corners_ref) + (xmin_ref, ymin_ref)
                corners.append(corners_ref)
                ids.append(ids_ref)
            
            if ids_target is None and ids_ref is None:
                ids = None
            else:   
                ids = np.vstack(ids)
                corners = np.vstack(corners, dtype=np.float32)
        else:
            corners, ids, _ = detector.detectMarkers(frame)

        local_corners_dict[frame_idx] = corners
        local_ids_dict[frame_idx] = ids

        frame_idx += 1
        
        if CROP and ids is not None and np.sum(ids<11) > 0 and np.sum(ids>=11) > 0:
            found = True
            corners = np.array(corners)
            xmin_target, ymin_target = np.max([[0,0],np.min(corners[ids<11].reshape(-1,2),axis=0)-CROP_PADDING],axis=0).astype(np.int32)
            xmax_target, ymax_target = np.min([[frame_width,frame_height],np.max(corners[ids<11].reshape(-1,2),axis=0)+CROP_PADDING],axis=0).astype(np.int32)
            xmin_ref, ymin_ref = np.max([[0,0],np.min(corners[ids>=11].reshape(-1,2),axis=0)-CROP_PADDING],axis=0).astype(np.int32)
            xmax_ref, ymax_ref = np.min([[frame_width,frame_height],np.max(corners[ids>=11].reshape(-1,2),axis=0)+CROP_PADDING],axis=0).astype(np.int32)
            frame = cv2.rectangle(frame, (xmin_target, ymin_target), (xmax_target, ymax_target),(0,255,0),3)
            frame = cv2.rectangle(frame, (xmin_ref, ymin_ref), (xmax_ref, ymax_ref),(0,255,0),3)
        elif CROP:
            found = False

    cap.release()
    return local_corners_dict, local_ids_dict

def runPlot3D(childConn):
    interval = 1000/RATE
    fig_plot3d = plt.figure()
    # ax_target = fig_plot3d.add_subplot(1,2,1, projection='3d')
    # ax_ref = fig_plot3d.add_subplot(1,2,2, projection='3d')
    ax_target = fig_plot3d.add_subplot(projection='3d')
    # ax_target.set_xlim3d(-100,100)
    # ax_target.set_ylim3d(-100,100)
    # ax_target.set_zlim3d(180, 320)
    # points_ref = np.zeros((3,))
    points_target = np.zeros((3,))
    scatter_target = ax_target.scatter(points_target[0], points_target[1], points_target[2])
    # scatter_ref = ax_ref.scatter(points_ref[0], points_ref[1], points_ref[2])
    
    def animate3d(i):
        if childConn.poll():
            points_target, objPoints_target= childConn.recv()
            ax_target.clear()
            # ax_ref.clear()
            ax_target.set(xlim=(-40,40), ylim=(-40,40), zlim=(215,285))
            ax_target.scatter(objPoints_target[0], objPoints_target[1], objPoints_target[2])
            ax_target.scatter(points_target[0], points_target[1], points_target[2])
            # ax_ref.scatter(points_ref[0], points_ref[1], points_ref[2])
    
    animplot = FuncAnimation(fig_plot3d, animate3d, interval=interval)
    plt.show()

def update_kalman(kalman: KalmanFilterCV, poses: list, covars: list):
    final_pose = kalman.predict().reshape((12,1))[0:6]
    # poses = [pose_1,pose_2,pose_3,pose_4,pose_5]
    # poses = []
    # covars = [covar_1,covar_2,covar_3,covar_4,covar_5]
    # covars = []
    kalman_measurement = np.array([])
    covariance_matrix = np.array([])
    num_cameras = 0
    for i in range(len(poses)):
        if poses[i] is not None:
            num_cameras += 1
            if len(kalman_measurement) == 0:
                kalman_measurement = poses[i]
                covariance_matrix = covars[i]        
            else:
                kalman_measurement = np.vstack((kalman_measurement,poses[i]))
                # Set size properly
                # [[1, 0],    [[1, 0, 0],
                #  [0, 1]] ->  [0, 1, 0]
                #              [0, 0, 1]]
                zero_block = np.zeros((covariance_matrix.shape[1],covars[i].shape[0]))
                covariance_matrix = np.block([[covariance_matrix,zero_block],
                                            [zero_block.T, covars[i]]])

    if num_cameras > 0:
        kalman.set_measurement(y_k=kalman_measurement) # 30x1 matrix; C matrix has to be 30x12 because also keeping track of x and x_dot
        kalman.set_measurement_matrices(num_measurements=num_cameras, new_R=covariance_matrix)
        # If can correct, return corrected position
        final_pose = kalman.correct()

    return kalman, final_pose

def twoAngleMean(theta1, theta2):
    if abs(theta1-theta2) > 180:
        newtheta = ((theta1+theta2)/2 + 360) % 360 - 180
    else: 
        newtheta = (theta1+theta2)/2
    return newtheta

def anglesMean(thetas):
    if np.max(thetas) - np.min(thetas) > 180:
        avg = thetas[0]
        N = 1
        for theta in thetas[1:]:
            avg = twoAngleMean(2*avg*N/(N+1), 2*theta/(N+1))
            N += 1
        return avg
    else:
        return np.mean(thetas)
    
def rigidTransform(A, B, repeats=None):
    # Returns the R and t which transform the points in A towards those in B while minimizing least squares error
    # A and B must be of shape 3xN
    
    if repeats is not None:
        if isinstance(repeats, np.ndarray):
            repeats = repeats.astype(int)  # Convert the array of repeats to integers
        else:
            repeats = int(repeats)  # Ensure repeats is an integer if it's a single value
        A = np.repeat(A, repeats, axis=1)
        B = np.repeat(B, repeats, axis=1)
    
    mean_A = np.mean(A, axis=1, keepdims=True)
    mean_B = np.mean(B, axis=1, keepdims=True)
    
    H = (A - mean_A) @ (B - mean_B).T
    U, S, Vh = np.linalg.svd(H)
    
    R = Vh.T @ U.T
    if np.linalg.det(R) < 0:
        Vh[2,:] *= -1
        R = Vh.T @ U.T
    t = -R @ mean_A + mean_B
    
    return R, t

def triangulate(projMats, imgPoints):
    A = np.zeros((len(imgPoints)*2, 4), dtype=np.float32)
    A[::2] = imgPoints[:,[1]] * projMats[:,2] - projMats[:,1]
    A[1::2] = projMats[:,0] - imgPoints[:,[0]] * projMats[:,2]
    
    U, S, Vh = np.linalg.svd(A.T@A)
    worldPoint = Vh[3,:3]/Vh[3,3]
    
    return worldPoint

def plot3D(pts, fig=None, ax=None):
    # pts should be 3xN
    if fig is None: fig = plt.figure()
    if ax is None: ax = fig.add_subplot(projection='3d')
    ax.scatter(pts[0], pts[1], pts[2])
    return fig, ax

if __name__ == "__main__":

    total_files = len([video for video in os.listdir(video_folder) if os.path.isdir(os.path.join(video_folder, video)) and not video.startswith('_')]) * 3
    precomputed_corners_and_ids = {}  # Store precomputed corners and IDs per video to reuse across 3 configs (top. left, right)

    for video in os.listdir(video_folder):
        if video.startswith('_'):
            continue
        video_path = os.path.join(video_folder, video)

        for config_name, camera_configs in camera_configs_dict.items():
            trial = video.split('_')[0]
            config_folder = f'{video_folder}/_{trial}_{config_name.upper()}'
            if not os.path.exists(config_folder):
                os.makedirs(config_folder)
            if os.path.isdir(f'{video_folder}/{video}') and not os.path.exists(f'{config_folder}/{video}_{config_name.lower()}_data.json'):
                # Initialize the data dictionary for this specific camera configuration
                data = {}
                processed_files = len([os.path.join(root, file) 
                                    for root, dirs, files in os.walk(video_folder) 
                                    for file in files if file.endswith('_data.json')])
                total_combinations = len(camera_configs) * len(fusion_combinations) - 3
                combination_counter = 0

                # If corners and IDs are already computed for this video, reuse them
                if video in precomputed_corners_and_ids:
                    all_corners, all_ids = precomputed_corners_and_ids[video]
                    corners_and_ids_collected = True
                else:
                    corners_and_ids_collected = False

                for camera_config in camera_configs:
                    cams = camera_config
                    N_cams = len(cams)

                    for fusion_config in fusion_combinations:
                        STEREO = fusion_config["STEREO"]

                        if OFFSET and not MARKER_MAPPER:
                            if STEREO:
                                targetPoints = dodecaBoard.generate(target_marker_size, target_pentagon_size, (-0.46397128, -0.19475517, 213.41259916), 'centre')
                            else:
                                targetPoints = dodecaBoard.generate(target_marker_size, target_pentagon_size, toolOffset, 'centre') #(-0.53453551, -0.18892355, 223.38273973)
                            refPoints = dodecaBoard.generate(ref_marker_size, ref_pentagon_size)

                        elif OFFSET and MARKER_MAPPER:
                            if STEREO:
                                targetPoints = load_markerMapper(mapperFile_target, (-0.26773738, 0.45818550, -273.27463244))
                            else:
                                targetPoints = load_markerMapper(mapperFile_target, toolOffset) #(0.53453551, 0.18892355, -284.62613973)
                            refPoints = load_markerMapper(mapperFile_ref)

                        elif MARKER_MAPPER and not OFFSET:
                            targetPoints = load_markerMapper(mapperFile_target)
                            refPoints = load_markerMapper(mapperFile_ref)

                        else:
                            if STEREO:
                                if calibFile_target: 
                                    targetPoints = np.loadtxt(calibFile_target, dtype=np.float32).reshape(-1,4,3)
                                else: 
                                    targetPoints = dodecaBoard.generate(target_marker_size, target_pentagon_size)
                                if calibFile_ref:
                                    refPoints = np.loadtxt(calibFile_ref, dtype=np.float32).reshape(-1,4,3)
                                else:
                                    refPoints = dodecaBoard.generate(ref_marker_size, ref_pentagon_size)
                            else: # Default marker mapper with offset for mono
                                targetPoints = load_markerMapper(mapperFile_target, toolOffset) #(0.53453551, 0.18892355, -284.62613973)
                                refPoints = load_markerMapper(mapperFile_ref)
                        # else:
                        #         if calibFile_target: 
                        #             targetPoints = np.loadtxt(calibFile_target, dtype=np.float32).reshape(-1,4,3)
                        #         else: 
                        #             targetPoints = dodecaBoard.generate(target_marker_size, target_pentagon_size)
                        #         if calibFile_ref:
                        #             refPoints = np.loadtxt(calibFile_ref, dtype=np.float32).reshape(-1,4,3)
                        #         else:
                        #             refPoints = dodecaBoard.generate(ref_marker_size, ref_pentagon_size)

                        targetTagCorners = np.array([[-target_marker_size/2, target_marker_size/2, 0], 
                                                    [target_marker_size/2, target_marker_size/2, 0],
                                                    [target_marker_size/2, -target_marker_size/2, 0],
                                                    [-target_marker_size/2, -target_marker_size/2, 0]], dtype=np.float32)
                        refTagCorners = np.array([[-ref_marker_size/2, ref_marker_size/2, 0],
                                                [ref_marker_size/2, ref_marker_size/2, 0],
                                                [ref_marker_size/2, -ref_marker_size/2, 0],
                                                [-ref_marker_size/2, -ref_marker_size/2, 0]], dtype=np.float32)
                        tagPoints = np.vstack([targetPoints, refPoints])

                        target_board = cv2.aruco.Board(targetPoints, aruco_dict, np.arange(11))
                        ref_board = cv2.aruco.Board(refPoints, aruco_dict, np.arange(11,22))

                        N_tagIds = len(target_board.getIds()) + len(ref_board.getIds())

                        criteria_refineLM = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_COUNT, 200, 1.19209e-07)

                        if STEREO and N_cams == 1:
                            continue

                        combination_counter += 1

                        if IGT:
                            server = pyigtl.OpenIGTLinkServer(port=IGT_Port)
                            print("IGTL server started!")

                        if KALMAN: kalman_filter = KalmanFilterCV(RATE)
                        processes = []
                        parentConns = []
                        childConns = []
                        stopEvent = mp.Event()
                        barrier = mp.Barrier(len(cams)+1)
                        
                        calibData = pd.read_json(calibFile, orient='records')
                        for i, id in enumerate(calibData.loc[:,'id']): 
                            calibData.at[i,'id'] = tuple(id) if type(id) == list else id
                        calibData = calibData.set_index('id')
                        calibData = calibData.applymap(np.array)
                        calibData = calibData.replace(np.nan, None)
                        cameraMatrices = np.array(calibData.loc[cams, 'cameraMatrix'])
                        R_toWorld = np.zeros((N_cams,3,3), dtype=np.float64)  
                        T_toWorld = np.zeros((N_cams,3,1), dtype=np.float64)
                        projMats = np.zeros((N_cams, 3, 4), dtype=np.float64)  
                        for i, cam in enumerate(cams):
                            R_fromWorld, T_fromWorld = calibData.at[cam, 'R'], calibData.at[cam, 'T']
                            R_toWorld[i] = R_fromWorld.T
                            T_toWorld[i] = -R_fromWorld.T @ T_fromWorld
                            projMats[i] = cameraMatrices[i] @ np.hstack([R_fromWorld,T_fromWorld])
                        cams_indices = np.arange(len(cams))
                        camPairs_indices = list(itertools.combinations(list(range(N_cams)), 2))
                        
                        start_time = time.time()
                        times = []
                        pose_estimates = []

                        if not corners_and_ids_collected:
                            print(f"[{video}] Collecting marker corners and IDs across all cameras and frames.\n")

                            all_corners = {}
                            all_ids = {}
                            
                            for cam in cams:
                                cameraMatrix = calibData.at[cam, "cameraMatrix"]
                                distCoeffs = calibData.at[cam, "distCoeffs"]
                                corners_dict, ids_dict = runCam(cam, cameraMatrix, distCoeffs, video_path)
                                all_corners[cam] = corners_dict
                                all_ids[cam] = ids_dict
                                 # Print the size of corners and ids for the current camera
                                print(f"[{video}] Collected corners and IDs for {cam}/5 cameras.", end='\r')
                            
                            precomputed_corners_and_ids[video] = (all_corners, all_ids)  # Store for reuse across 3 configs (top, left, right)
                            corners_and_ids_collected = True
                            print(f"[{video}] Corners and IDs collection completed. Beginning pose estimation.\n")

                        if PLOTTING_3D:
                            parentConn_plot3d, childConn_plot3d = mp.Pipe(True)
                            process_plot3d = mp.Process(target=runPlot3D, args=(childConn_plot3d,))
                            process_plot3d.start()

                        print(f"[{video}] (File {processed_files+1}/{total_files} - Step {combination_counter}/{total_combinations}) "
                            f"Config: {config_name} | Active Cameras: {N_cams} | Fusion Method: {fusion_desc(fusion_config)}")
                        
                        final_pose = np.zeros((6,1))

                        timestamps = [0] * len(cams)
                        
                        frameTime = 1/RATE
                        lastFrameTime = time.time()

                        # Iterate through all the frames of the stored corners and ids
                        for frame_idx in range(len(all_corners[cams[0]])):  # Assuming all cams have the same number of frames
                            poses = []
                            poses_full = [None] * N_cams
                            zero_covars = []

                            # Extract corners and ids for this frame
                            frame_corners = {cam: all_corners[cam][frame_idx] for cam in cams}
                            frame_ids = {cam: all_ids[cam][frame_idx] for cam in cams}

                            if STEREO:
                                allCorners = np.full((N_tagIds, N_cams, 4, 2), -1, dtype=np.float32)
                                for camIndex, cam in enumerate(cams):
                                    corners, ids = frame_corners[cam], frame_ids[cam]
                                    if ids is not None:
                                        allCorners[ids, camIndex, :, :] = corners

                                foundTags = np.any(allCorners != -1, axis=(2, 3))
                                reprojectTags = [np.where(foundTags[:, camIndex])[0] for camIndex in cams_indices]
                                triangulateTags = [np.where(np.all(foundTags[:, [cam1, cam2]], axis=1))[0].tolist() for cam1, cam2 in camPairs_indices]
                                triangulateCams = [np.where(foundTags[tag])[0].tolist() for tag in range(N_tagIds)]

                                objPoints_all = []
                                worldPoints_all = []
                                tags_all = []

                                if STEREO_REPROJECT:
                                    for tags_reproject, camIndex in zip(reprojectTags, cams_indices):
                                        if len(tags_reproject) == 0:
                                            continue
                                        tags_found = np.where(foundTags[:, camIndex])[0]
                                        imgPoints = allCorners[tags_found, camIndex]
                                        objPoints_target, imgPoints_target = target_board.matchImagePoints(imgPoints, tags_found)
                                        objPoints_ref, imgPoints_ref = ref_board.matchImagePoints(imgPoints, tags_found)

                                        ret_target, rvec_target, tvec_target = cv2.solvePnP(objPoints_target, imgPoints_target, cameraMatrices[camIndex], None, flags=cv2.SOLVEPNP_ITERATIVE)
                                        ret_ref, rvec_ref, tvec_ref = cv2.solvePnP(objPoints_ref, imgPoints_ref, cameraMatrices[camIndex], None, flags=cv2.SOLVEPNP_ITERATIVE)

                                        cv2.solvePnPRefineLM(objPoints_target, imgPoints_target, cameraMatrices[camIndex], None, rvec_target, tvec_target, criteria_refineLM)
                                        cv2.solvePnPRefineLM(objPoints_ref, imgPoints_ref, cameraMatrices[camIndex], None, rvec_ref, tvec_ref)

                                        objPoints_target_reproject = tagPoints[tags_reproject[tags_reproject < 11]]
                                        objPoints_ref_reproject = tagPoints[tags_reproject[tags_reproject >= 11]]
                                        objPoints_reproject = np.vstack([objPoints_target_reproject, objPoints_ref_reproject])

                                        camPoints_target = cv2.Rodrigues(rvec_target)[0] @ objPoints_target_reproject.reshape((-1, 3)).T + tvec_target
                                        camPoints_ref = cv2.Rodrigues(rvec_ref)[0] @ objPoints_ref_reproject.reshape((-1, 3)).T + tvec_ref

                                        worldPoints = R_toWorld[camIndex] @ np.hstack([camPoints_target, camPoints_ref]) + T_toWorld[camIndex]

                                        objPoints_all.append(objPoints_reproject.reshape((-1, 4, 3)))
                                        worldPoints_all.append(worldPoints.T.reshape((-1, 4, 3)))
                                        tags_all += tags_reproject.tolist()

                                if STEREO_TRIANGULATE_ALL:
                                    # Triangulation across every view
                                    for tag, camIndices in zip(np.arange(N_tagIds), triangulateCams):
                                        if len(camIndices) < 2: continue
                                        objPoints = tagPoints[tag].reshape((1,4,3))
                                        worldPoints = np.zeros((1,4,3))
                                        
                                        for corner in range(4):
                                            imgPoints = allCorners[tag, camIndices, corner]
                                            worldPoints[0,corner] = triangulate(projMats[camIndices], imgPoints)
                                        
                                        objPoints_all.append(objPoints)
                                        worldPoints_all.append(worldPoints)
                                        tags_all.append(tag)

                                if objPoints_all and worldPoints_all:
                                    objPoints_all = np.vstack(objPoints_all)
                                    worldPoints_all = np.vstack(worldPoints_all)

                                    tags_all = np.array(tags_all)
                                    objPoints_target = objPoints_all[tags_all < 11].reshape((-1, 3)).T
                                    objPoints_ref = objPoints_all[tags_all >= 11].reshape((-1, 3)).T
                                    worldPoints_target = worldPoints_all[tags_all < 11].reshape((-1, 3)).T
                                    worldPoints_ref = worldPoints_all[tags_all >= 11].reshape((-1, 3)).T

                                    if STEREO_TRIANGULATE_ALL:
                                        # repeats_target = np.repeat(np.sum(foundTags, axis=1)[tags_all[tags_all<11]]-1, 4)
                                        # repeats_ref = np.repeat(np.sum(foundTags, axis=1)[tags_all[tags_all>=11]]-1, 4)
                                        
                                        repeats_target = np.array([np.sum(np.arange(1,n)) for n in np.sum(foundTags, axis=1)[tags_all[tags_all<11]]])
                                        repeats_target = np.repeat(repeats_target, 4)
                                        repeats_ref = np.array([np.sum(np.arange(1,n)) for n in np.sum(foundTags, axis=1)[tags_all[tags_all>=11]]])
                                        repeats_ref = np.repeat(repeats_ref, 4)
                                        # repeats_target, repeats_ref = None, None
                                    else:
                                        repeats_target, repeats_ref = None, None
                                    
                                    R_target, t_target = rigidTransform(objPoints_target, worldPoints_target, repeats_target)
                                    R_ref, t_ref = rigidTransform(objPoints_ref, worldPoints_ref, repeats_ref)

                                    rel_trans = R_ref.T @ (t_target - t_ref)
                                    rel_rot_matrix = R_target.T @ R_ref
                                    rel_rot_ypr = R.from_matrix(rel_rot_matrix).as_euler('ZYX', degrees=True).reshape((3, 1))
                                    pose = np.vstack((rel_trans, rel_rot_ypr))

                                    if not np.isnan(pose).any():
                                        poses.append(pose)
                                        poses_full[0] = pose
                                        zero_covars = np.zeros((6, 1))

                                        if PLOTTING_3D:
                                            objToWorldPoints_target = R_target @ tagPoints[:11].reshape((-1, 3)).T + t_target
                                            parentConn_plot3d.send((worldPoints_target, objToWorldPoints_target))
                                    else:
                                        print(f"Skipping pose due to NaN values for [{video}] {N_cams} Cam {fusion_desc(fusion_config)}.")
                                else:
                                    print(f"Skipping stacking as objPoints_all or worldPoints_all is empty for [{video}] {N_cams} Cam {fusion_desc(fusion_config)}.")

                            else:
                                for i, cam in enumerate(cams):
                                    pose, covar = poseEstimator.estimate_pose_board(ref_board, target_board, frame_corners[cam], frame_ids[cam], cameraMatrices[i])
                                    if pose is not None:
                                        poses.append(pose)
                                        poses_full[i] = pose
                                        zero_covars.append(covar)
                                    else:
                                        poses_full.append(pose)

                            if len(poses) > 0:
                                if STEREO:
                                    poses = np.array(poses)
                                    final_pose[:3] = np.mean(poses[:, :3], axis=0)
                                    for i in range(3, 6):
                                        final_pose[i] = anglesMean(poses[:, i])

                                    curr_time = time.time() - start_time
                                    if TIME_RANGE[0] <= curr_time <= TIME_RANGE[1]:
                                        times.append(curr_time)
                                        pose_estimates.append(final_pose.flatten().tolist())
                                else:
                                    final_poses_per_camera = {}  # Dictionary to store final pose per camera
                                    # Store final poses for each camera without Kalman filter
                                    for cam_idx, pose in enumerate(poses):
                                        if pose is not None:
                                            final_poses_per_camera[cam_idx] = pose.flatten().tolist()

                                    # Record the time and the individual camera poses
                                    curr_time = time.time() - start_time
                                    if TIME_RANGE[0] <= curr_time <= TIME_RANGE[1]:
                                        times.append(curr_time)
                                        pose_estimates.append(final_poses_per_camera)

                            else:
                                continue

                            if IGT:
                                transform = np.eye(4)
                                transform[:3, :3] = R.from_euler(seq='ZYX', angles=final_pose[3:].ravel(), degrees=True).as_matrix().T
                                transform[:3, 3] = final_pose[:3].flatten()
                                position_message = pyigtl.TransformMessage(transform, device_name="PointerDevice")
                                server.send_message(position_message, wait=False)
                            
                        if IGT: 
                            server.stop()
                            print('Server port closed succesfully!')
                        
                        data[f"{N_cams} Cam {fusion_desc(fusion_config)}"] = {
                            'poses': pose_estimates,
                            'times': times
                        }

                        print(f"[{video}] ({config_name}) {N_cams} Cam {fusion_desc(fusion_config)} data has been collected.\n")
                
                filename = f'{config_folder}/{video}_{config_name.lower()}_init_data.json'
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=4)
                
                print(f"The data file for [{video}] ({config_name}) has been saved as {filename}.\n")

