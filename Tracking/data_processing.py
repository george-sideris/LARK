import json
import os
import numpy as np
from util.landmark_registration import register, apply_transform, calculate_TRE
import matplotlib.pyplot as plt
import sys
import contextlib
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree, KDTree
import shutil
import itertools as it
import trimesh

base_folder = '/home/george/MultiCameraTracking/Registration/Landmark_Registration_Trials'
TRIALS = [4]

GRID = False
LANDMARK_TEST = True
FLIP = False  # (R6 <-> 9, R8 <-> 6)
MEAN = False
PLOT = True
SAVE_PLOTS = False  # If True, save plots to folder; if False, show in windows
SHOW_3D_MODEL = True  # Show 3D head model overlay in trajectory plots
INDEX = 10
N_CAMS = 5
data_folders = [f'{base_folder}/' + ('G' if GRID else 'H') + f'T{trial:02d}' for trial in TRIALS]  # Directory that contains folders with recordings of each point of interest
N_trials = len(TRIALS)
fusion_methods = ['Monocular Pose Fusion', 'Monocular Pose Fusion Kalman', 'Monocular Pose Fusion Kalman Adaptive', 'Triangulation', 'Triangulation Kalman', 'Triangulation Kalman Adaptive']

# Mapping from data file method names to display names
method_display_names = {
    'Mono': 'Monocular Pose Fusion',
    'Mono Kalman': 'Monocular Pose Fusion with Kalman Filtering', 
    'Mono Kalman Adaptive': 'Monocular Pose Fusion with Adaptive Kalman Filtering',
    'Stereo': 'Triangulation',
    'Stereo Kalman': 'Triangulation with Kalman Filtering',
    'Stereo Kalman Adaptive': 'Triangulation with Adaptive Kalman Filtering'
}

def get_display_key(key):
    """Convert data key to display key with proper method names."""
    if ' Cam ' in key:
        camera_config, method = key.split(' Cam ')
        display_method = method_display_names.get(method, method)
        return f"{camera_config} Camera {display_method}"
    return key

if GRID:
    POINTS_FRE = ['01', '03', '05', '11', '13', '15', '21', '23', '25']
    POINTS_TRE = ['02', '04', '06', '07', '08', '09', '10', '12', '14', '16', '17', '18', '19', '20', '22', '24']
    TRAJECTORIES = ['RT1', 'RT2', 'RT3', 'RT4']
elif FLIP:
    POINTS_FRE = ['R1', 'R2', 'R3', 'R5', '09', 'R7', '06', 'R9']
    POINTS_TRE = ['02', '03', '04', '05', 'R8', '07', '08', 'R6', '10']
else:
    POINTS_FRE = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9']
    POINTS_TRE = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    TRAJECTORIES = ['RB', 'RF']
    
if GRID:
    CONFIGS = [f'C{i}' for i in range(31)]
else:
    CONFIGS = ['top', 'left', 'right']
    

# For grid experiment, use the ground truth grid:
if GRID: 
    landmark_points = {
        '01' : [0., 0., 0.],
        '02' : [100., 0., 0.],
        '03' : [200., 0., 0.],
        '04' : [300., 0., 0.],
        '05' : [400., 0., 0.],
        '06' : [0., 100., 0.],
        '07' : [100., 100., 0.],
        '08' : [200., 100., 0.],
        '09' : [300., 100., 0.],
        '10' : [400., 100., 0.],
        '11' : [0., 200., 0.],
        '12' : [100., 200., 0.],
        '13' : [200., 200., 0.],
        '14' : [300., 200., 0.],
        '15' : [400., 200., 0.],
        '16' : [0., 300., 0.],
        '17' : [100., 300., 0.],
        '18' : [200., 300., 0.],
        '19' : [300., 300., 0.],
        '20' : [400., 300., 0.],
        '21' : [0., 400., 0.],
        '22' : [100., 400., 0.],
        '23' : [200., 400., 0.],
        '24' : [300., 400., 0.],
        '25' : [400., 400., 0.]
    }
    
else:
    landmark_points = {
    'R1' : [-32.45655234245545, -261.07892978923536, -284.2474975133015],
    'R2' : [-46.08230033926727, -246.87811937673945, -288.2474975133015],
    'R3' : [-18.875123469712097, -247.9627579593944, -288.2474975133015],
    'R4' : [-89.72569313752015, -226.7922881506288, -291.2474975133015],
    'R5' : [23.288812752567793, -226.1610793832458, -291.2474975133016],
    'R6' : [-111.85976315415175, -161.08839598228957, -303.24749751330154],
    'R7' : [-112.90036878389031, -167.05432381335226, -318.24749751330154],
    'R8' : [43.08883637606088, -154.4931755124505, -308.24749751330154],
    'R9' : [42.53126353007378, -157.0715394662325, -323.24749751330154],
    '01' : [-33.77318050965114, -62.198733345188266, -311.07954203083057],
    '02' : [-34.60491720735975, -68.61256180321587, -218.98930180713296],
    '03' : [-34.127227881284725, -126.40123073234138, -180.10008286489867],
    '04' : [-36.12722788128472, -209.13836577750476, -190.7512213223227],
    '05' : [31.41372492957544, -86.95342989740425, -272.44521854046843],
    '06' : [37.83440512766741, -123.31362911288862, -222.10761945430107],
    '07' : [27.610779589151278, -194.28518102835042, -223.0204271663359],
    '08' : [-96.81155464296913, -85.96260189765843, -274.4082803595634],
    '09' : [-101.79489617603154, -111.68674116612557, -222.66858103254475],
    '10' : [-99.39393820482444, -196.10450748819855, -224.25848374003647],
    }
    
for key, val in landmark_points.items():
    landmark_points[key] = np.array(val)

# Load trajectories (ground truth for trajectories)
landmark_front_trajectory = np.load('../Registration/Landmark/Landmark Trajectories/front_trajectory.npy')
landmark_back_trajectory = np.load('../Registration/Landmark/Landmark Trajectories/back_trajectory.npy')

# Only load grid trajectories if running grid experiments
if GRID:
    trajectory1 = np.load("../Registration/Landmark/Landmark Trajectories/grid_traj1.npy")
    trajectory2 = np.load("../Registration/Landmark/Landmark Trajectories/grid_traj2.npy")
    trajectory3 = np.load("../Registration/Landmark/Landmark Trajectories/grid_traj3.npy")
    trajectory4 = np.load("../Registration/Landmark/Landmark Trajectories/grid_traj4.npy")

# --------------------------
# Utility functions
# --------------------------
def flip_directory_structure(original_folder):
    flipped_folder = f"{original_folder}_flipped"
    shutil.copytree(original_folder, flipped_folder)
    print(f"Created flipped directory: {flipped_folder}")
    return flipped_folder

def flip_config_files(flipped_folder):
    for config in ['_TOP', '_LEFT', '_RIGHT']:
        config_folder = os.path.join(flipped_folder, f"_{os.path.basename(flipped_folder).split('_')[0]}{config}")
        if os.path.exists(config_folder):
            for file_name in os.listdir(config_folder):
                if "R6" in file_name:
                    old_file = os.path.join(config_folder, file_name)
                    temp_file = os.path.join(config_folder, file_name.replace("R6", "R6_temp"))
                    os.rename(old_file, temp_file)
                    print(f"Temporarily renamed {old_file} to {temp_file}")
                elif "R8" in file_name:
                    old_file = os.path.join(config_folder, file_name)
                    temp_file = os.path.join(config_folder, file_name.replace("R8", "R8_temp"))
                    os.rename(old_file, temp_file)
                    print(f"Temporarily renamed {old_file} to {temp_file}")
            for file_name in os.listdir(config_folder):
                if "09" in file_name:
                    old_file = os.path.join(config_folder, file_name)
                    new_file = os.path.join(config_folder, file_name.replace("09", "R6"))
                    os.rename(old_file, new_file)
                    print(f"Renamed {old_file} to {new_file}")
                elif "06" in file_name:
                    old_file = os.path.join(config_folder, file_name)
                    new_file = os.path.join(config_folder, file_name.replace("06", "R8"))
                    os.rename(old_file, new_file)
                    print(f"Renamed {old_file} to {new_file}")
            for file_name in os.listdir(config_folder):
                if "R6_temp" in file_name:
                    temp_file = os.path.join(config_folder, file_name)
                    final_file = os.path.join(config_folder, file_name.replace("R6_temp", "09"))
                    os.rename(temp_file, final_file)
                    print(f"Renamed {temp_file} to {final_file}")
                elif "R8_temp" in file_name:
                    temp_file = os.path.join(config_folder, file_name)
                    final_file = os.path.join(config_folder, file_name.replace("R8_temp", "06"))
                    os.rename(temp_file, final_file)
                    print(f"Renamed {temp_file} to {final_file}")
        else:
            print(f"Config folder {config_folder} does not exist.")

def flip_IGTPos_files(flipped_folder):
    video_folder = flipped_folder
    pairs_to_flip = [("R6", "09"), ("R8", "06")]
    for pair in pairs_to_flip:
        folder_1 = os.path.join(video_folder, f"{os.path.basename(flipped_folder).split('_')[0]}_{pair[0]}")
        folder_2 = os.path.join(video_folder, f"{os.path.basename(flipped_folder).split('_')[0]}_{pair[1]}")
        if os.path.exists(folder_1) and os.path.exists(folder_2):
            file_1 = os.path.join(folder_1, "IGTPos.txt")
            file_2 = os.path.join(folder_2, "IGTPos.txt")
            temp_file_1 = file_1 + "_temp"
            temp_file_2 = file_2 + "_temp"
            if os.path.exists(file_1):
                os.rename(file_1, temp_file_1)
                print(f"Temporarily renamed {file_1} to {temp_file_1}")
            if os.path.exists(file_2):
                os.rename(file_2, temp_file_2)
                print(f"Temporarily renamed {file_2} to {temp_file_2}")
            if os.path.exists(temp_file_1):
                os.rename(temp_file_1, file_2)
                print(f"Renamed {temp_file_1} to {file_2}")
            if os.path.exists(temp_file_2):
                os.rename(temp_file_2, file_1)
                print(f"Renamed {temp_file_2} to {file_1}")
        else:
            print(f"One of the folders {folder_1} or {folder_2} does not exist. Skipping.")

@contextlib.contextmanager
def suppress_print():
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        sys.stdout = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            
def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    
def get_pose(data, mean=False):
    poses = np.array(data['poses'])
    if poses.size == 0:
        return None
    if mean:
        valid_poses = poses[~np.isnan(poses).any(axis=1)]
        if valid_poses.size > 0:
            return valid_poses.mean(axis=0)
        else:
            return poses.mean(axis=0)
    else:
        valid_poses = poses[~np.isnan(poses).any(axis=1)]
        if valid_poses.size > 0:
            return valid_poses[-1]
        else:
            return poses[-1]

def plot_3d_points(real_fiducial_points, transformed_fiducial_points,
                   real_target_points, transformed_target_points, title):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Enhanced color scheme and styling
    ax.scatter(real_fiducial_points[:, 0], real_fiducial_points[:, 1], real_fiducial_points[:, 2], 
               c='#E74C3C', marker='o', s=80, alpha=0.8, edgecolors='white', linewidth=1,
               label='Ground Truth Fiducial Points')
    ax.scatter(transformed_fiducial_points[:, 0], transformed_fiducial_points[:, 1], transformed_fiducial_points[:, 2], 
               c='#3498DB', marker='s', s=60, alpha=0.8, edgecolors='white', linewidth=1,
               label='Measured Fiducial Points')
    ax.scatter(real_target_points[:, 0], real_target_points[:, 1], real_target_points[:, 2], 
               c='#E67E22', marker='D', s=100, alpha=0.8, edgecolors='black', linewidth=1.5,
               label='Ground Truth Target Points')
    ax.scatter(transformed_target_points[:, 0], transformed_target_points[:, 1], transformed_target_points[:, 2], 
               c='#9B59B6', marker='^', s=80, alpha=0.8, edgecolors='black', linewidth=1.5,
               label='Measured Target Points')
    
    # Enhanced labels and styling
    ax.set_xlabel('X (mm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (mm)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z (mm)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Improved legend
    legend = ax.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=10, 
                      frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Grid and styling
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)
    
    # Set background color
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    
    plt.tight_layout()
    return fig

def plot_3d_combined(real_trajectories, measured_trajectories, 
                     real_fiducial_points=None, transformed_fiducial_points=None,
                     real_target_points=None, transformed_target_points=None,
                     title="Combined Trajectory and Point Registration", distance_threshold=10):
    """Combined plot showing trajectories, registration points, landmarks, and head model."""
    global landmark_points  # Move global declaration to the top
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    def is_consecutive(p1, p2):
        return np.linalg.norm(p1 - p2) < distance_threshold
    
    # Clean trajectories
    real_trajectories = [traj[~np.isnan(traj).any(axis=1)] for traj in real_trajectories]
    measured_trajectories = [traj[~np.isnan(traj).any(axis=1)] for traj in measured_trajectories]
    
    # Calculate offset to shift everything to positive coordinate space
    # Collect all data points first
    all_data_points = []
    if real_trajectories + measured_trajectories:
        all_data_points.extend(real_trajectories + measured_trajectories)
    
    # Add landmark points
    landmark_array = np.array([point for point in landmark_points.values()])
    all_data_points.append(landmark_array)
    
    # Add registration points if provided
    if real_fiducial_points is not None:
        all_data_points.append(real_fiducial_points)
    if transformed_fiducial_points is not None:
        all_data_points.append(transformed_fiducial_points)
    if real_target_points is not None:
        all_data_points.append(real_target_points)
    if transformed_target_points is not None:
        all_data_points.append(transformed_target_points)
    
    if all_data_points:
        all_points = np.concatenate(all_data_points)
        # Calculate offset to make minimum coordinates start near zero
        margin = 10
        data_offset = -all_points.min(axis=0) + margin
        
        # Apply offset to trajectories
        real_trajectories = [traj + data_offset if traj.shape[0] > 0 else traj for traj in real_trajectories]
        measured_trajectories = [traj + data_offset if traj.shape[0] > 0 else traj for traj in measured_trajectories]
        
        # Store offset for later use with landmarks and head model
        plot_offset = data_offset
    else:
        plot_offset = np.array([0, 0, 0])
    
    # Enhanced color palettes
    real_colors = ['#E74C3C', '#C0392B', '#E67E22', '#D35400']
    measured_colors = ['#3498DB', '#2980B9', '#9B59B6', '#8E44AD']
    markers = ['o', '^', 's', 'D', '*', 'p']
    
    # Plot real trajectories as smooth curves with consecutive point checking
    ground_truth_labeled = False
    for i, traj in enumerate(real_trajectories):
        if traj.shape[0] < 2:
            continue
        color = real_colors[i % len(real_colors)]
        label = "Ground Truth Trajectory" if not ground_truth_labeled else ""
        
        # Plot only consecutive segments to avoid random connections
        segments_plotted = False
        for j in range(len(traj)-1):
            if is_consecutive(traj[j], traj[j+1]):
                ax.plot(traj[j:j+2, 0], traj[j:j+2, 1], traj[j:j+2, 2],
                        color=color, linewidth=3, alpha=0.8, 
                        label=label if not segments_plotted else "")
                segments_plotted = True
                if segments_plotted and not ground_truth_labeled:
                    ground_truth_labeled = True
    
    # Plot measured trajectories with enhanced styling
    measured_labeled = False
    for i, traj in enumerate(measured_trajectories):
        if traj.shape[0] < 2:
            continue
        color = measured_colors[i % len(measured_colors)]
        label = "Tracked Trajectory" if not measured_labeled else ""
        
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                color=color, marker='o', markersize=6, linestyle='--',
                linewidth=2, alpha=0.9, markeredgecolor='white', markeredgewidth=0.5,
                label=label)
        measured_labeled = True
    
    # Plot landmarks with offset for positive axes
    landmark_labeled = False
    for i, (landmark, coords) in enumerate(landmark_points.items()):
        offset_coords = coords + data_offset
        ax.scatter(offset_coords[0], offset_coords[1], offset_coords[2],
                  color='gold', s=120, alpha=0.9, edgecolors='black',
                  linewidth=1.5, marker='o', 
                  label='Anatomical Landmarks' if not landmark_labeled else "")
        ax.text(offset_coords[0], offset_coords[1], offset_coords[2],
                f'{landmark}', fontsize=8, fontweight='bold')
        landmark_labeled = True
    
    # Plot registration points if provided
    if real_fiducial_points is not None:
        real_fiducial_offset = real_fiducial_points + data_offset
        ax.scatter(real_fiducial_offset[:, 0], real_fiducial_offset[:, 1], real_fiducial_offset[:, 2],
                  c='#E74C3C', marker='o', s=80, alpha=0.8, edgecolors='white', linewidth=1,
                  label='Ground Truth Fiducial Points')
    
    if transformed_fiducial_points is not None:
        transformed_fiducial_offset = transformed_fiducial_points + data_offset
        ax.scatter(transformed_fiducial_offset[:, 0], transformed_fiducial_offset[:, 1], transformed_fiducial_offset[:, 2],
                  c='#3498DB', marker='s', s=60, alpha=0.8, edgecolors='white', linewidth=1,
                  label='Tracked Fiducial Points')
    
    if real_target_points is not None:
        real_target_offset = real_target_points + data_offset
        ax.scatter(real_target_offset[:, 0], real_target_offset[:, 1], real_target_offset[:, 2],
                  c='#E67E22', marker='D', s=100, alpha=0.8, edgecolors='black', linewidth=1.5,
                  label='Ground Truth Target Points')
    
    if transformed_target_points is not None:
        transformed_target_offset = transformed_target_points + data_offset
        ax.scatter(transformed_target_offset[:, 0], transformed_target_offset[:, 1], transformed_target_offset[:, 2],
                  c='#9B59B6', marker='^', s=80, alpha=0.8, edgecolors='black', linewidth=1.5,
                  label='Tracked Target Points')
    
    # Load and plot 3D head model overlay if enabled
    transformed_vertices = None  # Initialize for axis calculation
    if SHOW_3D_MODEL:
        head_model_path = '../IBIS/GS Head Landmark Shell v2.stl'
        print(f"Checking for 3D model at: {head_model_path}")
        if os.path.exists(head_model_path):
            print("3D model file found, loading...")
            vertices, faces = load_3d_model(head_model_path)
            if vertices is not None:
                print("3D model loaded successfully, transforming to landmark coordinates...")
                transformed_vertices = transform_model_to_landmarks(vertices, landmark_points)
                if transformed_vertices is not None:
                    print("Plotting transformed wireframe...")
                    # Apply the same offset to head model
                    transformed_vertices_offset = transformed_vertices + data_offset
                    plot_3d_model_wireframe(ax, transformed_vertices_offset, faces, alpha=0.3, color='gray', linewidth=0.5)
                else:
                    print("Failed to transform 3D model")
            else:
                print("Failed to load 3D model")
        else:
            print(f"3D model file not found at {head_model_path}")
    
    # Enhanced axis limits calculation including all elements
    offset_data_points = []
    if real_trajectories + measured_trajectories:
        offset_data_points.extend(real_trajectories + measured_trajectories)
    
    # Add offset landmark points
    offset_landmark_array = landmark_array + data_offset
    offset_data_points.append(offset_landmark_array)
    
    # Add offset registration points
    if real_fiducial_points is not None:
        offset_data_points.append(real_fiducial_points + data_offset)
    if transformed_fiducial_points is not None:
        offset_data_points.append(transformed_fiducial_points + data_offset)
    if real_target_points is not None:
        offset_data_points.append(real_target_points + data_offset)
    if transformed_target_points is not None:
        offset_data_points.append(transformed_target_points + data_offset)
    
    # Add offset head model if available
    if SHOW_3D_MODEL and transformed_vertices is not None:
        transformed_vertices_offset = transformed_vertices + data_offset
        offset_data_points.append(transformed_vertices_offset)
    
    if offset_data_points:
        all_offset_points = np.concatenate(offset_data_points)
        margin = 10
        
        # Set axes bounds starting from 0
        ax.set_xlim([0, all_offset_points[:, 0].max() + margin])
        ax.set_ylim([0, all_offset_points[:, 1].max() + margin])
        ax.set_zlim([0, all_offset_points[:, 2].max() + margin])
    else:
        # Fallback to landmark-based range with 0+ axes
        range_x = landmark_array[:, 0].max() - landmark_array[:, 0].min() + 40
        range_y = landmark_array[:, 1].max() - landmark_array[:, 1].min() + 40
        range_z = landmark_array[:, 2].max() - landmark_array[:, 2].min() + 40
        ax.set_xlim([0, range_x])
        ax.set_ylim([0, range_y])
        ax.set_zlim([0, range_z])
    
    # Enhanced labels and styling
    ax.set_xlabel('X (mm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (mm)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z (mm)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Improved legend
    legend = ax.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=9,
                      frameon=True, fancybox=True, shadow=True, ncol=1)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Grid and styling
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)
    
    # Enhanced 3D pane styling
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    
    # Improve viewing angle
    ax.view_init(elev=26, azim=-56)
    
    plt.tight_layout()
    return fig

def plot_3d_trajectories(real_trajectories, measured_trajectories, title, distance_threshold=10):
    global landmark_points  # Move global declaration to the top
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    def is_consecutive(p1, p2):
        return np.linalg.norm(p1 - p2) < distance_threshold
    
    # Clean trajectories
    real_trajectories = [traj[~np.isnan(traj).any(axis=1)] for traj in real_trajectories]
    measured_trajectories = [traj[~np.isnan(traj).any(axis=1)] for traj in measured_trajectories]
    
    # Calculate offset to shift everything to positive coordinate space
    # Collect all data points first
    all_data_points = []
    if real_trajectories + measured_trajectories:
        all_data_points.extend(real_trajectories + measured_trajectories)
    
    # Add landmark points
    landmark_array = np.array([point for point in landmark_points.values()])
    all_data_points.append(landmark_array)
    
    if all_data_points:
        all_points = np.concatenate(all_data_points)
        # Calculate offset to make minimum coordinates start near zero
        margin = 10
        data_offset = -all_points.min(axis=0) + margin
        
        # Apply offset to trajectories
        real_trajectories = [traj + data_offset if traj.shape[0] > 0 else traj for traj in real_trajectories]
        measured_trajectories = [traj + data_offset if traj.shape[0] > 0 else traj for traj in measured_trajectories]
        
        # Store offset for later use with landmarks and head model
        plot_offset = data_offset
    else:
        plot_offset = np.array([0, 0, 0])
    
    # Enhanced color palettes
    real_colors = ['#E74C3C', '#C0392B', '#E67E22', '#D35400']
    measured_colors = ['#3498DB', '#2980B9', '#9B59B6', '#8E44AD']
    markers = ['o', '^', 's', 'D', '*', 'p']
    
    # Plot real trajectories as smooth curves with consecutive point checking
    ground_truth_labeled = False
    for i, traj in enumerate(real_trajectories):
        if traj.shape[0] < 2:
            continue
        color = real_colors[i % len(real_colors)]
        label = "Ground Truth Trajectory" if not ground_truth_labeled else ""
        
        # Plot only consecutive segments to avoid random connections
        segments_plotted = False
        for j in range(len(traj)-1):
            if is_consecutive(traj[j], traj[j+1]):
                ax.plot(traj[j:j+2, 0], traj[j:j+2, 1], traj[j:j+2, 2],
                        color=color, linewidth=3, alpha=0.8, 
                        label=label if not segments_plotted else "")
                segments_plotted = True
                if segments_plotted and not ground_truth_labeled:
                    ground_truth_labeled = True
    
    # Plot measured trajectories with enhanced styling
    measured_labeled = False
    for i, traj in enumerate(measured_trajectories):
        if traj.shape[0] < 2:
            continue
        marker = markers[i % len(markers)]
        color = measured_colors[i % len(measured_colors)]
        label = "Tracked Trajectory" if not measured_labeled else ""
        
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                color=color, marker='o', markersize=6, linestyle='--',
                linewidth=2, alpha=0.9, markeredgecolor='white', markeredgewidth=0.5,
                label=label)
        measured_labeled = True
    
    # Load and plot 3D head model overlay if enabled
    transformed_vertices = None  # Initialize for axis calculation
    if SHOW_3D_MODEL:
        head_model_path = '../IBIS/GS Head Landmark Shell v2.stl'
        print(f"Checking for 3D model at: {head_model_path}")
        if os.path.exists(head_model_path):
            print("3D model file found, loading...")
            vertices, faces = load_3d_model(head_model_path)
            if vertices is not None:
                print("3D model loaded successfully, transforming to landmark coordinates...")
                transformed_vertices = transform_model_to_landmarks(vertices, landmark_points)
                if transformed_vertices is not None:
                    print("Plotting transformed wireframe...")
                    # Apply the same offset to head model
                    transformed_vertices_offset = transformed_vertices + plot_offset
                    plot_3d_model_wireframe(ax, transformed_vertices_offset, faces, alpha=0.3, color='gray', linewidth=0.5)
                else:
                    print("Failed to transform 3D model")
            else:
                print("Failed to load 3D model")
        else:
            print(f"3D model file not found at {head_model_path}")
    
    # Enhanced axis limits - focus on actual data range
    # Collect all relevant points for bounds calculation
    all_points_list = []
    
    # Add trajectory points if available
    if real_trajectories + measured_trajectories:
        all_points_list.extend(real_trajectories + measured_trajectories)
    
    # Add landmark points
    landmark_array = np.array([point for point in landmark_points.values()])
    all_points_list.append(landmark_array)
    
    # Prepare FRE and TRE coordinate arrays for bounds calculation
    fre_coords = np.array([landmark_points[point] for point in POINTS_FRE if point in landmark_points])
    tre_coords = np.array([landmark_points[point] for point in POINTS_TRE if point in landmark_points])
    
    # Add head model vertices if available and 3D model is enabled
    if SHOW_3D_MODEL and transformed_vertices is not None:
        all_points_list.append(transformed_vertices)
    
    # Calculate bounds for the offset data (already shifted to positive space)
    offset_data_points = []
    if real_trajectories + measured_trajectories:
        offset_data_points.extend(real_trajectories + measured_trajectories)
    
    # Add offset landmark points
    offset_landmark_array = landmark_array + data_offset
    offset_data_points.append(offset_landmark_array)
    
    # Add offset FRE and TRE points to bounds calculation
    if fre_coords.size > 0:
        offset_data_points.append(fre_coords + data_offset)
    if tre_coords.size > 0:
        offset_data_points.append(tre_coords + data_offset)
    
    # Add offset head model if available
    if SHOW_3D_MODEL and transformed_vertices is not None:
        transformed_vertices_offset = transformed_vertices + data_offset
        offset_data_points.append(transformed_vertices_offset)
    
    if offset_data_points:
        all_offset_points = np.concatenate(offset_data_points)
        margin = 10
        
        # Set axes bounds starting from 0
        ax.set_xlim([0, all_offset_points[:, 0].max() + margin])
        ax.set_ylim([0, all_offset_points[:, 1].max() + margin])
        ax.set_zlim([0, all_offset_points[:, 2].max() + margin])
    else:
        # Fallback to landmark-based range with 0+ axes
        range_x = landmark_array[:, 0].max() - landmark_array[:, 0].min() + 40
        range_y = landmark_array[:, 1].max() - landmark_array[:, 1].min() + 40
        range_z = landmark_array[:, 2].max() - landmark_array[:, 2].min() + 40
        ax.set_xlim([0, range_x])
        ax.set_ylim([0, range_y])
        ax.set_zlim([0, range_z])
    
    # Enhanced labels and styling
    ax.set_xlabel('X (mm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (mm)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z (mm)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Improved legend
    legend = ax.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=9,
                      frameon=True, fancybox=True, shadow=True, ncol=1)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Grid and styling
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)
    
    # Enhanced 3D pane styling
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    
    # Improve viewing angle
    # ax.view_init(elev=33, azim=-22)
    ax.view_init(elev=26, azim=-56)
    
    plt.tight_layout()
    return fig

def load_3d_model(model_path):
    """Load a 3D model (STL or PLY) and return vertices and faces."""
    
    try:
        mesh = trimesh.load(model_path)
        return mesh.vertices, mesh.faces
    except Exception as e:
        print(f"Could not load 3D model from {model_path}: {e}")
        return None, None

def transform_model_to_landmarks(vertices, landmark_dict):
    """Transform 3D model vertices to align with landmark coordinate system."""
    if vertices is None:
        return None
    
    # Get landmark bounds
    landmark_array = np.array([point for point in landmark_dict.values()])
    landmark_center = landmark_array.mean(axis=0)
    landmark_size = landmark_array.max(axis=0) - landmark_array.min(axis=0)
    
    # Get model bounds
    model_center = vertices.mean(axis=0)
    model_size = vertices.max(axis=0) - vertices.min(axis=0)
    
    # Calculate transformation with proper proportional scaling
    # Use individual axis scaling to prevent squishing
    scale_factors = landmark_size / model_size
    # Use uniform scaling based on the largest dimension to maintain proportions
    scale_factor = np.max(scale_factors) * 1.1  # Keep original proportions
    
    # Apply transformation: center, scale uniformly, then translate to landmark space
    transformed_vertices = vertices - model_center  # Center at origin
    transformed_vertices *= scale_factor  # Scale uniformly to maintain proportions
    transformed_vertices += landmark_center  # Move to landmark center
    
    print(f"Model transformation: scale={scale_factor:.6f}, landmark_center={landmark_center}")
    print(f"Transformed model bounds: {transformed_vertices.min(axis=0)} to {transformed_vertices.max(axis=0)}")
    
    return transformed_vertices

def calculate_vertex_density(vertices, radius=15.0):
    """Calculate local vertex density for each vertex to identify main regions."""
    tree = KDTree(vertices)
    densities = []
    
    for vertex in vertices:
        # Count neighbors within radius
        neighbors = tree.query_ball_point(vertex, radius)
        densities.append(len(neighbors))
    
    return np.array(densities)

def plot_3d_model_wireframe(ax, vertices, faces, alpha=0.35, color='gray', linewidth=0.5):
    """Plot a 3D model as a connected wireframe mesh with enhanced visibility."""
    if vertices is None or faces is None:
        return
    
    print(f"Plotting 3D model wireframe with {len(vertices)} vertices, {len(faces)} faces")
    
    # Calculate vertex density to identify main skull region
    vertex_density = calculate_vertex_density(vertices, radius=15.0)  # 15mm radius
    density_threshold = np.percentile(vertex_density, 15)  # Keep top 85% density regions
    
    # Create edge set to avoid duplicating edges
    edges = set()
    sampled_faces = faces[::2]  # Sample every 2nd face for denser coverage
    
    # Extract unique edges from faces, filtering by vertex density
    for face in sampled_faces:
        if len(face) >= 3:
            # Check if this face is in a dense region (main skull)
            face_density = np.mean([vertex_density[v] for v in face if v < len(vertex_density)])
            if face_density >= density_threshold:
                for i in range(len(face)):
                    edge = tuple(sorted([face[i], face[(i+1) % len(face)]]))
                    edges.add(edge)
    
    # Additional filtering: remove very long edges (likely spanning gaps)
    filtered_edges = []
    for edge in edges:
        v1, v2 = edge
        if v1 < len(vertices) and v2 < len(vertices):
            p1, p2 = vertices[v1], vertices[v2]
            edge_length = np.linalg.norm(p2 - p1)
            if edge_length < 8.0:  # Filter out edges longer than 8mm
                filtered_edges.append(edge)
    
    print(f"Filtered to {len(filtered_edges)} edges from {len(edges)} total edges")
    
    # Plot all filtered edges with uniform enhanced visibility
    plotted_count = 0
    for edge in filtered_edges[:20000]:  # Use more edges for better definition
        try:
            v1, v2 = edge
            if v1 < len(vertices) and v2 < len(vertices):
                p1, p2 = vertices[v1], vertices[v2]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                       color=color, alpha=alpha, linewidth=linewidth)
                plotted_count += 1
        except (IndexError, ValueError):
            continue
    
    print(f"Successfully plotted {plotted_count} wireframe edges")

def plot_landmark_points(ax, landmark_dict, color='gold', size=60, alpha=0.9):
    """Plot landmark points on the 3D axis."""
    landmark_array = np.array([point for point in landmark_dict.values()])
    ax.scatter(landmark_array[:, 0], landmark_array[:, 1], landmark_array[:, 2],
               c=color, s=size, alpha=alpha, marker='*', 
               edgecolors='black', linewidth=1.5, label='Anatomical Landmarks')

def regularize_points(points, epsilon=1e-6):
    return points + epsilon * np.random.randn(*points.shape)

def validate_points(points):
    if points.size == 0 or np.all(points == points[0]):
        return False
    return True

# --------------------------
# Processing Functions
# --------------------------
def point_acc_table(data_folder, mean=False, plot3D=False):
    files = sorted([f for f in os.listdir(data_folder) if f.endswith('.json')])
    results = {'FRE': {}, 'TRE': {}}
    fre_range, tre_range = (range(INDEX - 1), range(INDEX)) if INDEX == 10 else (range(INDEX + 1), range(INDEX)) if INDEX == 9 else (None, None)
    fre_errors_per_point = [[] for _ in fre_range]
    tre_errors_per_point = [[] for _ in tre_range]
    measured_poses = {key: [] for key in load_data(os.path.join(data_folder, files[0])).keys()}
    for file_name in files[0:19]:
        file_path = os.path.join(data_folder, file_name)
        data = load_data(file_path)
        for key, value in data.items():
            pose = get_pose(value, mean)
            if pose is None:
                print(f"Skipping {key} in {file_name} due to missing data.")
                continue
            measured_poses[key].append(pose[:3])
    for key in measured_poses.keys():
        measured_pose = np.array(measured_poses[key])
        if not validate_points(measured_pose):
            print(f"Invalid points detected for {key}. Skipping.")
            results['FRE'][key] = "N/A"
            results['TRE'][key] = "N/A"
            continue
        if np.isnan(measured_pose[10:]).any():
            nan_indices = np.where(np.isnan(measured_pose[INDEX:]).any(axis=1))[0]
            for i in nan_indices:
                print(f"All {key} poses for measured fiducial point {i+1} are NaN. Check the data file.")
            results['TRE'][key] = "N/A"
            continue
        with suppress_print():
            try:
                R, T, fre_rmse, fre_ppe = register(fixed=landmark_points, moving=measured_pose[INDEX:])
            except np.linalg.LinAlgError:
                print(f"Skipping {key} due to SVD convergence issue.")
                results['FRE'][key] = "N/A"
                results['TRE'][key] = "N/A"
                continue
        results['FRE'][key] = f"{fre_rmse:.4f}"
        transformed_points = apply_transform(measured_pose[:INDEX], R, T)
        if np.isnan(transformed_points).any():
            nan_indices = np.where(np.isnan(transformed_points).any(axis=1))[0]
            for i in nan_indices:
                print(f"All {key} poses for measured target point {i+1} are NaN. Check the data file.")
            results['TRE'][key] = "N/A"
            continue
        with suppress_print():
            _, tre_rmse, tre_ppe = calculate_TRE(fixed=target_points, moving=measured_pose[:INDEX], R=R, T=T)
        results['TRE'][key] = f"{tre_rmse:.4f}"
        for i, error in enumerate(fre_ppe):
            if i < len(fre_errors_per_point):
                fre_errors_per_point[i].append(error)
        for i, error in enumerate(tre_ppe):
            if i < len(tre_errors_per_point):
                tre_errors_per_point[i].append(error)
        if plot3D:
            transformed_fiducial_points = apply_transform(measured_pose[INDEX:], R, T)
            plot_3d_points(landmark_points, measured_pose[INDEX:], transformed_fiducial_points, target_points, f"Registration Plot for {key}")
    
    avg_fre_ppe = [np.median(errors) if errors else 0 for errors in fre_errors_per_point]
    std_fre_ppe = [np.std(errors) if errors else 0 for errors in fre_errors_per_point]
    avg_tre_ppe = [np.median(errors) if errors else 0 for errors in tre_errors_per_point]
    std_tre_ppe = [np.std(errors) if errors else 0 for errors in tre_errors_per_point]
    print("\nMedian Per-Point FRE Across All Configs [mm]:")
    for i, (avg_error, std_error) in enumerate(zip(avg_fre_ppe, std_fre_ppe)):
        print(f"Point {i+1}: {avg_error:.4f} (±{std_error:.4f})")
    print("\nMedian Per-Point TRE Across All Configs [mm]:")
    for i, (avg_error, std_error) in enumerate(zip(avg_tre_ppe, std_tre_ppe)):
        print(f"Point {i+1}: {avg_error:.4f} (±{std_error:.4f})")
    camera_configs = sorted(set(key.split(' Cam ')[0] for key in results['FRE'].keys()))
    fusion_methods = sorted(set(key.split(' Cam ')[1] for key in results['FRE'].keys()))
    fre_df = pd.DataFrame(index=camera_configs, columns=fusion_methods)
    tre_df = pd.DataFrame(index=camera_configs, columns=fusion_methods)
    for config in camera_configs:
        for method in fusion_methods:
            key = f"{config} Cam {method}"
            fre_df.at[config, method] = results['FRE'].get(key, "N/A")
            tre_df.at[config, method] = results['TRE'].get(key, "N/A")
    fre_df.columns.name = 'Cams'
    tre_df.columns.name = 'Cams'
    print("\nFRE RMSE Results [mm]:")
    print(fre_df.to_string(index=True, formatters={'float': '{:.4f}'.format}, justify='center'))
    print("\nTRE RMSE Results [mm]:")
    print(tre_df.to_string(index=True, formatters={'float': '{:.4f}'.format}, justify='center'))
    if MEAN:
        with open(f'{data_folder}/{os.path.basename(data_folder)}_results_mean.json', 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to {data_folder}/{os.path.basename(data_folder)}_results_mean.json")
    else:
        with open(f'{data_folder}/{os.path.basename(data_folder)}_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to {data_folder}/{os.path.basename(data_folder)}_results.json")

def results_table(data_folder, points_FRE, points_TRE, mean=False, plot3D=False, savePlots=True):
    """
    Calculates and displays registration results, and optionally plots 3D points and trajectories.

    Parameters:
    - data_folder (str): The directory containing the data.
    - points_FRE (list): List of point identifiers used for Fiducial Registration Error (FRE) calculation.
    - points_TRE (list): List of point identifiers used for Target Registration Error (TRE) calculation.
    - mean (bool): If True, uses the mean of the poses; otherwise, uses the last pose.
    - plot3D (bool): If True, generates 3D plots.
    - savePlots (bool): If True, saves the plots as figures; if False, displays them interactively.
    """
    # Collect all JSON files in the data folder, excluding ones that start with '_'
    files = {f.split('_')[1]: f for f in os.listdir(data_folder) if f.endswith('.json') and not f.startswith('_')}
    frontTraj = 'RF' in files
    backTraj = 'RB' in files
    results = {'FRE': {}, 'TRE': {}, 'Front Trajectory TRE': {}, 'Back Trajectory TRE': {}, 'RT1 Trajectory TRE': {}, 'RT2 Trajectory TRE': {}, 'RT3 Trajectory TRE': {}, 'RT4 Trajectory TRE': {}}
    coords_FRE = np.array([landmark_points[point] for point in points_FRE])
    coords_TRE = np.array([landmark_points[point] for point in points_TRE])
    
    # Initialization of error accumulation lists
    fre_errors_per_point = [[] for _ in range(len(points_FRE))]
    tre_errors_per_point = [[] for _ in range(len(points_TRE))]
    traj_errors = [[] for _ in range(len(TRAJECTORIES))]

    # Initialize measured poses dictionary
    sample_file = next(iter(files.values()))
    sample_data = load_data(os.path.join(data_folder, sample_file))
    measured_poses = {key: {} for key in sample_data.keys()}

    # Dictionary to store transformed trajectories by key
    transformed_trajectories = {}

    # Process each point file
    for point, file_name in files.items():
        if point in ['RF', 'RB', 'RT1', 'RT2', 'RT3', 'RT4']:
            continue  # Skip trajectory files for now
        file_path = os.path.join(data_folder, file_name)
        data = load_data(file_path)
        
        for key, value in data.items():
            pose = get_pose(value, mean)
            if pose is None:
                print(f"Skipping {key} in {file_name} due to missing data.")
                continue
            measured_poses[key][point] = pose[:3]

            # Ensure the key is initialized in transformed_trajectories
            if key not in transformed_trajectories:
                transformed_trajectories[key] = {
                    'fiducial': None,
                    'target': None,
                    'front_trajectory': None,
                    'back_trajectory': None,
                    'rt1_trajectory': None,
                    'rt2_trajectory': None,
                    'rt3_trajectory': None,
                    'rt4_trajectory': None,
                    'front_untracked_percentage': 0,  # Initialize percentages to 0
                    'back_untracked_percentage': 0,
                    'rt1_untracked_percentage': 0,
                    'rt2_untracked_percentage': 0,
                    'rt3_untracked_percentage': 0,
                    'rt4_untracked_percentage': 0
                }

    for key in measured_poses.keys():
        # Extract measured poses for FRE and TRE points
        try:
            measured_pose_FRE = np.array([measured_poses[key][point] for point in points_FRE])
            measured_pose_TRE = np.array([measured_poses[key][point] for point in points_TRE])
        except KeyError as e:
            print(f"Missing data for point {e} in key {key}. Skipping.")
            results['FRE'][key] = "N/A"
            results['TRE'][key] = "N/A"
            continue

        if not validate_points(np.vstack([measured_pose_FRE, measured_pose_TRE])):
            print(f"Invalid points detected for {key}. Skipping.")
            results['FRE'][key] = "N/A"
            results['TRE'][key] = "N/A"
            continue

        # Perform fiducial registration for the current configuration and fusion method
        with suppress_print():
            try:
                R, T, fre_rmse, fre_ppe = register(fixed=coords_FRE, moving=measured_pose_FRE)
            except np.linalg.LinAlgError:
                print(f"Skipping {key} due to SVD convergence issue.")
                results['FRE'][key] = "N/A"
                results['TRE'][key] = "N/A"
                continue

        results['FRE'][key] = f"{fre_rmse:.4f}"

        # Apply the transformation to the target points
        transformed_points = apply_transform(measured_pose_TRE, R, T)

        # Calculate TRE
        with suppress_print():
            _, tre_rmse, tre_ppe = calculate_TRE(fixed=coords_TRE, moving=measured_pose_TRE, R=R, T=T)
        results['TRE'][key] = f"{tre_rmse:.4f}"

        # Store the fiducial and target points for later plotting
        transformed_trajectories[key]['fiducial'] = (coords_FRE, measured_pose_FRE, apply_transform(measured_pose_FRE, R, T))
        transformed_trajectories[key]['target'] = (coords_TRE, measured_pose_TRE, transformed_points)

        # Initialize a list to accumulate distances from all Front/Back or RT trajectories for this key
        combined_rt_distances = []

        # Load and transform the trajectories using the obtained transformation
        if backTraj:
            # Process Back Trajectory Data
            back_trajectory_data = load_data(os.path.join(data_folder, files['RB']))
            back_data = back_trajectory_data.get(key, {})
            back_poses = np.array(back_data.get('poses', []))

            if back_poses.size == 0:
                results['Back Trajectory TRE'][key] = "N/A"
                continue

            back_measured_pose = np.squeeze(back_poses[:, :3])

            # Calculate untracked percentage and filter NaN
            back_untracked_percentage = np.isnan(back_measured_pose).any(axis=1).mean() * 100
            back_measured_pose = back_measured_pose[~np.isnan(back_measured_pose).any(axis=1)]

            if back_measured_pose.size == 0:
                results['Back Trajectory TRE'][key] = "N/A"
                transformed_trajectories[key]['back_untracked_percentage'] = back_untracked_percentage
                if back_untracked_percentage > 0:
                    print(f"{key} Back Trajectory Untracked: {back_untracked_percentage:.2f}%")
                continue

            transformed_back_trajectory = apply_transform(back_measured_pose, R, T)
            transformed_trajectories[key]['back_trajectory'] = transformed_back_trajectory
            transformed_trajectories[key]['back_untracked_percentage'] = back_untracked_percentage
            back_tree = cKDTree(landmark_back_trajectory)
            back_distances, _ = back_tree.query(transformed_back_trajectory)
            back_tre_rmse = np.sqrt(np.mean(back_distances**2))
            results['Back Trajectory TRE'][key] = f"{back_tre_rmse:.4f}"
            if back_untracked_percentage > 0:
                print(f"{key} Back Trajectory Untracked: {back_untracked_percentage:.2f}%")
            combined_rt_distances.append(back_distances)
            traj_errors[0].append(back_tre_rmse)

        if frontTraj:
            # Process Front Trajectory Data
            front_trajectory_data = load_data(os.path.join(data_folder, files['RF']))
            front_data = front_trajectory_data.get(key, {})
            front_poses = np.array(front_data.get('poses', []))

            if front_poses.size == 0:
                results['Front Trajectory TRE'][key] = "N/A"
                continue

            front_measured_pose = np.squeeze(front_poses[:, :3])

            # Calculate untracked percentage and filter NaN
            front_untracked_percentage = np.isnan(front_measured_pose).any(axis=1).mean() * 100
            front_measured_pose = front_measured_pose[~np.isnan(front_measured_pose).any(axis=1)]

            if front_measured_pose.size == 0:
                results['Front Trajectory TRE'][key] = "N/A"
                transformed_trajectories[key]['front_untracked_percentage'] = front_untracked_percentage
                if front_untracked_percentage > 0:
                    print(f"{key} Front Trajectory Points Untracked: {front_untracked_percentage:.2f}%")
                continue

            transformed_front_trajectory = apply_transform(front_measured_pose, R, T)
            transformed_trajectories[key]['front_trajectory'] = transformed_front_trajectory
            transformed_trajectories[key]['front_untracked_percentage'] = front_untracked_percentage

            front_tree = cKDTree(landmark_front_trajectory)
            front_distances, _ = front_tree.query(transformed_front_trajectory)
            front_tre_rmse = np.sqrt(np.mean(front_distances**2))
            results['Front Trajectory TRE'][key] = f"{front_tre_rmse:.4f}"
            if front_untracked_percentage > 0:
                print(f"{key} Front Trajectory Points Untracked: {front_untracked_percentage:.2f}%")
            combined_rt_distances.append(front_distances)
                
            traj_errors[1].append(front_tre_rmse)

        

        # Process RT1 Trajectory Data
        if 'RT1' in files:
            rt1_trajectory_data = load_data(os.path.join(data_folder, files['RT1']))
            rt1_data = rt1_trajectory_data.get(key, {})
            rt1_poses = np.array(rt1_data.get('poses', []))
            if rt1_poses.size == 0:
                results['RT1 Trajectory TRE'][key] = "N/A"
            else:
                rt1_measured_pose = np.squeeze(rt1_poses[:, :3])
                rt1_untracked_percentage = np.isnan(rt1_measured_pose).any(axis=1).mean() * 100
                rt1_measured_pose = rt1_measured_pose[~np.isnan(rt1_measured_pose).any(axis=1)]
                if rt1_measured_pose.size == 0:
                    results['RT1 Trajectory TRE'][key] = "N/A"
                    transformed_trajectories[key]['rt1_untracked_percentage'] = rt1_untracked_percentage
                else:
                    transformed_rt1_trajectory = apply_transform(rt1_measured_pose, R, T)
                    transformed_trajectories[key]['rt1_trajectory'] = transformed_rt1_trajectory
                    transformed_trajectories[key]['rt1_untracked_percentage'] = rt1_untracked_percentage
                    rt1_tree = cKDTree(trajectory1)
                    rt1_distances, _ = rt1_tree.query(transformed_rt1_trajectory)
                    rt1_tre_rmse = np.sqrt(np.mean(rt1_distances**2))
                    results['RT1 Trajectory TRE'][key] = f"{rt1_tre_rmse:.4f}"
                    if rt1_untracked_percentage > 0:
                        print(f"{key} RT1 Trajectory Points Untracked: {rt1_untracked_percentage:.2f}%")
                    # Append the distances to the combined list
                    combined_rt_distances.append(rt1_distances)
                    traj_errors[0].append(rt1_tre_rmse)
        else:
            results['RT1 Trajectory TRE'][key] = "N/A"

        # Process RT2 Trajectory Data
        if 'RT2' in files:
            rt2_trajectory_data = load_data(os.path.join(data_folder, files['RT2']))
            rt2_data = rt2_trajectory_data.get(key, {})
            rt2_poses = np.array(rt2_data.get('poses', []))

            if rt2_poses.size == 0:
                results['RT2 Trajectory TRE'][key] = "N/A"
                continue

            rt2_measured_pose = np.squeeze(rt2_poses[:, :3])

            # Calculate untracked percentage and filter NaN
            rt2_untracked_percentage = np.isnan(rt2_measured_pose).any(axis=1).mean() * 100
            rt2_measured_pose = rt2_measured_pose[~np.isnan(rt2_measured_pose).any(axis=1)]

            if rt2_measured_pose.size == 0:
                results['RT2 Trajectory TRE'][key] = "N/A"
                transformed_trajectories[key]['rt2_untracked_percentage'] = rt2_untracked_percentage
                if rt2_untracked_percentage > 0:
                    print(f"{key} RT2 Trajectory Points Untracked: {rt2_untracked_percentage:.2f}%")
                continue

            transformed_rt2_trajectory = apply_transform(rt2_measured_pose, R, T)
            transformed_trajectories[key]['rt2_trajectory'] = transformed_rt2_trajectory
            transformed_trajectories[key]['rt2_untracked_percentage'] = rt2_untracked_percentage

            rt2_tree = cKDTree(trajectory2)
            rt2_distances, _ = rt2_tree.query(transformed_rt2_trajectory)
            rt2_tre_rmse = np.sqrt(np.mean(rt2_distances**2))
            results['RT2 Trajectory TRE'][key] = f"{rt2_tre_rmse:.4f}"
            if rt2_untracked_percentage > 0:
                print(f"{key} RT2 Trajectory Points Untracked: {rt2_untracked_percentage:.2f}%")
            combined_rt_distances.append(rt2_distances)
            traj_errors[1].append(rt2_tre_rmse)

        # Process RT3 Trajectory Data
        if 'RT3' in files:
            rt3_trajectory_data = load_data(os.path.join(data_folder, files['RT3']))
            rt3_data = rt3_trajectory_data.get(key, {})
            rt3_poses = np.array(rt3_data.get('poses', []))

            if rt3_poses.size == 0:
                results['RT3 Trajectory TRE'][key] = "N/A"
                continue

            rt3_measured_pose = np.squeeze(rt3_poses[:, :3])

            # Calculate untracked percentage and filter NaN
            rt3_untracked_percentage = np.isnan(rt3_measured_pose).any(axis=1).mean() * 100
            rt3_measured_pose = rt3_measured_pose[~np.isnan(rt3_measured_pose).any(axis=1)]

            if rt3_measured_pose.size == 0:
                results['RT3 Trajectory TRE'][key] = "N/A"
                transformed_trajectories[key]['rt3_untracked_percentage'] = rt3_untracked_percentage
                if rt3_untracked_percentage > 0:
                    print(f"{key} RT3 Trajectory Points Untracked: {rt3_untracked_percentage:.2f}%")
                continue

            transformed_rt3_trajectory = apply_transform(rt3_measured_pose, R, T)
            transformed_trajectories[key]['rt3_trajectory'] = transformed_rt3_trajectory
            transformed_trajectories[key]['rt3_untracked_percentage'] = rt3_untracked_percentage

            rt3_tree = cKDTree(trajectory3)
            rt3_distances, _ = rt3_tree.query(transformed_rt3_trajectory)
            rt3_tre_rmse = np.sqrt(np.mean(rt3_distances**2))
            results['RT3 Trajectory TRE'][key] = f"{rt3_tre_rmse:.4f}"
            if rt3_untracked_percentage > 0:
                print(f"{key} RT3 Trajectory Points Untracked: {rt3_untracked_percentage:.2f}%")
            combined_rt_distances.append(rt3_distances)
            traj_errors[2].append(rt3_tre_rmse)

        # Process RT4 Trajectory Data
        if 'RT4' in files:
            rt4_trajectory_data = load_data(os.path.join(data_folder, files['RT4']))
            rt4_data = rt4_trajectory_data.get(key, {})
            rt4_poses = np.array(rt4_data.get('poses', []))

            if rt4_poses.size == 0:
                results['RT4 Trajectory TRE'][key] = "N/A"
                continue

            rt4_measured_pose = np.squeeze(rt4_poses[:, :3])

            # Calculate untracked percentage and filter NaN
            rt4_untracked_percentage = np.isnan(rt4_measured_pose).any(axis=1).mean() * 100
            rt4_measured_pose = rt4_measured_pose[~np.isnan(rt4_measured_pose).any(axis=1)]

            if rt4_measured_pose.size == 0:
                results['RT4 Trajectory TRE'][key] = "N/A"
                transformed_trajectories[key]['rt4_untracked_percentage'] = rt4_untracked_percentage
                if rt4_untracked_percentage > 0:
                    print(f"{key} RT4 Trajectory Points Untracked: {rt4_untracked_percentage:.2f}%")
                continue

            transformed_rt4_trajectory = apply_transform(rt4_measured_pose, R, T)
            transformed_trajectories[key]['rt4_trajectory'] = transformed_rt4_trajectory
            transformed_trajectories[key]['rt4_untracked_percentage'] = rt4_untracked_percentage

            rt4_tree = cKDTree(trajectory4)
            rt4_distances, _ = rt4_tree.query(transformed_rt4_trajectory)
            rt4_tre_rmse = np.sqrt(np.mean(rt4_distances**2))
            results['RT4 Trajectory TRE'][key] = f"{rt4_tre_rmse:.4f}"
            if rt4_untracked_percentage > 0:
                print(f"{key} RT4 Trajectory Points Untracked: {rt4_untracked_percentage:.2f}%")
            combined_rt_distances.append(rt4_distances)
            traj_errors[3].append(rt4_tre_rmse)
            
        # Compute Total Trajectory TRE from all RT distances
        if combined_rt_distances:
            all_distances = np.concatenate(combined_rt_distances)
            total_tre_rmse = np.sqrt(np.mean(all_distances**2))
            # Store total trajectory TRE for this key
            if 'Total Trajectory TRE' not in results:
                results['Total Trajectory TRE'] = {}
            results['Total Trajectory TRE'][key] = f"{total_tre_rmse:.4f}"
        else:
            if 'Total Trajectory TRE' not in results:
                results['Total Trajectory TRE'] = {}
            results['Total Trajectory TRE'][key] = "N/A"


        # Accumulate per-point errors for FRE and TRE
        for i, error in enumerate(fre_ppe):
            if i < len(fre_errors_per_point):
                fre_errors_per_point[i].append(error)
        for i, error in enumerate(tre_ppe):
            if i < len(tre_errors_per_point):
                tre_errors_per_point[i].append(error)

    # Calculate median per-point errors
    avg_fre_ppe = [np.median(errors) if errors else 0 for errors in fre_errors_per_point]
    std_fre_ppe = [np.std(errors) if errors else 0 for errors in fre_errors_per_point]
    avg_tre_ppe = [np.median(errors) if errors else 0 for errors in tre_errors_per_point]
    std_tre_ppe = [np.std(errors) if errors else 0 for errors in tre_errors_per_point]

    print("\nMedian Per-Point FRE Across All Configs [mm]:")
    for i, (avg_error, std_error) in enumerate(zip(avg_fre_ppe, std_fre_ppe)):
        print(f"Point {points_FRE[i]}: {avg_error:.4f} (±{std_error:.4f})")

    print("\nMedian Per-Point TRE Across All Configs [mm]:")
    for i, (avg_error, std_error) in enumerate(zip(avg_tre_ppe, std_tre_ppe)):
        print(f"Point {points_TRE[i]}: {avg_error:.4f} (±{std_error:.4f})")

    # Collect results into DataFrames
    camera_configs = sorted(set(key.split(' Cam ')[0] for key in results['FRE'].keys()))
    # fusion_methods = sorted(set(key.split(' Cam ')[1] for key in results['FRE'].keys()))

    fre_df = pd.DataFrame(index=camera_configs, columns=fusion_methods)
    tre_df = pd.DataFrame(index=camera_configs, columns=fusion_methods)
    if backTraj: 
        back_traj_tre_df = pd.DataFrame(index=camera_configs, columns=fusion_methods)
    if frontTraj: 
        front_traj_tre_df = pd.DataFrame(index=camera_configs, columns=fusion_methods)
    if 'RT1' in files: 
        rt1_traj_tre_df = pd.DataFrame(index=camera_configs, columns=fusion_methods)
    if 'RT2' in files: 
        rt2_traj_tre_df = pd.DataFrame(index=camera_configs, columns=fusion_methods)
    if 'RT3' in files: 
        rt3_traj_tre_df = pd.DataFrame(index=camera_configs, columns=fusion_methods)
    if 'RT4' in files: 
        rt4_traj_tre_df = pd.DataFrame(index=camera_configs, columns=fusion_methods)
    if 'Total Trajectory TRE' in results:
        total_traj_tre_df = pd.DataFrame(index=camera_configs, columns=fusion_methods)

    for config in camera_configs:
        for method in fusion_methods:
            key = f"{config} Cam {method}"
            fre_df.at[config, method] = results['FRE'].get(key, "N/A")
            tre_df.at[config, method] = results['TRE'].get(key, "N/A")
            if backTraj:
                back_traj_tre_df.at[config, method] = results['Back Trajectory TRE'].get(key, "N/A")
            if frontTraj:
                front_traj_tre_df.at[config, method] = results['Front Trajectory TRE'].get(key, "N/A")
            if 'RT1' in files:
                rt1_traj_tre_df.at[config, method] = results['RT1 Trajectory TRE'].get(key, "N/A")
            if 'RT2' in files:
                rt2_traj_tre_df.at[config, method] = results['RT2 Trajectory TRE'].get(key, "N/A")
            if 'RT3' in files:
                rt3_traj_tre_df.at[config, method] = results['RT3 Trajectory TRE'].get(key, "N/A")
            if 'RT4' in files:
                rt4_traj_tre_df.at[config, method] = results['RT4 Trajectory TRE'].get(key, "N/A")
            if 'Total Trajectory TRE' in results:
                total_traj_tre_df.at[config, method] = results['Total Trajectory TRE'].get(key, "N/A")

    # Set column header names
    fre_df.columns.name = 'Cams'
    tre_df.columns.name = 'Cams'
    if backTraj: back_traj_tre_df.columns.name = 'Cams'
    if frontTraj: front_traj_tre_df.columns.name = 'Cams'
    if 'RT1' in files: rt1_traj_tre_df.columns.name = 'Cams'
    if 'RT2' in files: rt2_traj_tre_df.columns.name = 'Cams'
    if 'RT3' in files: rt3_traj_tre_df.columns.name = 'Cams'
    if 'RT4' in files: rt4_traj_tre_df.columns.name = 'Cams'
    if 'Total Trajectory TRE' in results: total_traj_tre_df.columns.name = 'Cams'

    # Print DataFrames
    print("\nFRE RMSE Results [mm]:")
    print(fre_df.to_string(index=True, justify='center'))

    print("\nTRE RMSE Results [mm]:")
    print(tre_df.to_string(index=True, justify='center'))

    if frontTraj:
        print("\nFront Trajectory TRE RMSE Results [mm]:")
        print(front_traj_tre_df.to_string(index=True, justify='center'))

    if backTraj:
        print("\nBack Trajectory TRE RMSE Results [mm]:")
        print(back_traj_tre_df.to_string(index=True, justify='center'))

    if 'RT1' in files:
        print("\nRT1 Trajectory TRE RMSE Results [mm]:")
        print(rt1_traj_tre_df.to_string(index=True, justify='center'))

    if 'RT2' in files:
        print("\nRT2 Trajectory TRE RMSE Results [mm]:")
        print(rt2_traj_tre_df.to_string(index=True, justify='center'))

    if 'RT3' in files:
        print("\nRT3 Trajectory TRE RMSE Results [mm]:")
        print(rt3_traj_tre_df.to_string(index=True, justify='center'))

    if 'RT4' in files:
        print("\nRT4 Trajectory TRE RMSE Results [mm]:")
        print(rt4_traj_tre_df.to_string(index=True, justify='center'))

    if 'Total Trajectory TRE' in results:
        print("\nTotal Trajectory TRE RMSE [mm]:")
        print(total_traj_tre_df.to_string(index=True, justify='center'))

    # Save results to JSON file
    results_path = os.path.join(data_folder, f"{os.path.basename(data_folder)}_results.json")
    jsonDict = {'FRE': fre_df.to_dict(), 'TRE': tre_df.to_dict()}
    if backTraj: jsonDict['Back Trajectory TRE'] = back_traj_tre_df.to_dict()
    if frontTraj: jsonDict['Front Trajectory TRE'] = front_traj_tre_df.to_dict()
    if 'RT1' in files: jsonDict['RT1 Trajectory TRE'] = rt1_traj_tre_df.to_dict()
    if 'RT2' in files: jsonDict['RT2 Trajectory TRE'] = rt2_traj_tre_df.to_dict()
    if 'RT3' in files: jsonDict['RT3 Trajectory TRE'] = rt3_traj_tre_df.to_dict()
    if 'RT4' in files: jsonDict['RT4 Trajectory TRE'] = rt4_traj_tre_df.to_dict()
    if 'Total Trajectory TRE' in results: jsonDict['Total Trajectory TRE'] = total_traj_tre_df.to_dict()
    
    with open(results_path, 'w') as json_file:
        json.dump(jsonDict, json_file, indent=4)

    print(f"\nResults saved to {results_path}")

    # Create a directory to save plots if saving is enabled
    if savePlots:
        save_folder = os.path.join(data_folder, "plots")
        os.makedirs(save_folder, exist_ok=True)

    # Plot everything together at the end, but separately for each key
    if plot3D:
        for key, trajectories in transformed_trajectories.items():
            if trajectories['fiducial'] is not None:
                try:
                    # Unpack the fiducial data
                    landmarks, measured_fiducials, transformed_fiducials = trajectories['fiducial']
                    targets, measured_targets, transformed_targets = trajectories['target']

                    # Plotting the fiducial and target points
                    fig = plot_3d_points(
                        real_fiducial_points=landmarks,
                        transformed_fiducial_points=transformed_fiducials,
                        real_target_points=targets,
                        transformed_target_points=transformed_targets,
                        title=f"Point Registration for {key}"
                    )

                    # Decide whether to save or show the plot
                    if savePlots:
                        # Save the point registration plot
                        fig.savefig(os.path.join(save_folder, f"{key}_points_plot.png"))
                        plt.close(fig)
                    else:
                        plt.show(fig)

                    # Plot trajectories with untracked percentages
                    back_untracked = trajectories.get('back_untracked_percentage', 0)
                    front_untracked = trajectories.get('front_untracked_percentage', 0)
                    rt1_untracked = trajectories.get('rt1_untracked_percentage', 0)
                    rt2_untracked = trajectories.get('rt2_untracked_percentage', 0)
                    rt3_untracked = trajectories.get('rt3_untracked_percentage', 0)
                    rt4_untracked = trajectories.get('rt4_untracked_percentage', 0)
                    title = get_display_key(key)
                    if back_untracked > 0 or front_untracked > 0 or rt1_untracked > 0 or rt2_untracked > 0 or rt3_untracked > 0 or rt4_untracked > 0:
                        title += f"\nBack Points Untracked: {back_untracked:.2f}%, Front Points Untracked: {front_untracked:.2f}%, RT1 Points Untracked: {rt1_untracked:.2f}%, RT2 Points Untracked: {rt2_untracked:.2f}%, RT3 Points Untracked: {rt3_untracked:.2f}%, RT4 Points Untracked: {rt4_untracked:.2f}%"

                    # Prepare trajectories for plotting based on experiment type
                    if GRID:
                        real_trajectories = [trajectory1, trajectory2, trajectory3, trajectory4]
                        measured_trajectories = [trajectories.get('rt1_trajectory'), trajectories.get('rt2_trajectory'), trajectories.get('rt3_trajectory'), trajectories.get('rt4_trajectory')]
                    else:
                        real_trajectories = [landmark_back_trajectory, landmark_front_trajectory]
                        measured_trajectories = [trajectories.get('back_trajectory'), trajectories.get('front_trajectory')]

                    # Handle cases where trajectories might be None
                    for i in range(len(measured_trajectories)):
                        if measured_trajectories[i] is None:
                            measured_trajectories[i] = np.empty((0, 3))

                    fig = plot_3d_trajectories(
                        real_trajectories,
                        measured_trajectories,
                        title  # Title for the plot
                    )

                    # Decide whether to save or show the plot
                    if savePlots:
                        # Save the trajectory plot
                        fig.savefig(os.path.join(save_folder, f"{key}_trajectory_plot.png"))
                        plt.close(fig)
                    else:
                        plt.show(fig)

                    # Generate combined plot showing both trajectories and registration points
                    combined_title = f"Combined Analysis: {get_display_key(key)}"
                    fig_combined = plot_3d_combined(
                        real_trajectories=real_trajectories,
                        measured_trajectories=measured_trajectories,
                        real_fiducial_points=landmarks,
                        transformed_fiducial_points=transformed_fiducials,
                        real_target_points=targets,
                        transformed_target_points=transformed_targets,
                        title=combined_title
                    )
                    
                    # Decide whether to save or show the combined plot
                    if savePlots:
                        # Save the combined plot
                        fig_combined.savefig(os.path.join(save_folder, f"{key}_combined_plot.png"))
                        plt.close(fig_combined)
                    else:
                        plt.show(fig_combined)

                except Exception as e:
                    print(f"Error unpacking or plotting data for key {key}: {e}")
            else:
                print(f"Skipping key {key} due to missing data.")
    
    return fre_errors_per_point, tre_errors_per_point, traj_errors, measured_poses.keys()

def read_last_pose_from_txt(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    last_pose = np.array([float(x) for x in lines[-1].strip().split()])
    return last_pose

def read_poses_from_txt(file_path):
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            pose = np.array([float(x) for x in line.strip().split()])
            poses.append(pose)
    return np.array(poses)

def results_atracsys(data_folder, points_FRE, points_TRE, mean=False, plot3D=False):
    fre_ppe = []
    tre_ppe = []
    default_pose = [1.669328460693359375e+02, 1.925300478935241699e+00, -2.327872467041015625e+01]
    trial_name = os.path.basename(data_folder).split('_')[0]
    results = {
        'FRE': {}, 'TRE': {}, 
        'Front Trajectory TRE': {}, 'Back Trajectory TRE': {}, 
        'RT1 Trajectory TRE': {}, 'RT2 Trajectory TRE': {}, 
        'RT3 Trajectory TRE': {}, 'RT4 Trajectory TRE': {},
        'Total Trajectory TRE': {}
    }
    
    coords_FRE = np.array([landmark_points[point] for point in points_FRE])
    coords_TRE = np.array([landmark_points[point] for point in points_TRE])
    
    target_poses = []
    fiducial_poses = []
    
    traj_errors = [[] for _ in range(len(TRAJECTORIES))]
    
    # Target points
    for point in points_TRE:
        subfolder = f"{trial_name}_{point}"
        folder_path = os.path.join(data_folder, subfolder)
        igtpos_file = os.path.join(folder_path, 'IGTPos.txt')
        if not os.path.exists(igtpos_file):
            print(f"Missing IGTPos.txt in {subfolder}. Skipping.")
            continue
        if mean:
            target_poses.append(np.mean(read_poses_from_txt(igtpos_file), axis=0))
        else:
            target_poses.append(read_last_pose_from_txt(igtpos_file))
    
    # Fiducial points
    for point in points_FRE:
        subfolder = f"{trial_name}_{point}"
        folder_path = os.path.join(data_folder, subfolder)
        igtpos_file = os.path.join(folder_path, 'IGTPos.txt')
        if not os.path.exists(igtpos_file):
            print(f"Missing IGTPos.txt in {subfolder}. Skipping.")
            continue
        if mean:
            fiducial_poses.append(np.mean(read_poses_from_txt(igtpos_file), axis=0))
        else:
            fiducial_poses.append(read_last_pose_from_txt(igtpos_file))
    
    # Convert lists to numpy arrays
    target_poses = np.array(target_poses) if target_poses else np.array([])
    fiducial_poses = np.array(fiducial_poses) if fiducial_poses else np.array([])

    # Initialize transformed_trajectories
    transformed_trajectories = {
        'fiducial': None,
        'target': None,
        'back_trajectory': None,
        'front_trajectory': None,
        'rt1_trajectory': None,
        'rt2_trajectory': None,
        'rt3_trajectory': None,
        'rt4_trajectory': None,
        'back_untracked_percentage': 0,
        'front_untracked_percentage': 0,
        'rt1_untracked_percentage': 0,
        'rt2_untracked_percentage': 0,
        'rt3_untracked_percentage': 0,
        'rt4_untracked_percentage': 0
    }

    if fiducial_poses.size > 0 and target_poses.size > 0:
        # Perform registration and calculate FRE/TRE
        with suppress_print():
            try:
                R, T, fre_rmse, fre_ppe = register(fixed=coords_FRE, moving=fiducial_poses)
                results['FRE'] = f"{fre_rmse:.4f}"
            except np.linalg.LinAlgError:
                print(f"Skipping due to SVD convergence issue.")
                results['FRE'] = "N/A"
                results['TRE'] = "N/A"
                fre_ppe = []
                tre_ppe = []
                return fre_ppe, tre_ppe

        transformed_points = apply_transform(target_poses, R, T)
        
        with suppress_print():
            _, tre_rmse, tre_ppe = calculate_TRE(fixed=coords_TRE, moving=target_poses, R=R, T=T)
        results['TRE'] = f"{tre_rmse:.4f}"

        # Store the transformed trajectories for plotting later
        transformed_trajectories['fiducial'] = {
            'real': coords_FRE,
            'measured': fiducial_poses,
            'transformed': apply_transform(fiducial_poses, R, T)
        }
        transformed_trajectories['target'] = {
            'real': coords_TRE,
            'measured': target_poses,
            'transformed': transformed_points
        }
    else:
        print("No fiducial or target poses available.")
        results['FRE'] = "N/A"
        results['TRE'] = "N/A"
        fre_ppe = []
        tre_ppe = []
        return fre_ppe, tre_ppe

    # Process Back Trajectory Data
    back_trajectory_folder = os.path.join(data_folder, f'{trial_name}_RB')
    back_igtpos_file = os.path.join(back_trajectory_folder, 'IGTPos.txt')
    if os.path.exists(back_igtpos_file):
        back_measured_pose = read_poses_from_txt(back_igtpos_file)
        # Filter out the default pose
        valid_back_poses = []
        for pose in back_measured_pose:
            if not np.allclose(pose, default_pose) and not np.isnan(pose).any():
                valid_back_poses.append(pose)
        back_measured_pose = np.array(valid_back_poses)
        back_untracked_percentage = np.isnan(back_measured_pose).any(axis=1).mean() * 100
        back_measured_pose = back_measured_pose[~np.isnan(back_measured_pose).any(axis=1)]

        if back_measured_pose.size > 0:
            transformed_back_trajectory = apply_transform(back_measured_pose, R, T)
            back_tree = cKDTree(landmark_back_trajectory)
            back_distances, _ = back_tree.query(transformed_back_trajectory)
            back_tre_rmse = np.sqrt(np.mean(back_distances**2))
            results['Back Trajectory TRE'] = f"{back_tre_rmse:.4f}"
            transformed_trajectories['back_trajectory'] = {
                'real': landmark_back_trajectory,
                'measured': back_measured_pose,
                'transformed': transformed_back_trajectory
            }
            transformed_trajectories['back_untracked_percentage'] = back_untracked_percentage

            if back_untracked_percentage > 0:
                print(f"Back Trajectory Untracked: {back_untracked_percentage:.2f}%")
                
            traj_errors[0].append(back_tre_rmse)
        else:
            results['Back Trajectory TRE'] = "N/A"
            transformed_trajectories['back_untracked_percentage'] = back_untracked_percentage
    else:
        results['Back Trajectory TRE'] = "N/A"

    # Process Front Trajectory Data
    front_trajectory_folder = os.path.join(data_folder, f'{trial_name}_RF')
    front_igtpos_file = os.path.join(front_trajectory_folder, 'IGTPos.txt')
    if os.path.exists(front_igtpos_file):
        front_measured_pose = read_poses_from_txt(front_igtpos_file)
        # Filter out the default pose
        valid_front_poses = []
        for pose in front_measured_pose:
            if not np.allclose(pose, default_pose) and not np.isnan(pose).any():
                valid_front_poses.append(pose)
        front_measured_pose = np.array(valid_front_poses)
        front_untracked_percentage = np.isnan(front_measured_pose).any(axis=1).mean() * 100
        front_measured_pose = front_measured_pose[~np.isnan(front_measured_pose).any(axis=1)]

        if front_measured_pose.size > 0:
            transformed_front_trajectory = apply_transform(front_measured_pose, R, T)
            front_tree = cKDTree(landmark_front_trajectory)
            front_distances, _ = front_tree.query(transformed_front_trajectory)
            front_tre_rmse = np.sqrt(np.mean(front_distances**2))
            results['Front Trajectory TRE'] = f"{front_tre_rmse:.4f}"
            transformed_trajectories['front_trajectory'] = {
                'real': landmark_front_trajectory,
                'measured': front_measured_pose,
                'transformed': transformed_front_trajectory
            }
            transformed_trajectories['front_untracked_percentage'] = front_untracked_percentage

            if front_untracked_percentage > 0:
                print(f"Front Trajectory Points Untracked: {front_untracked_percentage:.2f}%")
                
            traj_errors[1].append(front_tre_rmse)
        else:
            results['Front Trajectory TRE'] = "N/A"
            transformed_trajectories['front_untracked_percentage'] = front_untracked_percentage
    else:
        results['Front Trajectory TRE'] = "N/A"

    # Process RT1 Trajectory Data
    rt1_trajectory_folder = os.path.join(data_folder, f'{trial_name}_RT1')
    rt1_igtpos_file = os.path.join(rt1_trajectory_folder, 'IGTPos.txt')
    if os.path.exists(rt1_igtpos_file):
        rt1_measured_pose = read_poses_from_txt(rt1_igtpos_file)
        # Filter out the default pose
        valid_rt1_poses = []
        for pose in rt1_measured_pose:
            if not np.allclose(pose, default_pose) and not np.isnan(pose).any():
                valid_rt1_poses.append(pose)
        rt1_measured_pose = np.array(valid_rt1_poses)
        rt1_untracked_percentage = np.isnan(rt1_measured_pose).any(axis=1).mean() * 100
        rt1_measured_pose = rt1_measured_pose[~np.isnan(rt1_measured_pose).any(axis=1)]

        if rt1_measured_pose.size > 0:
            transformed_rt1_trajectory = apply_transform(rt1_measured_pose, R, T)
            rt1_tree = cKDTree(trajectory1)
            rt1_distances, _ = rt1_tree.query(transformed_rt1_trajectory)
            rt1_tre_rmse = np.sqrt(np.mean(rt1_distances**2))
            results['RT1 Trajectory TRE'] = f"{rt1_tre_rmse:.4f}"
            transformed_trajectories['rt1_trajectory'] = {
                'real': trajectory1,
                'measured': rt1_measured_pose,
                'transformed': transformed_rt1_trajectory
            }
            transformed_trajectories['rt1_untracked_percentage'] = rt1_untracked_percentage

            if rt1_untracked_percentage > 0:
                print(f"RT1 Trajectory Points Untracked: {rt1_untracked_percentage:.2f}%")
                
            traj_errors[0].append(rt1_tre_rmse)
        else:
            results['RT1 Trajectory TRE'] = "N/A"
            transformed_trajectories['rt1_untracked_percentage'] = rt1_untracked_percentage
    else:
        results['RT1 Trajectory TRE'] = "N/A"

    # Process RT2 Trajectory Data
    rt2_trajectory_folder = os.path.join(data_folder, f'{trial_name}_RT2')
    rt2_igtpos_file = os.path.join(rt2_trajectory_folder, 'IGTPos.txt')
    if os.path.exists(rt2_igtpos_file):
        rt2_measured_pose = read_poses_from_txt(rt2_igtpos_file)
        # Filter out the default pose
        valid_rt2_poses = []
        for pose in rt2_measured_pose:
            if not np.allclose(pose, default_pose) and not np.isnan(pose).any():
                valid_rt2_poses.append(pose)
        rt2_measured_pose = np.array(valid_rt2_poses)
        rt2_untracked_percentage = np.isnan(rt2_measured_pose).any(axis=1).mean() * 100
        rt2_measured_pose = rt2_measured_pose[~np.isnan(rt2_measured_pose).any(axis=1)]

        if rt2_measured_pose.size > 0:
            transformed_rt2_trajectory = apply_transform(rt2_measured_pose, R, T)
            rt2_tree = cKDTree(trajectory2)
            rt2_distances, _ = rt2_tree.query(transformed_rt2_trajectory)
            rt2_tre_rmse = np.sqrt(np.mean(rt2_distances**2))
            results['RT2 Trajectory TRE'] = f"{rt2_tre_rmse:.4f}"
            transformed_trajectories['rt2_trajectory'] = {
                'real': trajectory2,
                'measured': rt2_measured_pose,
                'transformed': transformed_rt2_trajectory
            }
            transformed_trajectories['rt2_untracked_percentage'] = rt2_untracked_percentage

            if rt2_untracked_percentage > 0:
                print(f"RT2 Trajectory Points Untracked: {rt2_untracked_percentage:.2f}%")
                
            traj_errors[1].append(rt2_tre_rmse)
        else:
            results['RT2 Trajectory TRE'] = "N/A"
            transformed_trajectories['rt2_untracked_percentage'] = rt2_untracked_percentage
    else:
        results['RT2 Trajectory TRE'] = "N/A"

    # Process RT3 Trajectory Data
    rt3_trajectory_folder = os.path.join(data_folder, f'{trial_name}_RT3')
    rt3_igtpos_file = os.path.join(rt3_trajectory_folder, 'IGTPos.txt')
    if os.path.exists(rt3_igtpos_file):
        rt3_measured_pose = read_poses_from_txt(rt3_igtpos_file)
        # Filter out the default pose
        valid_rt3_poses = []
        for pose in rt3_measured_pose:
            if not np.allclose(pose, default_pose) and not np.isnan(pose).any():
                valid_rt3_poses.append(pose)
        rt3_measured_pose = np.array(valid_rt3_poses)
        rt3_untracked_percentage = np.isnan(rt3_measured_pose).any(axis=1).mean() * 100
        rt3_measured_pose = rt3_measured_pose[~np.isnan(rt3_measured_pose).any(axis=1)]

        if rt3_measured_pose.size > 0:
            transformed_rt3_trajectory = apply_transform(rt3_measured_pose, R, T)
            rt3_tree = cKDTree(trajectory3)
            rt3_distances, _ = rt3_tree.query(transformed_rt3_trajectory)
            rt3_tre_rmse = np.sqrt(np.mean(rt3_distances**2))
            results['RT3 Trajectory TRE'] = f"{rt3_tre_rmse:.4f}"
            transformed_trajectories['rt3_trajectory'] = {
                'real': trajectory3,
                'measured': rt3_measured_pose,
                'transformed': transformed_rt3_trajectory
            }
            transformed_trajectories['rt3_untracked_percentage'] = rt3_untracked_percentage

            if rt3_untracked_percentage > 0:
                print(f"RT3 Trajectory Points Untracked: {rt3_untracked_percentage:.2f}%")
                
            traj_errors[2].append(rt3_tre_rmse)
        else:
            results['RT3 Trajectory TRE'] = "N/A"
            transformed_trajectories['rt3_untracked_percentage'] = rt3_untracked_percentage
    else:
        results['RT3 Trajectory TRE'] = "N/A"

    # Process RT4 Trajectory Data
    rt4_trajectory_folder = os.path.join(data_folder, f'{trial_name}_RT4')
    rt4_igtpos_file = os.path.join(rt4_trajectory_folder, 'IGTPos.txt')
    if os.path.exists(rt4_igtpos_file):
        rt4_measured_pose = read_poses_from_txt(rt4_igtpos_file)
        # Filter out the default pose
        valid_rt4_poses = []
        for pose in rt4_measured_pose:
            if not np.allclose(pose, default_pose) and not np.isnan(pose).any():
                valid_rt4_poses.append(pose)
        rt4_measured_pose = np.array(valid_rt4_poses)
        rt4_untracked_percentage = np.isnan(rt4_measured_pose).any(axis=1).mean() * 100
        rt4_measured_pose = rt4_measured_pose[~np.isnan(rt4_measured_pose).any(axis=1)]

        if rt4_measured_pose.size > 0:
            transformed_rt4_trajectory = apply_transform(rt4_measured_pose, R, T)
            rt4_tree = cKDTree(trajectory4)
            rt4_distances, _ = rt4_tree.query(transformed_rt4_trajectory)
            rt4_tre_rmse = np.sqrt(np.mean(rt4_distances**2))
            results['RT4 Trajectory TRE'] = f"{rt4_tre_rmse:.4f}"
            transformed_trajectories['rt4_trajectory'] = {
                'real': trajectory4,
                'measured': rt4_measured_pose,
                'transformed': transformed_rt4_trajectory
            }
            transformed_trajectories['rt4_untracked_percentage'] = rt4_untracked_percentage

            if rt4_untracked_percentage > 0:
                print(f"RT4 Trajectory Points Untracked: {rt4_untracked_percentage:.2f}%")
                
            traj_errors[3].append(rt3_tre_rmse)
        else:
            results['RT4 Trajectory TRE'] = "N/A"
            transformed_trajectories['rt4_untracked_percentage'] = rt4_untracked_percentage
    else:
        results['RT4 Trajectory TRE'] = "N/A"

    # Calculate average trajectory TRE across all trajectories
    trajectory_tre_values = []
    if 'Back Trajectory TRE' in results and results['Back Trajectory TRE'] != "N/A":
        trajectory_tre_values.append(float(results['Back Trajectory TRE']))
    if 'Front Trajectory TRE' in results and results['Front Trajectory TRE'] != "N/A":
        trajectory_tre_values.append(float(results['Front Trajectory TRE']))
    if 'RT1 Trajectory TRE' in results and results['RT1 Trajectory TRE'] != "N/A":
        trajectory_tre_values.append(float(results['RT1 Trajectory TRE']))
    if 'RT2 Trajectory TRE' in results and results['RT2 Trajectory TRE'] != "N/A":
        trajectory_tre_values.append(float(results['RT2 Trajectory TRE']))
    if 'RT3 Trajectory TRE' in results and results['RT3 Trajectory TRE'] != "N/A":
        trajectory_tre_values.append(float(results['RT3 Trajectory TRE']))
    if 'RT4 Trajectory TRE' in results and results['RT4 Trajectory TRE'] != "N/A":
        trajectory_tre_values.append(float(results['RT4 Trajectory TRE']))

    if trajectory_tre_values:
        avg_trajectory_tre = np.mean(trajectory_tre_values)
        results['Total Trajectory TRE'] = f"{avg_trajectory_tre:.4f}"
    else:
        results['Total Trajectory TRE'] = "N/A"

    # Print results
    print("\nFRE RMSE [mm]:", results['FRE'])
    print("\nTRE RMSE [mm]:", results['TRE'])
    print("\nBack Trajectory TRE RMSE [mm]:", results['Back Trajectory TRE'])
    print("\nFront Trajectory TRE RMSE [mm]:", results['Front Trajectory TRE'])
    print("\nRT1 Trajectory TRE RMSE [mm]:", results['RT1 Trajectory TRE'])
    print("\nRT2 Trajectory TRE RMSE [mm]:", results['RT2 Trajectory TRE'])
    print("\nRT3 Trajectory TRE RMSE [mm]:", results['RT3 Trajectory TRE'])
    print("\nRT4 Trajectory TRE RMSE [mm]:", results['RT4 Trajectory TRE'])
    print("\nTotal Trajectory TRE RMSE [mm]:", results['Total Trajectory TRE'])

    # Save results to JSON file
    results_path = os.path.join(data_folder, f"{os.path.basename(data_folder)}_results_atracsys.json")
    with open(results_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    print(f"\nResults saved to {results_path}")

    # Create a directory to save plots
    save_folder = os.path.join(data_folder, "atracsys_plots")
    os.makedirs(save_folder, exist_ok=True)

    # Plot everything together at the end
    if plot3D:
        if transformed_trajectories['fiducial'] is not None:
            try:
                # Unpack the fiducial and target data
                landmarks = transformed_trajectories['fiducial']['real']
                transformed_fiducials = transformed_trajectories['fiducial']['transformed']
                targets = transformed_trajectories['target']['real']
                transformed_targets = transformed_trajectories['target']['transformed']

                # Plotting the fiducial and target points
                fig = plot_3d_points(
                    real_fiducial_points=landmarks,
                    transformed_fiducial_points=transformed_fiducials,
                    real_target_points=targets,
                    transformed_target_points=transformed_targets,
                    title="Point Registration"
                )

                # Save the point registration plot
                fig.savefig(os.path.join(save_folder, "points_plot.png"))
                plt.show(fig)

            except TypeError as e:
                print(f"Error unpacking or plotting data: {e}")

        # Plot trajectories if available
        if transformed_trajectories['back_trajectory'] or transformed_trajectories['front_trajectory'] or transformed_trajectories['rt1_trajectory'] or transformed_trajectories['rt2_trajectory'] or transformed_trajectories['rt3_trajectory'] or transformed_trajectories['rt4_trajectory']:
            try:
                back_untracked = transformed_trajectories.get('back_untracked_percentage', 0)
                front_untracked = transformed_trajectories.get('front_untracked_percentage', 0)
                rt1_untracked = transformed_trajectories.get('rt1_untracked_percentage', 0)
                rt2_untracked = transformed_trajectories.get('rt2_untracked_percentage', 0)
                rt3_untracked = transformed_trajectories.get('rt3_untracked_percentage', 0)
                rt4_untracked = transformed_trajectories.get('rt4_untracked_percentage', 0)
                title = "Trajectory Tracking"
                if back_untracked > 0 or front_untracked > 0 or rt1_untracked > 0 or rt2_untracked > 0 or rt3_untracked > 0 or rt4_untracked > 0:
                    title += f"\nBack Points Untracked: {back_untracked:.2f}%, Front Points Untracked: {front_untracked:.2f}%, RT1 Points Untracked: {rt1_untracked:.2f}%, RT2 Points Untracked: {rt2_untracked:.2f}%, RT3 Points Untracked: {rt3_untracked:.2f}%, RT4 Points Untracked: {rt4_untracked:.2f}%"

                real_trajectories = []
                measured_trajectories = []

                if transformed_trajectories['back_trajectory']:
                    real_trajectories.append(transformed_trajectories['back_trajectory']['real'])
                    measured_trajectories.append(transformed_trajectories['back_trajectory']['transformed'])
                else:
                    real_trajectories.append(np.empty((0, 3)))
                    measured_trajectories.append(np.empty((0, 3)))

                if transformed_trajectories['front_trajectory']:
                    real_trajectories.append(transformed_trajectories['front_trajectory']['real'])
                    measured_trajectories.append(transformed_trajectories['front_trajectory']['transformed'])
                else:
                    real_trajectories.append(np.empty((0, 3)))
                    measured_trajectories.append(np.empty((0, 3)))

                if transformed_trajectories['rt1_trajectory']:
                    real_trajectories.append(transformed_trajectories['rt1_trajectory']['real'])
                    measured_trajectories.append(transformed_trajectories['rt1_trajectory']['transformed'])
                else:
                    real_trajectories.append(np.empty((0, 3)))
                    measured_trajectories.append(np.empty((0, 3)))

                if transformed_trajectories['rt2_trajectory']:
                    real_trajectories.append(transformed_trajectories['rt2_trajectory']['real'])
                    measured_trajectories.append(transformed_trajectories['rt2_trajectory']['transformed'])
                else:
                    real_trajectories.append(np.empty((0, 3)))
                    measured_trajectories.append(np.empty((0, 3)))

                if transformed_trajectories['rt3_trajectory']:
                    real_trajectories.append(transformed_trajectories['rt3_trajectory']['real'])
                    measured_trajectories.append(transformed_trajectories['rt3_trajectory']['transformed'])
                else:
                    real_trajectories.append(np.empty((0, 3)))
                    measured_trajectories.append(np.empty((0, 3)))

                if transformed_trajectories['rt4_trajectory']:
                    real_trajectories.append(transformed_trajectories['rt4_trajectory']['real'])
                    measured_trajectories.append(transformed_trajectories['rt4_trajectory']['transformed'])
                else:
                    real_trajectories.append(np.empty((0, 3)))
                    measured_trajectories.append(np.empty((0, 3)))

                fig = plot_3d_trajectories(
                    real_trajectories,
                    measured_trajectories,
                    title
                )

                # Save the trajectory plot
                fig.savefig(os.path.join(save_folder, "trajectory_plot.png"))
                plt.show(fig)

            except TypeError as e:
                print(f"Error unpacking or plotting data: {e}")
    
    return fre_ppe, tre_ppe, traj_errors

def aggregate_results(data_folder, configs=['top', 'left', 'right']):
    """Aggregate the results for all configurations and Atracsys into one final table."""
    metrics = ['FRE', 'TRE', 'Back Trajectory TRE', 'Front Trajectory TRE', 'RT1 Trajectory TRE', 'RT2 Trajectory TRE', 'RT3 Trajectory TRE', 'RT4 Trajectory TRE', 'Total Trajectory TRE']
    aggregated_data = {metric: {i: {method: [] for method in [
            'Monocular Pose Fusion', 'Monocular Pose Fusion Kalman', 'Monocular Pose Fusion Kalman Adaptive', 
            'Triangulation', 'Triangulation Kalman', 'Triangulation Kalman Adaptive'
        ]} for i in range(1, 6)} for metric in metrics}
    trial_name = os.path.basename(data_folder).split('_')[0]
    for config in configs:
        config_folder = os.path.join(data_folder, f'_{trial_name}_{config.upper()}')
        json_file = os.path.join(config_folder, f"{os.path.basename(config_folder)}_results.json")
        if os.path.exists(json_file):
            with open(json_file, 'r') as file:
                config_results = json.load(file)
            for metric in metrics:
                if metric in config_results:
                    for method, results_dict in config_results[metric].items():
                        if isinstance(results_dict, dict):
                            for camera_num, value in results_dict.items():
                                camera_num = int(camera_num)
                                numeric_value = extract_numeric_value(value)
                                if isinstance(numeric_value, (int, float)) and not np.isnan(numeric_value):
                                    aggregated_data[metric][camera_num][method].append(numeric_value)
    atracsys_json_file = os.path.join(data_folder, f"{os.path.basename(data_folder)}_results_atracsys.json")
    atracsys_results = None
    if os.path.exists(atracsys_json_file):
        with open(atracsys_json_file, 'r') as file:
            atracsys_results = json.load(file)
    final_tables = {}
    for metric in metrics:
        df = pd.DataFrame(index=range(1, 6), columns=[
            'Monocular Pose Fusion', 'Monocular Pose Fusion Kalman', 'Monocular Pose Fusion Kalman Adaptive', 
            'Triangulation', 'Triangulation Kalman', 'Triangulation Kalman Adaptive'
        ])
        for camera_num in range(1, 6):
            for method in df.columns:
                if method in aggregated_data[metric][camera_num]:
                    avg_std = calculate_avg_and_std(aggregated_data[metric][camera_num][method])
                    df.at[camera_num, method] = avg_std
                else:
                    df.at[camera_num, method] = "N/A"
        if atracsys_results and metric in atracsys_results:
            atracsys_value = extract_numeric_value(atracsys_results.get(metric, "N/A"))
            if isinstance(atracsys_value, (int, float)) and not np.isnan(atracsys_value):
                df.loc['Atracsys', :] = f"{atracsys_value:.4f}"
            else:
                df.loc['Atracsys', :] = "N/A"
        final_tables[metric] = df
    return final_tables

def aggregate_results_all(data_folders, results_allTrials, configs=['top', 'left', 'right']):
    """Aggregate the results for all configurations and Atracsys into one final table."""
    metrics = ['FRE', 'TRE', 'Back Trajectory TRE', 'Front Trajectory TRE', 'RT1 Trajectory TRE', 'RT2 Trajectory TRE', 'RT3 Trajectory TRE', 'RT4 Trajectory TRE', 'Total Trajectory TRE']
    aggregated_data = {metric: {i: {method: [] for method in fusion_methods} for i in range(1, 6)} for metric in metrics}  # Separate structures for each metric
    atracsys_results = {metric: [] for metric in metrics}
    
    for i_trial, data_folder in enumerate(data_folders):
        # Load data from each configuration
        trial_name = os.path.basename(data_folder).split('_')[0]
        
        # j_trial = i_trial*dataPts_trial  # first index for this trial in the df
        # results_df.loc[j_trial:j_trial+dataPts_trial, 'Trial'] = trial_name

        for i_config, config in enumerate(configs):
            # j_config = j_trial + i_config*dataPts_config
            # results_df.loc[j_config:j_config+dataPts_config, 'Config'] = config
            
            config_folder = os.path.join(data_folder, f'_{trial_name}_{config.upper()}')
            json_file = os.path.join(config_folder, f"{os.path.basename(config_folder)}_results.json")
            if os.path.exists(json_file):
                with open(json_file, 'r') as file:
                    config_results = json.load(file)
                
                # Iterate through each metric separately
                for metric in metrics:
                    if metric in config_results:
                        for i_method, (method, results) in enumerate(config_results[metric].items()):
                            # j_method = j_config + i_method*dataPts_method
                            # method_split = method.split(' ', maxsplit=1)
                            # method_tracking = method_split[0]
                            # method_fusion = method_split[1] if len(method_split)>1 else 'Basic'
                            # results_df.loc[j_method:j_method+dataPts_method, 'Method_Tracking'] = method_tracking
                            # results_df.loc[j_method:j_method+dataPts_method, 'Method_Fusion'] = method_fusion
                            if isinstance(results, dict):
                                for i_cam, (camera_num, value) in enumerate(results.items()):
                                    camera_num = int(camera_num)
                                    # j = j_method+i_cam  # Index in results_df where the value is being stored
                                    # results_df.at[j, 'Cams'] = camera_num
                                    numeric_value = extract_numeric_value(value)
                                    if isinstance(numeric_value, (int, float)) and not np.isnan(numeric_value):
                                        aggregated_data[metric][camera_num][method].append(numeric_value)
                                        # results_df.at[j, metric] = numeric_value
        
        
        # Atracsys results
        atracsys_json_file = os.path.join(data_folder, f"{os.path.basename(data_folder)}_results_atracsys.json")
        if os.path.exists(atracsys_json_file):
            with open(atracsys_json_file, 'r') as file:
                atracsys_json = json.load(file)
                for metric in metrics:
                    numeric_value = extract_numeric_value(atracsys_json[metric])
                    atracsys_results[metric].append(numeric_value)
                    
    # CSV output for stats analysis
    metrics_df = ['FRE', 'TRE', 'Traj_RMSE']
    dataPts_method = 1 if GRID else N_CAMS
    dataPts_point = len(fusion_methods) * dataPts_method
    dataPts_config = (len(POINTS_FRE) + len(POINTS_TRE) + len(TRAJECTORIES)) * dataPts_point
    dataPts_ring = len(configs) * dataPts_config
    dataPts_atracsys = len(POINTS_FRE) + len(POINTS_TRE) + len(TRAJECTORIES)
    dataPts_trial = dataPts_ring + dataPts_atracsys
    dataPts_total = N_trials * dataPts_trial
    # df_cols = ['Trial', 'Device', 'Config', 'Point', 'Tracking', 'Fusion', 'Cams', 'FRE', 'TRE', 'Traj_RMSE']
    df_cols = ['Trial', 'Device', 'Config', 'Point', 'Tracking', 'Fusion', 'Cams', 'Metric', 'Error']
    results_df = pd.DataFrame(np.nan, index=np.arange(dataPts_total), columns=df_cols, dtype="object")
    # results_df = results_df.astype(dtype={"Cams":"float64", "FRE":"float64", "TRE":"float64", "Traj_RMSE":"float64"})
    results_df = results_df.astype(dtype={"Trial":"float64", "Cams":"float64", "Error":"float64"})
    
    for i_trial, trial_name in enumerate(TRIALS):
        j_trial = i_trial*dataPts_trial  # first index for this trial in the df
        results_df.loc[j_trial:j_trial+dataPts_trial-1, 'Trial'] = trial_name
        results_df.loc[j_trial:j_trial+dataPts_ring-1, 'Device'] = 'Ring'
        
        for i_config, config in enumerate(configs):
            j_config = j_trial + i_config*dataPts_config
            results_df.loc[j_config:j_config+dataPts_config-1, 'Config'] = config
            
            keys = results_allTrials[i_trial][i_config][-1]
            i_point = 0  # Point index in the combined list POINTS_FRE + POINTS_TRE + TRAJECTORIES
            for i_metric, (metric, points) in enumerate(zip(metrics_df, (POINTS_FRE, POINTS_TRE, TRAJECTORIES))):
                for k_point, point in enumerate(points):  # k_point is the pt index within its own set (FRE, TRE, or Traj)
                    j_point = j_config + i_point*dataPts_point  # j_point is the first index for this point in the full dataframe
                    results_df.loc[j_point:j_point+dataPts_point-1, 'Point'] = point
                    for i_key, key in enumerate(keys):
                        j = j_point + i_key
                        camera_num, method = key.split(' Cam ')
                        camera_num = int(camera_num)
                        method_split = method.split(' ', maxsplit=1)
                        method_tracking = method_split[0]
                        method_fusion = method_split[1] if len(method_split)>1 else 'Basic'
                        # results_df.loc[j, ['Tracking','Fusion','Cams']] = method_tracking, method_fusion, camera_num
                        # results_df.at[j, metric] = results_allTrials[i_trial][i_config][i_metric][k_point][i_key]
                        results_df.loc[j, ['Tracking','Fusion','Cams','Metric']] = method_tracking, method_fusion, camera_num, metric
                        results_df.at[j, "Error"] = results_allTrials[i_trial][i_config][i_metric][k_point][i_key]
                        
                    i_point += 1
        
        j_atracsys = j_trial+dataPts_ring
        results_df.loc[j_atracsys:j_atracsys+dataPts_atracsys-1, 'Device'] = 'Atracsys'
        i_point = 0
        for i_metric, (metric, points) in enumerate(zip(metrics_df, (POINTS_FRE, POINTS_TRE, TRAJECTORIES))):
            for k_point, point in enumerate(points):
                j = j_atracsys + i_point
                # results_df.at[j, 'Point'] = point
                # results_df.at[j, metric] = results_allTrials[i_trial][-1][i_metric][k_point]
                results_df.loc[j, ['Point','Metric']] = point, metric
                results_df.at[j, 'Error'] = results_allTrials[i_trial][-1][i_metric][k_point]
                i_point += 1
                
    results_df.dropna(inplace=True, subset=['Error'])
    results_df = results_df.astype(dtype={"Trial":"int64"})
    # results_df.reset_index(drop=True, inplace=True)
    # results_df = results_df.astype(dtype={"Cams":"float64"})
    results_df.to_csv(f'{base_folder}/results_' + ('G' if GRID else 'H') + ','.join([str(t) for t in TRIALS]) + '.csv', index=False, na_rep='NA')
    
    # Create DataFrame for final results
    final_tables = {}
    for metric in metrics:
        df = pd.DataFrame(index=range(1, 6), columns=[
            'Monocular Pose Fusion', 'Monocular Pose Fusion Kalman', 'Monocular Pose Fusion Kalman Adaptive', 
            'Triangulation', 'Triangulation Kalman', 'Triangulation Kalman Adaptive'
        ])

        for camera_num in range(1, 6):
            for method in df.columns:
                if method in aggregated_data[metric][camera_num]:
                    avg_std = calculate_avg_and_std(aggregated_data[metric][camera_num][method])
                    df.at[camera_num, method] = avg_std
                else:
                    df.at[camera_num, method] = "N/A"

        # Add Atracsys results as the last row
        if atracsys_results and metric in atracsys_results:
            atracsys_value = atracsys_results.get(metric, "N/A")
            df.loc['Atracsys', :] = calculate_avg_and_std(atracsys_value)

        final_tables[metric] = df

    # Create separate tables for each config
    config_tables = {}
    for config in configs:
        config_tables[config] = {}
        
        # Create aggregated data structure for this specific config only
        config_aggregated_data = {metric: {i: {method: [] for method in fusion_methods} for i in range(1, 6)} for metric in metrics}
        
        # Collect data from all trials for this specific config using the same approach as the main aggregation
        for i_trial, data_folder in enumerate(data_folders):
            trial_name = os.path.basename(data_folder).split('_')[0]
            config_folder = os.path.join(data_folder, f'_{trial_name}_{config.upper()}')
            json_file = os.path.join(config_folder, f"{os.path.basename(config_folder)}_results.json")
            
            if os.path.exists(json_file):
                with open(json_file, 'r') as file:
                    config_results = json.load(file)
                
                # Process each metric for this config
                for metric in metrics:
                    if metric in config_results:
                        for method, results in config_results[metric].items():
                            if isinstance(results, dict):
                                for camera_num_str, value in results.items():
                                    camera_num = int(camera_num_str)
                                    numeric_value = extract_numeric_value(value)
                                    if isinstance(numeric_value, (int, float)) and not np.isnan(numeric_value):
                                        config_aggregated_data[metric][camera_num][method].append(numeric_value)
        
        # Create DataFrames for this config
        for metric in metrics:
            df = pd.DataFrame(index=range(1, 6), columns=fusion_methods)
            for camera_num in range(1, 6):
                for method in fusion_methods:
                    data = config_aggregated_data[metric][camera_num][method]
                    if data:
                        median_std = calculate_avg_and_std(data)
                        df.at[camera_num, method] = median_std
                    else:
                        df.at[camera_num, method] = "N/A"
            
            # Add Atracsys data if available (same for all configs since it's not config-specific)
            if atracsys_results and metric in atracsys_results:
                atracsys_value = atracsys_results.get(metric, "N/A")
                df.loc['Atracsys', :] = calculate_avg_and_std(atracsys_value)
            
            config_tables[config][metric] = df
    
    return final_tables, config_tables


def extract_numeric_value(value):
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return np.nan
    return value

def export_tables_to_latex(final_results, config_results, trials, output_folder, grid=False):
    """Export tables to LaTeX files for Overleaf"""
    
    # Define metric mappings for LaTeX
    metric_latex_names = {
        'FRE': 'Fiducial Point\\\\Registration RMSE',
        'TRE': 'Target Point\\\\Registration RMSE', 
        'Back Trajectory TRE': 'T2 Trajectory\\\\Registration RMSE',
        'Front Trajectory TRE': 'T1 Trajectory\\\\Registration RMSE',
        'RT1 Trajectory TRE': 'RT1 Trajectory\\\\Registration RMSE',
        'RT2 Trajectory TRE': 'RT2 Trajectory\\\\Registration RMSE', 
        'RT3 Trajectory TRE': 'RT3 Trajectory\\\\Registration RMSE',
        'RT4 Trajectory TRE': 'RT4 Trajectory\\\\Registration RMSE',
        'Total Trajectory TRE': 'Total Trajectory\\\\Registration RMSE'
    }
    
    experiment_type = "Grid" if grid else "Head"
    trial_text = f"{len(trials)} {experiment_type} Experiment trials" if len(trials) > 1 else f"{experiment_type} Trial {trials[0]}"
    
    def create_main_table(results, title_suffix=""):
        # Select key metrics for main table
        key_metrics = ['FRE', 'TRE']
        if grid:
            key_metrics.append('Total Trajectory TRE')
        else:
            key_metrics.extend(['Front Trajectory TRE', 'Back Trajectory TRE', 'Total Trajectory TRE'])
        
        latex_output = []
        latex_output.append("\\begin{table}[H]")
        latex_output.append("\\centering")
        caption = f"\\caption{{Registration RMSE [mm (±std)] using medians across {trial_text}{title_suffix}.}}"
        latex_output.append(caption)
        latex_output.append("\\resizebox{\\textwidth}{!}{")
        latex_output.append("\\begin{tabular}{l*{8}{c}}")
        latex_output.append("\\toprule")
        
        # Header
        latex_output.append(" & \\multicolumn{1}{c}{\\multirow{2}{*}{\\# of Cameras}} & \\multicolumn{3}{c}{Monocular Pose Fusion} & \\multicolumn{3}{c}{Multi-View Triangulation} & \\multicolumn{1}{c}{\\multirow{2}{*}{FusionTrack 500}} \\\\")
        latex_output.append("\\cmidrule(lr){3-5} \\cmidrule(lr){6-8}")
        latex_output.append(" & & Basic & Kalman & Kalman Adaptive & Basic & Kalman & Kalman Adaptive & \\\\")
        latex_output.append("\\midrule")
        
        # Data rows
        for metric in key_metrics:
            if metric in results:
                df = results[metric]
                latex_name = metric_latex_names.get(metric, metric)
                
                # First row with metric name
                latex_output.append(f"\\multirow{{5}}{{*}}{{\\makecell{{{latex_name}}}}}")
                
                for cam_num in range(1, 6):
                    if cam_num == 1:
                        line = f" & {cam_num}"
                    else:
                        line = f" & {cam_num}"
                    
                    # Monocular methods
                    for method in ['Monocular Pose Fusion', 'Monocular Pose Fusion Kalman', 'Monocular Pose Fusion Kalman Adaptive']:
                        value = df.loc[cam_num, method] if method in df.columns else "N/A"
                        line += f" & {value}"
                    
                    # Stereo methods  
                    for method in ['Triangulation', 'Triangulation Kalman', 'Triangulation Kalman Adaptive']:
                        if cam_num == 1:
                            line += " & N/A"  # Triangulation needs at least 2 cameras
                        else:
                            value = df.loc[cam_num, method] if method in df.columns else "N/A"
                            line += f" & {value}"
                    
                    # Atracsys (only on first row) - same value across all columns since it's aggregated across configs
                    if cam_num == 1:
                        if 'Atracsys' in df.index:
                            atracsys_value = df.loc['Atracsys', df.columns[0]]
                        else:
                            atracsys_value = "N/A"
                        line += f" & \\multirow{{5}}{{*}}{{{atracsys_value}}} \\\\"
                    else:
                        line += " & \\\\"
                    
                    latex_output.append(line)
                
                latex_output.append("\\midrule")
        
        # Remove last midrule and add bottomrule
        if latex_output[-1] == "\\midrule":
            latex_output[-1] = "\\bottomrule"
        
        latex_output.append("\\end{tabular}")
        latex_output.append("}")
        table_label = f"tab:{'grid' if grid else 'head'}_results{'_' + title_suffix.lower().replace(' ', '_').replace('-', '_') if title_suffix else ''}"
        latex_output.append(f"\\label{{{table_label}}}")
        latex_output.append("\\end{table}")
        
        return "\\n".join(latex_output)
    
    # Create output files
    trial_str = "_".join(map(str, trials))
    
    # Main aggregated table
    main_table = create_main_table(final_results)
    main_filename = f"{output_folder}/latex_table_main_{experiment_type.lower()}_trials_{trial_str}.tex"
    with open(main_filename, 'w') as f:
        f.write(main_table)
    print(f"Main aggregated table saved to: {main_filename}")
    
    # Config-specific tables
    for config in ['top', 'left', 'right']:
        if config in config_results:
            config_table = create_main_table(config_results[config], f" - {config.upper()} Config Only")
            config_filename = f"{output_folder}/latex_table_{config}_{experiment_type.lower()}_trials_{trial_str}.tex"
            with open(config_filename, 'w') as f:
                f.write(config_table)
            print(f"{config.upper()} config table saved to: {config_filename}")
    
    print(f"All LaTeX tables saved! You can copy-paste the contents of these .tex files directly into Overleaf.")


def calculate_avg_and_std(data):
    numeric_data = [float(value) for value in data if isinstance(value, (int, float)) or (isinstance(value, str) and value.replace('.', '', 1).isdigit())]
    if len(numeric_data) == 0:
        return "N/A"
    median = np.median(numeric_data)
    std = np.std(numeric_data)
    return f"{median:.4f} (±{std:.4f})"

def save_final_results(data_folder, final_results):
    aggregated_results_path = os.path.join(data_folder, f"{os.path.basename(data_folder)}_aggregated_results.json")
    results_dict = {metric: df.to_dict() for metric, df in final_results.items()}
    with open(aggregated_results_path, 'w') as file:
        json.dump(results_dict, file, indent=4)
    print(f"\nAggregated Results for {os.path.basename(data_folder)}:\n")
    for metric, df in final_results.items():
        print(f"{metric} RMSE [mm]:")
        print(df.to_string(index=True, justify='center'))
        print("")

def run_trial(data_folder, points_FRE, points_TRE, configs=['top', 'left', 'right']):
    """Run the results table calculations for top, left, and right configurations and Atracsys."""
    trial_name = os.path.basename(data_folder).split('_')[0]
    # configs = ['top', 'left', 'right']

    results_allConfigs = []  # Stores errors organized as Config > (FRE, TRE, Traj, Key) > Point/Traj # > Error
    
    # Step 1: Calculate results for each config
    for config in configs:
        config_folder = os.path.join(data_folder, f'_{trial_name}_{config.upper()}')
        
        # Debugging: Print the configuration folder being processed
        print(f"\nProcessing configuration: {config}")
        print(f"Configuration folder: {config_folder}")
        
        if not os.path.exists(config_folder):
            print(f"Configuration folder '{config}' not found in {data_folder}")
            continue
        
        # Debugging: Print the files found in the configuration folder
        files = {f.split('_')[1]: f for f in os.listdir(config_folder) if f.endswith('.json') and not f.startswith('_')}
        print(f"Files found in {config_folder}: {files}")
        
        print(f"\nCalculating results for {config} configuration...")
        results_allConfigs.append(results_table(config_folder, points_FRE, points_TRE, mean=MEAN, plot3D=PLOT, savePlots=SAVE_PLOTS))

    
    # Step 2: Calculate Atracsys results
    print(f"\nCalculating results for Atracsys...")
    results_allConfigs.append(results_atracsys(data_folder, points_FRE, points_TRE, mean=MEAN, plot3D=PLOT))
    
    # Step 3: Aggregate results into final table
    final_results = aggregate_results(data_folder, configs=configs)
    
    # Step 4: Save and display final aggregated results
    save_final_results(data_folder, final_results)
    
    return results_allConfigs

# %%
if __name__ == "__main__":
    results_allTrials = []  # Stores errors organized as Trial >  Config > (FRE, TRE, Traj, Key) > Point/Traj # > Error
    for data_folder in data_folders:
        results_allTrials.append(run_trial(data_folder, POINTS_FRE, POINTS_TRE, configs=CONFIGS))
        
    final_results, config_results = aggregate_results_all(data_folders, results_allTrials, configs=CONFIGS)
    # Display the aggregated results (across all configs)
    print(f"\nAggregated Results Across All Configs for Trials {TRIALS}:\n")
    for metric, df in final_results.items():
        print(f"{metric} RMSE [mm]:")
        print(df.to_string(index=True, justify='center'))
        print("")
    
    # Display config-specific results
    for config in CONFIGS:
        print(f"\nResults for {config.upper()} Config Only (Trials {TRIALS}):\n")
        for metric, df in config_results[config].items():
            print(f"{metric} RMSE [mm] - {config.upper()} Config:")
            print(df.to_string(index=True, justify='center'))
            print("")
    
    # Export LaTeX tables
    print("\n" + "="*60)
    print("EXPORTING LATEX TABLES...")
    print("="*60)
    export_tables_to_latex(final_results, config_results, TRIALS, base_folder, grid=GRID)
    
    if LANDMARK_TEST:
        print(f'\nAtracsys landmark accuracy analysis:')
        points_all = POINTS_FRE + POINTS_TRE
        N_points = len(points_all)
        N_trials = len(data_folders)
        tre_all = dict([(point, np.full(N_trials, np.nan, dtype=np.float32)) for point in points_all])
        
        for trialIndex, data_folder in enumerate(data_folders):
            trial_name = os.path.basename(data_folder).split('_')[0]
            print(f'\nTrial {trial_name}:')
            with suppress_print():
                for i, points in enumerate(it.combinations(points_all[::-1], len(points_all)-1)):
                    points_FRE = list(points)
                    point_TRE = [points_all[i]]
                    mask = np.ones(N_points,bool)
                    mask[i] = False
                    fre_ppe, tre_ppe, _ = results_atracsys(data_folder, points_FRE, point_TRE, mean=True)
                    tre_all[points_all[i]][trialIndex] = tre_ppe[0]
            for point in points_all:
                print(f'{point}: {tre_all[point][trialIndex]:.4f}')
        
        tre_avg = dict([(point, np.median(tre_all[point])) for point in points_all])
        tre_std = dict([(point, np.std(tre_all[point])) for point in points_all])
        print('\nMedian of trials:')
        for point in points_all:
            print(f'{point}: {tre_avg[point]:.4f} (±{tre_std[point]:.4f})')