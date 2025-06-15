import json
import os
import numpy as np
from util.landmark_registration import register, apply_transform, calculate_TRE
import matplotlib.pyplot as plt
import sys
import contextlib
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree
import shutil
import itertools as it

base_folder = '/home/george/MultiCameraTracking/Registration/Landmark_Registration_Trials'
trials = [1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18, 19, 20]
# trials = [15, 16, 17, 18]

GRID = False
LANDMARK_TEST = True
FLIP = False  # (R6 <-> 9, R8 <-> 6)
MEAN = False
PLOT = False
INDEX = 10
data_folders = [f'{base_folder}/' + ('G' if GRID else 'H') + f'T{trial:02d}' for trial in trials]  # Directory that contains folders with recordings of each point of interest

if GRID:
    POINTS_FRE = ['01', '03', '05', '11', '13', '15', '21', '23', '25']
    # POINTS_TRE = ['01', '03', '05', '11', '13', '15', '21', '23', '25']
    POINTS_TRE = ['02', '04', '06', '07', '08', '09', '10', '12', '14', '16', '17', '18', '19', '20', '22', '24']
elif FLIP:
    POINTS_FRE = ['R1', 'R2', 'R3', 'R5', '09', 'R7', '06', 'R9']
    POINTS_TRE = ['02', '03', '04', '05', 'R8', '07', '08', 'R6', '10']
else:
    POINTS_FRE = ['R1', 'R2', 'R3', 'R5', 'R6', 'R7', 'R8', 'R9']
    POINTS_TRE = ['02', '03', '04', '05', '06', '07', '08', '09', '10']
    # POINTS_TRE = ['04']

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

# Load grid trajectories (ground truth for trajectories)
landmark_front_trajectory = np.load('../Registration/Landmark/Landmark Trajectories/front_trajectory.npy')
landmark_back_trajectory = np.load('../Registration/Landmark/Landmark Trajectories/back_trajectory.npy')
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
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(real_fiducial_points[:, 0], real_fiducial_points[:, 1], real_fiducial_points[:, 2], 
               c='r', marker='o', label='Real Fiducial Points')
    ax.scatter(transformed_fiducial_points[:, 0], transformed_fiducial_points[:, 1], transformed_fiducial_points[:, 2], 
               c='b', marker='s', label='Measured Fiducial Points')
    ax.scatter(real_target_points[:, 0], real_target_points[:, 1], real_target_points[:, 2], 
               c='r', marker='o', label='Real Target Points', edgecolors='black')
    ax.scatter(transformed_target_points[:, 0], transformed_target_points[:, 1], transformed_target_points[:, 2], 
               c='b', marker='s', label='Measured Target Points', edgecolors='black')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend(loc='lower right')
    return fig

def plot_3d_trajectories(real_trajectories, measured_trajectories, title, distance_threshold=10):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    def is_consecutive(p1, p2):
        return np.linalg.norm(p1 - p2) < distance_threshold
    real_trajectories = [traj[~np.isnan(traj).any(axis=1)] for traj in real_trajectories]
    measured_trajectories = [traj[~np.isnan(traj).any(axis=1)] for traj in measured_trajectories]
    # Plot real trajectories with different markers for each
    for i, traj in enumerate(real_trajectories):
        if traj.shape[0] < 2:
            continue
        marker = ['o','^','s','D','*','p'][i % 6]
        label = f"Real Trajectory {i+1}"
        for j in range(len(traj)-1):
            if is_consecutive(traj[j], traj[j+1]):
                ax.plot(traj[j:j+2, 0], traj[j:j+2, 1], traj[j:j+2, 2],
                        c='r', marker=marker, markersize=5, label=label if j==0 else "")
    # Plot measured trajectories similarly
    for i, traj in enumerate(measured_trajectories):
        if traj.shape[0] < 2:
            continue
        marker = ['o','^','s','D','*','p'][i % 6]
        label = f"Measured Trajectory {i+1}"
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                c='b', marker=marker, markersize=5, linestyle='-', label=label)
    all_points = np.concatenate(real_trajectories + measured_trajectories)
    ax.set_xlim([all_points[:, 0].min() - 10, all_points[:, 0].max() + 10])
    ax.set_ylim([all_points[:, 1].min() - 10, all_points[:, 1].max() + 10])
    ax.set_zlim([all_points[:, 2].min() - 10, all_points[:, 2].max() + 10])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend(loc='lower right')
    return fig

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
    
    avg_fre_ppe = [np.mean(errors) if errors else 0 for errors in fre_errors_per_point]
    std_fre_ppe = [np.std(errors) if errors else 0 for errors in fre_errors_per_point]
    avg_tre_ppe = [np.mean(errors) if errors else 0 for errors in tre_errors_per_point]
    std_tre_ppe = [np.std(errors) if errors else 0 for errors in tre_errors_per_point]
    print("\nAverage Per-Point FRE Across All Configs [mm]:")
    for i, (avg_error, std_error) in enumerate(zip(avg_fre_ppe, std_fre_ppe)):
        print(f"Point {i+1}: {avg_error:.4f} (±{std_error:.4f})")
    print("\nAverage Per-Point TRE Across All Configs [mm]:")
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

    # Calculate average per-point errors
    avg_fre_ppe = [np.mean(errors) if errors else 0 for errors in fre_errors_per_point]
    std_fre_ppe = [np.std(errors) if errors else 0 for errors in fre_errors_per_point]
    avg_tre_ppe = [np.mean(errors) if errors else 0 for errors in tre_errors_per_point]
    std_tre_ppe = [np.std(errors) if errors else 0 for errors in tre_errors_per_point]

    print("\nAverage Per-Point FRE Across All Configs [mm]:")
    for i, (avg_error, std_error) in enumerate(zip(avg_fre_ppe, std_fre_ppe)):
        print(f"Point {points_FRE[i]}: {avg_error:.4f} (±{std_error:.4f})")

    print("\nAverage Per-Point TRE Across All Configs [mm]:")
    for i, (avg_error, std_error) in enumerate(zip(avg_tre_ppe, std_tre_ppe)):
        print(f"Point {points_TRE[i]}: {avg_error:.4f} (±{std_error:.4f})")

    # Collect results into DataFrames
    camera_configs = sorted(set(key.split(' Cam ')[0] for key in results['FRE'].keys()))
    fusion_methods = sorted(set(key.split(' Cam ')[1] for key in results['FRE'].keys()))

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
                    title = f"Trajectory Registration for {key}"
                    if back_untracked > 0 or front_untracked > 0 or rt1_untracked > 0 or rt2_untracked > 0 or rt3_untracked > 0 or rt4_untracked > 0:
                        title += f"\nBack Points Untracked: {back_untracked:.2f}%, Front Points Untracked: {front_untracked:.2f}%, RT1 Points Untracked: {rt1_untracked:.2f}%, RT2 Points Untracked: {rt2_untracked:.2f}%, RT3 Points Untracked: {rt3_untracked:.2f}%, RT4 Points Untracked: {rt4_untracked:.2f}%"

                    # Prepare trajectories for plotting
                    real_trajectories = [landmark_back_trajectory, landmark_front_trajectory, trajectory1, trajectory2, trajectory3, trajectory4]
                    measured_trajectories = [trajectories.get('back_trajectory'), trajectories.get('front_trajectory'), trajectories.get('rt1_trajectory'), trajectories.get('rt2_trajectory'), trajectories.get('rt3_trajectory'), trajectories.get('rt4_trajectory')]

                    # Handle cases where trajectories might be None
                    for i in range(6):
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

                except Exception as e:
                    print(f"Error unpacking or plotting data for key {key}: {e}")
            else:
                print(f"Skipping key {key} due to missing data.")
    
    # return fre_rmse, tre_rmse, back_tre_rmse, front_tre_rmse, rt1_tre_rmse, rt2_tre_rmse, rt3_tre_rmse, rt4_tre_rmse

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
    default_pose = [1.956366119384765625e+02, -6.101075363159179688e+01, -2.138027000427246094e+01]
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
                title = "Trajectory Registration"
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
    
    return fre_ppe, tre_ppe

def aggregate_results(data_folder, configs=['top', 'left', 'right']):
    """Aggregate the results for all configurations and Atracsys into one final table."""
    metrics = ['FRE', 'TRE', 'Back Trajectory TRE', 'Front Trajectory TRE', 'RT1 Trajectory TRE', 'RT2 Trajectory TRE', 'RT3 Trajectory TRE', 'RT4 Trajectory TRE', 'Total Trajectory TRE']
    aggregated_data = {metric: {i: {method: [] for method in [
            'Mono', 'Mono Kalman', 'Mono Kalman Adaptive', 
            'Stereo', 'Stereo Kalman', 'Stereo Kalman Adaptive'
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
            'Mono', 'Mono Kalman', 'Mono Kalman Adaptive', 
            'Stereo', 'Stereo Kalman', 'Stereo Kalman Adaptive'
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

def aggregate_results_all(data_folders, configs=['top', 'left', 'right']):
    """Aggregate the results for all configurations and Atracsys into one final table."""
    metrics = ['FRE', 'TRE', 'Back Trajectory TRE', 'Front Trajectory TRE', 'RT1 Trajectory TRE', 'RT2 Trajectory TRE', 'RT3 Trajectory TRE', 'RT4 Trajectory TRE', 'Total Trajectory TRE']
    aggregated_data = {metric: {i: {method: [] for method in [
            'Mono', 'Mono Kalman', 'Mono Kalman Adaptive', 
            'Stereo', 'Stereo Kalman', 'Stereo Kalman Adaptive'
        ]} for i in range(1, 6)} for metric in metrics}  # Separate structures for each metric
    
    atracsys_results = {metric: [] for metric in metrics}
    
    for data_folder in data_folders:
        # Load data from each configuration
        trial_name = os.path.basename(data_folder).split('_')[0]

        for config in configs:
            config_folder = os.path.join(data_folder, f'_{trial_name}_{config.upper()}')
            json_file = os.path.join(config_folder, f"{os.path.basename(config_folder)}_results.json")
            if os.path.exists(json_file):
                with open(json_file, 'r') as file:
                    config_results = json.load(file)
                
                # Iterate through each metric separately
                for metric in metrics:
                    if metric in config_results:
                        for method, results in config_results[metric].items():
                            if isinstance(results, dict):
                                for camera_num, value in results.items():
                                    camera_num = int(camera_num)
                                    numeric_value = extract_numeric_value(value)
                                    if isinstance(numeric_value, (int, float)) and not np.isnan(numeric_value):
                                        aggregated_data[metric][camera_num][method].append(numeric_value)
        
        # Atracsys results
        atracsys_json_file = os.path.join(data_folder, f"{os.path.basename(data_folder)}_results_atracsys.json")
        if os.path.exists(atracsys_json_file):
            with open(atracsys_json_file, 'r') as file:
                atracsys_json = json.load(file)
                for metric in metrics:
                    numeric_value = extract_numeric_value(atracsys_json[metric])
                    atracsys_results[metric].append(numeric_value)
    
    # Create DataFrame for final results
    final_tables = {}
    for metric in metrics:
        df = pd.DataFrame(index=range(1, 6), columns=[
            'Mono', 'Mono Kalman', 'Mono Kalman Adaptive', 
            'Stereo', 'Stereo Kalman', 'Stereo Kalman Adaptive'
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

    return final_tables

def extract_numeric_value(value):
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return np.nan
    return value

def calculate_avg_and_std(data):
    numeric_data = [float(value) for value in data if isinstance(value, (int, float)) or (isinstance(value, str) and value.replace('.', '', 1).isdigit())]
    if len(numeric_data) == 0:
        return "N/A"
    avg = np.mean(numeric_data)
    std = np.std(numeric_data)
    return f"{avg:.4f} (±{std:.4f})"

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

def run_trial(data_folder, points_FRE, points_TRE):
    """Run the results table calculations for top, left, and right configurations and Atracsys."""
    trial_name = os.path.basename(data_folder).split('_')[0]
    configs = ['top', 'left', 'right']
    
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
        results_table(config_folder, points_FRE, points_TRE, mean=MEAN, plot3D=PLOT, savePlots=False)
    
    # Step 2: Calculate Atracsys results
    print(f"\nCalculating results for Atracsys...")
    results_atracsys(data_folder, points_FRE, points_TRE, mean=MEAN, plot3D=PLOT)
    
    # Step 3: Aggregate results into final table
    final_results = aggregate_results(data_folder, configs=configs)
    
    # Step 4: Save and display final aggregated results
    save_final_results(data_folder, final_results)

# %%
if __name__ == "__main__":
    for data_folder in data_folders:
        configs = ['top', 'left', 'right']
        run_trial(data_folder, POINTS_FRE, POINTS_TRE)
        
    final_results = aggregate_results_all(data_folders, configs=configs)
    # Display the results
    print(f"\nAggregated Results for Trials {trials}:\n")
    for metric, df in final_results.items():
        print(f"{metric} RMSE [mm]:")
        print(df.to_string(index=True, justify='center'))
        print("")
    
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
                    fre_ppe, tre_ppe = results_atracsys(data_folder, points_FRE, point_TRE, mean=True)
                    tre_all[points_all[i]][trialIndex] = tre_ppe[0]
            for point in points_all:
                print(f'{point}: {tre_all[point][trialIndex]:.4f}')
        
        tre_avg = dict([(point, np.mean(tre_all[point])) for point in points_all])
        tre_std = dict([(point, np.std(tre_all[point])) for point in points_all])
        print('\nAverage of trials:')
        for point in points_all:
            print(f'{point}: {tre_avg[point]:.4f} (±{tre_std[point]:.4f})')