import h5py
import glob
import json
import os
import math
import pickle

import blosc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.signal import argrelextrema
import torch
import torch.nn.functional as F
import PIL.Image as Image
import cv2
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple, Literal, Dict, Optional
import transformers
from tqdm import tqdm




TextEncoder = Literal["bert", "clip"]

def find_gripper_keypoints(gripper_position_state):
    """
    Go through the gripper positions in sequence and find the keypoints that meet the following criterion:
    if gripper_position_state[0] < 0.1, gripper starts from open
    We look for the first time the gripper is closed, which needs to satisfy: gripper_position_state[i] > 0.5 and abs(mean(gripper_position_state[i:i+5]) - mean(gripper_position_state[i+5:i+10])) < 0.05 and mean(gripper_position_state[i+10:i+15]) > 0.5
    record that as the first keypoint
    Next, we look for the first time the gripper is open, which needs to satisfy: gripper_position_state[i] < 0.5 and abs(mean(gripper_position_state[i:i+5]) - mean(gripper_position_state[i+5:i+10])) < 0.05 and mean(gripper_position_state[i+10:i+15]) < 0.5
    record that as the second keypoint
    repeat until the end of the sequence
    """
    keypoints = []
    length = len(gripper_position_state)
    
    def mean_segment(start, end):
        """Compute mean of segment from start to end."""
        if end > length:
            return np.mean(gripper_position_state[-(end - start):])
        return np.mean(gripper_position_state[start:end])
    
    def find_keypoint(start_pos=0, open_to_close=True):
        """Find the next keypoint based on open_to_close flag."""
        for i in range(start_pos, length):
            if open_to_close:
                if gripper_position_state[i] > 0.5 and \
                abs(mean_segment(i, i+5) - mean_segment(i+5, i+10)) < 0.02 and \
                mean_segment(i+10, i+15) > 0.5:
                    return i
            else:
                if gripper_position_state[i] < 0.5 and \
                abs(mean_segment(i, i+5) - mean_segment(i+5, i+10)) < 0.02 and \
                mean_segment(i+10, i+15) < 0.5:
                    return i
        return -1
    if len(gripper_position_state) < 15:
        return keypoints  # Not enough data to find keypoints
    # if gripper_position_state[0] < 0.1:
    # Gripper starts from open
    # print("Gripper starts from open")
    open_to_close = (gripper_position_state[0] < 0.1)
    keypoint_idx = 0
    while find_keypoint(keypoint_idx, open_to_close) != -1:
        keypoint_idx = find_keypoint(keypoint_idx, open_to_close)
        keypoints.append(keypoint_idx)
        print(f"{open_to_close} at {keypoint_idx}")
        open_to_close = not open_to_close
    
    
    return np.array(keypoints)

def load_demos_of_fname(file_name):
    with h5py.File(file_name, 'r+') as fid:
        demo = {}
        for k in fid['data']['demo_0'].keys():
            if hasattr(fid['data/demo_0'][k], 'keys'):
                demo[str(k)] = {
                    k_local: np.array(fid['data/demo_0'][k][k_local])
                    for k_local in fid['data/demo_0'][k].keys()
                }
            else:
                demo[str(k)] = np.array(fid['data/demo_0'][k])
    return demo


def depth2cloud(depth_img, intrinsics, extrinsics):
    # depth_img: (H, W), instrinsics (3, 3), extrinsics (4, 4)
    im_height, im_width = depth_img.shape
    ww = np.linspace(0, im_width - 1, im_width)
    hh = np.linspace(0, im_height - 1, im_height)
    xmap, ymap = np.meshgrid(ww, hh)
    points_2d = np.column_stack((xmap.ravel(), ymap.ravel()))

    depth_img = depth_img.flatten()

    homogenous = np.pad(points_2d, ((0, 0), (0, 1)), constant_values=1.0)
    points_in_camera_axes = np.matmul(
        np.linalg.inv(intrinsics),
        homogenous.T * depth_img[None]
    )
    points_in_camera_axes_homogenous = np.pad(points_in_camera_axes, ((0, 1), (0, 0)), constant_values=1.0)
    points_in_world_frame_homogenous = np.matmul(
        np.linalg.inv(extrinsics), points_in_camera_axes_homogenous
    )
    return points_in_world_frame_homogenous[:3, :]


def axisangle2quat(vec):
    """
    Converts scaled axis-angle to quat.

    Args:
        vec (np.array): (ax,ay,az) axis-angle exponential coordinates

    Returns:
        np.array: (x,y,z,w) vec4 float angles
    """
    # Grab angle
    angle = np.linalg.norm(vec)

    # handle zero-rotation case
    if math.isclose(angle, 0.0):
        return np.array([0.0, 0.0, 0.0, 1.0])

    # make sure that axis is a unit vector
    axis = vec / angle

    q = np.zeros(4)
    q[3] = np.cos(angle / 2.0)
    q[:3] = axis * np.sin(angle / 2.0)
    return q


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))



def get_eef_velocity_from_trajectories(trajectories):
    trajectories = np.stack([trajectories[0]] + trajectories, axis=0)
    velocities = trajectories[1:] - trajectories[:-1]

    V = np.linalg.norm(velocities[:, :3], axis=-1)
    W = np.linalg.norm(velocities[:, 3:6], axis=-1)

    velocities = np.concatenate(
        [velocities, [velocities[-1]]],
        # [velocities[[0]], velocities],
        axis=0
    )
    accelerations = velocities[1:] - velocities[:-1]

    A = np.linalg.norm(accelerations[:, :3], axis=-1)

    return V, W, A


def gripper_state_changed(trajectories):
        return find_gripper_keypoints(trajectories[:, -1])



def get_camera_extrinsic_matrix(calibration_6d):
    calibration_matrix = np.array(calibration_6d)
    cam_pose = calibration_matrix[:3]
    cam_euler = calibration_matrix[3:]
    rotation_matrix = R.from_euler("xyz", cam_euler).as_matrix()
    extrinsic_matrix = np.hstack((rotation_matrix, cam_pose.reshape(3, 1)))
    extrinsic_matrix = np.vstack((extrinsic_matrix, np.array([0, 0, 0, 1])))
    return extrinsic_matrix


def cam_pointcloud_to_world_pointcloud(cam_pointcloud, extrinsics):
    """Converts a point cloud from the camera frame to the world frame.
    cam_pointcloud: (H, W, 3) -> (3, N)
    extrinsics: (4, 4)
    return: (3, H, W)
    """
    H, W = cam_pointcloud.shape[:2]
    cam_pointcloud = cam_pointcloud.reshape(-1, 3).T
    return (np.dot(extrinsics[:3, :3], cam_pointcloud) + extrinsics[:3, 3][:, None]).reshape(3, H, W)

def euler_to_quaternion(euler_angles):
    # Create a Rotation object
    rotations = R.from_euler('XYZ', euler_angles)

    # Extract the quaternions
    quaternions = rotations.as_quat() # (X, Y, Z, W)
    return quaternions


def keypoint_discovery(trajectories, scene_states=None, task=None,
                       buffer_size=5):
    """Determine way point from the trajectories.

    Args:
        trajectories: a list of 1-D np arrays.  Each array is
            7-dimensional (x, y, z, euler_x, euler_y, euler_z, opene).
        stopping_delta: the minimum velocity to determine if the
            end effector is stopped.

    Returns:
        an Integer array indicates the indices of waypoints
    """
    V, W, A = get_eef_velocity_from_trajectories(trajectories)

    # waypoints are local minima of gripper movement
    _local_max_A = argrelextrema(A, np.greater)[0]
    topK = np.sort(A)[::-1][int(A.shape[0] * 0.2)]
    large_A = A[_local_max_A] >= topK
    _local_max_A = _local_max_A[large_A].tolist()

    local_max_A = [_local_max_A.pop(0)]
    for i in _local_max_A:
        if i - local_max_A[-1] >= buffer_size:
            local_max_A.append(i)

    # waypoints are frames with changing gripper states
    gripper_changed = gripper_state_changed(trajectories)
    one_frame_before_gripper_changed = (
        gripper_changed[gripper_changed > 1] - 1
    )

    # waypoints is the last pose in the trajectory
    last_frame = [len(trajectories) - 1]

    keyframe_inds = (
        # local_max_A +
        gripper_changed.tolist() +
        # one_frame_before_gripper_changed.tolist() +
        last_frame
    )
    keyframe_inds = np.unique(keyframe_inds)

    keyframes = [trajectories[i] for i in keyframe_inds]

    return keyframes, keyframe_inds


def interpolate_fill_inf(arr):
    ok = ~np.isinf(arr)
    xp = ok.ravel().nonzero()[0]
    fp = arr[~np.isinf(arr)]
    x  = np.isinf(arr).ravel().nonzero()[0]

    # Replacing nan values
    arr[np.isinf(arr)] = np.interp(x, xp, fp)
    return arr

def interpolate_fill_nan(arr):
    ok = ~np.isnan(arr)
    xp = ok.ravel().nonzero()[0] # indices of non-nan values
    fp = arr[~np.isnan(arr)]
    x  = np.isnan(arr).ravel().nonzero()[0]

    # Replacing nan values
    try:
        arr[np.isnan(arr)] = np.interp(x, xp, fp)
    except:
        print(np.isnan(arr).all())
        print(xp)
        print(fp)
        breakpoint()
    return arr


def convert23dda(traj, languages, tokenizer, model):
    # # Extract keyposes
    # _, k_ids = keypoint_discovery(demo['actions_abs'])

    # Convert to quaternion
    # actions_abs = np.stack([
    #     np.concatenate((action[:3], axisangle2quat(action[3:6]), action[6:]))
    #     for action in demo['actions_abs']
    # ])

    cartesian_position_state = traj['observation']['robot_state']['cartesian_position'][()]
    gripper_position_state = traj['observation']['robot_state']['gripper_position'][()]
    try:
        cartesian_position_state = np.concatenate(
            [cartesian_position_state[:, :3], euler_to_quaternion(cartesian_position_state[:, 3:6]), gripper_position_state[:, None]],
            axis=-1
        )
    except:
        print(cartesian_position_state.shape) # (0, 6)
        breakpoint()
    actions_abs = np.concatenate((traj["action/abs_pos"][()], 
                                  euler_to_quaternion(traj['action/abs_rot_euler'][()]),
                                  traj['action/gripper_position'][()]), axis=1)


    wrist_cam_left_extrinsics = traj['observation']['camera']['extrinsics']['hand_camera_left'][()]
    # wrist_cam_right_extrinsics = traj['observation']['camera']['extrinsics']['hand_camera_right'][()]
    ext1_cam_left_extrinsics = traj['observation']['camera']['extrinsics']['varied_camera_1_left'][()]
    # ext1_cam_right_extrinsics = traj['observation']['camera']['extrinsics']['varied_camera_1_right'][()]
    ext2_cam_left_extrinsics = traj['observation']['camera']['extrinsics']['varied_camera_2_left'][()]
    # ext2_cam_right_extrinsics = traj['observation']['camera']['extrinsics']['varied_camera_2_right'][()]

    
    ext1_image_left = traj['observation']['camera']['image']['varied_camera_1_left_image'][()]
    ext1_pcd_left = traj['observation']['camera']['pointcloud']['varied_camera_1_left_image'][()]

    # ext1_image_right = traj['observation']['camera']['image']['varied_camera_1_right_image'][()]

    ext2_image_left = traj['observation']['camera']['image']['varied_camera_2_left_image'][()]
    ext2_pcd_left = traj['observation']['camera']['pointcloud']['varied_camera_2_left_image'][()]

    # ext2_image_right = traj['observation']['camera']['image']['varied_camera_2_right_image'][()]

    wrist_image_left = traj['observation']['camera']['image']['hand_camera_left_image'][()]
    wrist_pcd_left = traj['observation']['camera']['pointcloud']['hand_camera_left_image'][()]

    # wrist_image_right = traj['observation']['camera']['image']['hand_camera_right_image'][()]

    wrist_pcd_left = interpolate_fill_inf(wrist_pcd_left)
    ext1_pcd_left = interpolate_fill_inf(ext1_pcd_left)
    ext2_pcd_left = interpolate_fill_inf(ext2_pcd_left)
    if np.isnan(wrist_pcd_left).all():
        breakpoint()
    elif np.isnan(ext1_pcd_left).all():
        breakpoint()
    elif np.isnan(ext2_pcd_left).all():
        breakpoint()
    try:
        wrist_pcd_left = interpolate_fill_nan(wrist_pcd_left)
        ext1_pcd_left = interpolate_fill_nan(ext1_pcd_left)
        ext2_pcd_left = interpolate_fill_nan(ext2_pcd_left)
    except:
        print(np.isnan(wrist_pcd_left).any())
        print(np.isnan(ext1_pcd_left).any())
        print(np.isnan(ext2_pcd_left).any())
        breakpoint()
    assert not np.isinf(wrist_pcd_left).any()
    assert not np.isinf(ext1_pcd_left).any()
    assert not np.isinf(ext2_pcd_left).any()


    # Extract keyposes
    _, k_ids = keypoint_discovery(cartesian_position_state)

    # Frame_ids
    frame_ids = np.arange(len(k_ids))

    # Observation numpy arrays
    obs_arrays = []
    languages_embedding_arrays = []
    for k in [0] + k_ids.tolist()[:-1]:
        # World
        ext1_color = ext1_image_left[k] / 255.0 # (H, W, 3)
        ext2_color = ext2_image_left[k] / 255.0 # (H, W, 3)
        ext1_pose_left = get_camera_extrinsic_matrix(ext1_cam_left_extrinsics[k])
        ext2_pose_left = get_camera_extrinsic_matrix(ext2_cam_left_extrinsics[k])
        ext1_points_in_world_frame_homogenous_pcd = cam_pointcloud_to_world_pointcloud(ext1_pcd_left[k], ext1_pose_left) # (3, H, W)
        ext2_points_in_world_frame_homogenous_pcd = cam_pointcloud_to_world_pointcloud(ext2_pcd_left[k], ext2_pose_left) # (3, H, W)
        ext1_obs = np.stack((ext1_color.transpose(2, 0, 1), ext1_points_in_world_frame_homogenous_pcd))
        ext2_obs = np.stack((ext2_color.transpose(2, 0, 1), ext2_points_in_world_frame_homogenous_pcd))
        

        # Hand
        obs_hand = wrist_image_left[k] / 255.0 # (H, W, 3)
        wrist_pose_left = get_camera_extrinsic_matrix(wrist_cam_left_extrinsics[k])
        wrist_points_in_world_frame_homogenous_pcd = cam_pointcloud_to_world_pointcloud(wrist_pcd_left[k], wrist_pose_left) # (3, H, W)
        obs_hand = np.stack((obs_hand.transpose(2, 0, 1), wrist_points_in_world_frame_homogenous_pcd))  
        assert not np.isnan(obs_hand).any()
        assert not np.isnan(ext1_obs).any()
        assert not np.isnan(ext2_obs).any()
        
        # Combine
        obs = np.stack((ext1_obs, ext2_obs, obs_hand))
        obs_arrays.append(obs)
    obs_arrays = np.stack(obs_arrays)
    assert not np.isnan(obs_arrays).any()

    # Action tensors (will not be used)
    action_tensors = np.stack([actions_abs[k] for k in k_ids])

    # Camera dicts (not used)
    camera_dicts = []

    # Gripper tensors
    gripper_tensors = np.stack([
        cartesian_position_state[k] for k in [0] + k_ids.tolist()[:-1]
    ])

    # Trajectories
    trajectories = [
        cartesian_position_state[start:end + 1]
        for start, end in zip([0] + k_ids.tolist()[:-1], k_ids.tolist())
    ]
    
    for instr in languages:
        tokens = tokenizer(instr, padding="max_length")["input_ids"]

        tokens = torch.tensor(tokens).to("cuda")
        print(tokens.shape)
        if tokens.shape[0] > 53:
            print(tokens.shape)
            tokens = tokens[:53]
        tokens = tokens.view(1, -1)
        with torch.no_grad():
            try:
                pred = model(tokens).last_hidden_state.cpu().numpy()
            except:
                print(tokens.shape)
                breakpoint()
        languages_embedding_arrays.append(pred)
    try:
        languages_embedding_arrays = np.stack(languages_embedding_arrays)
    except:
        print(languages_embedding_arrays)
        print([x.shape for x in languages_embedding_arrays])
        breakpoint()

    # Store
    state_list = [
        frame_ids,
        obs_arrays,
        action_tensors,
        camera_dicts,
        gripper_tensors,
        trajectories,
        languages,
        languages_embedding_arrays
    ]
    return state_list


def load_model(encoder: TextEncoder) -> transformers.PreTrainedModel:
    if encoder == "bert":
        model = transformers.BertModel.from_pretrained("bert-base-uncased")
    elif encoder == "clip":
        model = transformers.CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    else:
        raise ValueError(f"Unexpected encoder {encoder}")
    if not isinstance(model, transformers.PreTrainedModel):
        raise ValueError(f"Unexpected encoder {encoder}")
    return model


def load_tokenizer(encoder: TextEncoder) -> transformers.PreTrainedTokenizer:
    if encoder == "bert":
        tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    elif encoder == "clip":
        tokenizer = transformers.CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
    else:
        raise ValueError(f"Unexpected encoder {encoder}")
    if not isinstance(tokenizer, transformers.PreTrainedTokenizer):
        raise ValueError(f"Unexpected encoder {encoder}")
    return tokenizer

def main(path):
    # find if there is a "mark.txt" file in the same directory as the path
    mark_file = glob.glob(os.path.join(os.path.dirname(path), "mark.txt"))
    if len(mark_file) > 0:
        # load and check if it contains "success"
        with open(mark_file[0], "r") as f:
            mark = f.readline().strip()
        if mark == "success":
            # dataset already processed, skip
            return
        
    cnt = 0
    
    min_ = 300 * np.ones(3)
    max_ = -300 * np.ones(3)
    rel_min_ = 300 * np.ones(3)
    rel_max_ = -300 * np.ones(3)
    
    with open("/lustre/fsw/portfolios/nvr/users/lawchen/project/droid/droid/aggregated-annotations-030724.json", 'r') as file:
        annotations = json.load(file)
    
    tokenizer = load_tokenizer("clip")
    tokenizer.model_max_length = 53 #args.model_max_length

    model = load_model("clip")
    model = model.to("cuda")

    
    
    # get dir
    save_pth = os.path.dirname(path)
    os.makedirs(save_pth, exist_ok=True)
    
    metadata_files = glob.glob(os.path.join(os.path.dirname(path), "metadata*.json"))
    metadata_file = metadata_files[0]
    # extract "TRI+52ca9b6a+2023-11-07-15h-30m-09s" from "metadata_TRI+52ca9b6a+2023-11-07-15h-30m-09s.json"
    metadata_name = os.path.basename(metadata_file).split(".")[0]
    metadata_name = metadata_name.split("_")[1]
    assert metadata_name in annotations.keys()

    languages = list(set(annotations[metadata_name].values()))

    with h5py.File(path, 'r') as traj:
        if traj['observation']['robot_state']['cartesian_position'][()].shape[0] == 0:
            return
        if 'abs_pos' not in traj['action']:
            return
        if np.isnan(traj['observation']['camera']['pointcloud']['hand_camera_left_image'][()]).all():
            print("handcam all nan:", path)
            return
        if np.isnan(traj['observation']['camera']['pointcloud']['varied_camera_1_left_image'][()]).all():
            print("cam1 all nan:", path)
            return
        if np.isnan(traj['observation']['camera']['pointcloud']['varied_camera_2_left_image'][()]).all():
            print("cam2 all nan:", path)
            return
        # Convert to 3DDA data format
        state_list = convert23dda(traj, languages, tokenizer, model)
        if state_list is None:
            print(f"Failed to convert {path}")
            return
        ep_save_path = f'{save_pth}/sparse_trajectory.dat'
        with open(ep_save_path, "wb") as f:
            f.write(blosc.compress(pickle.dumps(state_list)))

        # Boundaries
        all_actions = np.concatenate(state_list[5])  # trajectories
        min_ = np.minimum(min_, all_actions[:, :3].min(0))
        max_ = np.maximum(max_, all_actions[:, :3].max(0))
        # Boundaries for relative actions
        all_actions = np.concatenate([
            traj - traj[0][None]
            for traj in state_list[5]
        ])
        rel_min_ = np.minimum(rel_min_, all_actions[:, :3].min(0))
        rel_max_ = np.maximum(rel_max_, all_actions[:, :3].max(0))
        cnt += 1
    # Store boundaries
    with open(f'{save_pth}/location_bounds.json', 'w') as fid:
        json.dump({'pickup': [min_.tolist(), max_.tolist()]}, fid)
    with open(f'{save_pth}/rel_location_bounds.json', 'w') as fid:
        json.dump({'pickup': [rel_min_.tolist(), rel_max_.tolist()]}, fid)



    # mark the dataset as processed
    with open(os.path.join(os.path.dirname(path), "mark.txt"), "w") as f:
        f.write("success")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--folder",
        type=str,
        help="folder containing hdf5's to add camera images to",
        default="/lustre/fsw/portfolios/nvr/users/lawchen/droid_raw/droid_raw/droid_with_pcd_20k"
    )


    parser.add_argument(
        "--cuda_device", 
        type=int, 
        default=0
    )
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)

    # path_list = ["/home/lawchen/project/droid/data/test/Thu_Nov__2_21:23:36_2023/trajectory_im_pcd.h5"]
    datasets = []

    for root, dirs, files in os.walk(os.path.expanduser(args.folder)):
        for f in files:
            if f == "trajectory_im_pcd.h5":
                datasets.append(os.path.join(root, f))
    print("converting datasets...")
    # randomly shuffle the datasets
    np.random.shuffle(datasets)
    for d in tqdm(datasets):
        d = os.path.expanduser(d)
        # print(d)
        main(d)
