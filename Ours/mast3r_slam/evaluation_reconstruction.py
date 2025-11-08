import os
import numpy as np
import open3d as o3d
import cv2
from tqdm import tqdm
import torch
from scipy.spatial import cKDTree
import argparse

# Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_keyframe_indices(res_pose_file):
    """Return a list of integers corresponding to keyframe indices."""
    indices = []
    with open(res_pose_file, "r") as f:
        for line in f:
            idx = int(float(line.strip().split()[0]))  # first column is frame index
            indices.append(idx)
    return indices


def load_camera_poses(gt_pose_file):
    """Load ground-truth camera poses (7-DoF: timestamp, tx, ty, tz, qx, qy, qz, qw)."""
    poses = []
    with open(gt_pose_file, 'r') as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) == 8:
                t = np.array(vals[1:4])
                qx, qy, qz, qw = vals[4:]
                R = o3d.geometry.get_rotation_matrix_from_quaternion([qw, qx, qy, qz])
                T = np.eye(4)
                T[:3, :3], T[:3, 3] = R, t
                poses.append(T)
    return poses


def backproject_depth_to_points(depth, K, T_WC):
    """Back-project a depth map into 3D world coordinates using pose T_WC."""
    h, w = depth.shape
    i, j = torch.meshgrid(torch.arange(w, device=device), torch.arange(h, device=device))
    z = torch.tensor(depth.flatten(), dtype=torch.float32, device=device) / 1000.0  # Convert mm → m if applicable
    x = (i.flatten() - K[0, 2]) * z / K[0, 0]
    y = (j.flatten() - K[1, 2]) * z / K[1, 1]
    pts_cam = torch.stack([x, y, z], dim=1)
    pts_cam = pts_cam[z > 0]

    # Apply transformation to world coordinates
    pts_world = (T_WC[:3, :3] @ pts_cam.T + T_WC[:3, 3:4]).T
    return pts_world.cpu().numpy()


def build_reference_pcd_keyframes(depth_dir, gt_pose_file, keyframe_indices, K):
    """Back-project only keyframes to build reference point cloud."""
    poses = load_camera_poses(gt_pose_file)

    # Use a list to store points on the GPU (use a list to avoid reallocation issues)
    all_points = []

    for idx in tqdm(keyframe_indices, desc="Back-projecting keyframes"):
        depth_file = os.path.join(depth_dir, f"depth{idx:06d}.png")
        if not os.path.exists(depth_file):
            print(f"Warning: {depth_file} does not exist, skipping.")
            continue
        depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        depth_tensor = torch.tensor(depth, dtype=torch.float32, device=device)
        pts = backproject_depth_to_points(depth_tensor, K, poses[idx])
        all_points.append(pts)

    all_points = torch.cat([torch.tensor(pts, dtype=torch.float32, device=device) for pts in all_points], dim=0)
    all_points_cpu = all_points.cpu().numpy()

    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(all_points_cpu))


def voxel_down_sample_gpu(pcd, voxel_size):
    """Downsample point cloud on the GPU."""
    points = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device='cuda')
    # Create a grid based on voxel size
    voxel_grid = (points / voxel_size).floor()
    # Keep unique voxel grid points
    unique_voxels, inverse_indices = torch.unique(voxel_grid, dim=0, return_inverse=True)
    downsampled_points = points[unique_voxels]

    return downsampled_points


# Use it for your ICP function:
def align_icp(source_pcd, target_pcd, voxel_size=0.02):
    """Align two point clouds using ICP with GPU-accelerated downsampling."""
    source_points = voxel_down_sample_gpu(source_pcd, voxel_size)
    target_points = voxel_down_sample_gpu(target_pcd, voxel_size)

    # ICP registration logic using GPU-based downsampled point clouds
    threshold = 0.1
    reg = o3d.pipelines.registration.registration_icp(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_points.cpu().numpy())),
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_points.cpu().numpy())),
        threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    source_pcd.transform(reg.transformation)
    return source_pcd, reg.transformation

def chamfer_metrics(pts_est, pts_ref, threshold=0.5):
    """
    Compute accuracy, completion, and Chamfer distance between two point sets.
    Both metrics are truncated by a 0.5 m maximum distance threshold
    and averaged across all points.
    """
    pts_est = torch.tensor(pts_est, dtype=torch.float32, device=device)
    pts_ref = torch.tensor(pts_ref, dtype=torch.float32, device=device)

    # Calculate distance using GPU (optimized by FAISS or custom search)
    d_est2ref = torch.norm(pts_est.unsqueeze(1) - pts_ref.unsqueeze(0), dim=2, p=2)
    d_ref2est = torch.norm(pts_ref.unsqueeze(1) - pts_est.unsqueeze(0), dim=2, p=2)

    # Apply the 0.5 m distance cap
    d_est2ref = torch.minimum(d_est2ref, threshold)
    d_ref2est = torch.minimum(d_ref2est, threshold)

    # Compute Chamfer distance
    accuracy = torch.sqrt(torch.mean(d_est2ref**2)).cpu().item()
    completion = torch.sqrt(torch.mean(d_ref2est**2)).cpu().item()
    chamfer = 0.5 * (accuracy + completion)
    return accuracy, completion, chamfer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dataset_path", required=True, help="Base dataset directory path")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g., room0_agent_0")
    parser.add_argument("--GT", required=True, help="Path to ground truth pose file")
    parser.add_argument("--ResDir", required=True, help="Directory containing reconstruction results")
    args = parser.parse_args()

    parts = args.dataset.split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid dataset name: {args.dataset}. Expected format 'scene_agent'.")

    scene, agent = parts
    depth_dir = os.path.join(args.base_dataset_path, scene, agent, "results")
    ResPointClouds = os.path.join(args.ResDir, "results.ply")
    ResPose = os.path.join(args.ResDir, "results.txt")

    print("Depth directory:", depth_dir)
    print("Ground truth file:", args.GT)
    print("Estimated reconstruction:", ResPointClouds)

    # Camera intrinsics (replace with actual calibration if known)
    K = None
    if "MA_Replica" in args.base_dataset_path:
        K = np.array([[600.0, 0, 599.5],
                      [0, 600.0, 339.5],
                      [0, 0, 1.0]])
    if "MA_ADT" in args.base_dataset_path:
        K = np.array([[280.0, 0, 255.5],
                      [0, 280.0, 255.5],
                      [0, 0, 1.0]])

    print("Building reference point cloud...")
    keyframe_indices = load_keyframe_indices(ResPose)
    ref_pcd = build_reference_pcd_keyframes(depth_dir, args.GT, keyframe_indices, K)
    print(f"Reference PCD contains {len(ref_pcd.points)} points.")

    print("Loading estimated reconstruction...")
    est_pcd = o3d.io.read_point_cloud(ResPointClouds)

    print("Aligning estimated reconstruction via ICP...")
    est_aligned, _ = align_icp(est_pcd, ref_pcd)
    print("ICP alignment complete.")

    print("Evaluating reconstruction (threshold = 0.5 m)...")
    accuracy, completion, chamfer = chamfer_metrics(
        np.asarray(est_aligned.points),
        np.asarray(ref_pcd.points),
        threshold=0.5
    )

    print(f"\n=== Evaluation Results ===")
    print(f"Accuracy (m):        {accuracy:.4f}")
    print(f"Completion (m):      {completion:.4f}")
    print(f"Chamfer Distance (m): {chamfer:.4f}")

# import os
# import numpy as np
# import open3d as o3d
# import cv2
# from tqdm import tqdm
# from scipy.spatial import cKDTree
# import argparse
#
# def load_keyframe_indices(res_pose_file):
#     """Return a list of integers corresponding to keyframe indices."""
#     indices = []
#     with open(res_pose_file, "r") as f:
#         for line in f:
#             idx = int(float(line.strip().split()[0]))  # first column is frame index
#             indices.append(idx)
#     return indices
#
# def load_camera_poses(gt_pose_file):
#     """Load ground-truth camera poses (7-DoF: timestamp, tx, ty, tz, qx, qy, qz, qw)."""
#     poses = []
#     with open(gt_pose_file, 'r') as f:
#         for line in f:
#             vals = list(map(float, line.strip().split()))
#             if len(vals) == 8:
#                 t = np.array(vals[1:4])
#                 qx, qy, qz, qw = vals[4:]
#                 R = o3d.geometry.get_rotation_matrix_from_quaternion([qw, qx, qy, qz])
#                 T = np.eye(4)
#                 T[:3, :3], T[:3, 3] = R, t
#                 poses.append(T)
#     return poses
#
# def backproject_depth_to_points(depth, K, T_WC):
#     """Back-project a depth map into 3D world coordinates using pose T_WC."""
#     h, w = depth.shape
#     i, j = np.meshgrid(np.arange(w), np.arange(h))
#     z = depth.flatten() / 1000.0  # Convert mm → m if applicable
#     x = (i.flatten() - K[0, 2]) * z / K[0, 0]
#     y = (j.flatten() - K[1, 2]) * z / K[1, 1]
#     pts_cam = np.stack([x, y, z], axis=1)
#     pts_cam = pts_cam[z > 0]
#     pts_world = (T_WC[:3, :3] @ pts_cam.T + T_WC[:3, 3:4]).T
#     return pts_world
#
# def build_reference_pcd_keyframes(depth_dir, gt_pose_file, keyframe_indices, K):
#     """Back-project only keyframes to build reference point cloud."""
#     poses = load_camera_poses(gt_pose_file)
#     all_points = []
#
#     for idx in tqdm(keyframe_indices, desc="Back-projecting keyframes"):
#         depth_file = os.path.join(depth_dir, f"depth{idx:06d}.png")
#         if not os.path.exists(depth_file):
#             print(f"Warning: {depth_file} does not exist, skipping.")
#             continue
#         depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
#         pts = backproject_depth_to_points(depth, K, poses[idx])
#         all_points.append(pts)
#
#     all_points = np.concatenate(all_points, axis=0)
#     return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(all_points))
#
# def align_icp(source_pcd, target_pcd, voxel_size=0.02):
#     """Align two point clouds using ICP."""
#     source_down = source_pcd.voxel_down_sample(voxel_size)
#     target_down = target_pcd.voxel_down_sample(voxel_size)
#     threshold = 0.1  # 10 cm for initial alignment
#     reg = o3d.pipelines.registration.registration_icp(
#         source_down, target_down, threshold, np.eye(4),
#         o3d.pipelines.registration.TransformationEstimationPointToPoint()
#     )
#     source_pcd.transform(reg.transformation)
#     return source_pcd, reg.transformation
#
# def chamfer_metrics(pts_est, pts_ref, threshold=0.5):
#     """
#     Compute accuracy, completion, and Chamfer distance between two point sets.
#     Both metrics are truncated by a 0.5 m maximum distance threshold
#     and averaged across all points.
#     """
#     tree_ref = cKDTree(pts_ref)
#     tree_est = cKDTree(pts_est)
#
#     d_est2ref, _ = tree_ref.query(pts_est, k=1)
#     d_ref2est, _ = tree_est.query(pts_ref, k=1)
#
#     # Apply the 0.5 m distance cap
#     d_est2ref = np.minimum(d_est2ref, threshold)
#     d_ref2est = np.minimum(d_ref2est, threshold)
#
#     # Root mean squared error under the truncation
#     accuracy = np.sqrt(np.mean(np.square(d_est2ref)))
#     completion = np.sqrt(np.mean(np.square(d_ref2est)))
#     chamfer = 0.5 * (accuracy + completion)
#     return accuracy, completion, chamfer
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--base_dataset_path", required=True, help="Base dataset directory path")
#     parser.add_argument("--dataset", required=True, help="Dataset name, e.g., room0_agent_0")
#     parser.add_argument("--GT", required=True, help="Path to ground truth pose file")
#     parser.add_argument("--ResDir", required=True, help="Directory containing reconstruction results")
#     args = parser.parse_args()
#
#     parts = args.dataset.split("_", 1)
#     if len(parts) != 2:
#         raise ValueError(f"Invalid dataset name: {args.dataset}. Expected format 'scene_agent'.")
#
#     scene, agent = parts
#     depth_dir = os.path.join(args.base_dataset_path, scene, agent, "results")
#     ResPointClouds= os.path.join(args.ResDir, "results.ply")
#     ResPose= os.path.join(args.ResDir, "results.txt")
#
#     print("Depth directory:", depth_dir)
#     print("Ground truth file:", args.GT)
#     print("Estimated reconstruction:", ResPointClouds)
#
#     # Camera intrinsics (replace with actual calibration if known)
#     K=None
#     if  "MA_Replica" in args.base_dataset_path:
#         K = np.array([[600.0, 0, 599.5],
#                       [0, 600.0, 339.5],
#                       [0,   0,    1.0]])
#     if "MA_ADT" in args.base_dataset_path:
#         K = np.array([[280.0, 0, 255.5],
#                       [0, 280.0, 255.5],
#                       [0,   0,    1.0]])
#
#     print("Building reference point cloud...")
#     keyframe_indices=load_keyframe_indices(ResPose)
#     ref_pcd = build_reference_pcd_keyframes(depth_dir, args.GT, keyframe_indices, K)
#     print(f"Reference PCD contains {len(ref_pcd.points)} points.")
#
#     print("Loading estimated reconstruction...")
#     est_pcd = o3d.io.read_point_cloud(ResPointClouds)
#
#     print("Aligning estimated reconstruction via ICP...")
#     est_aligned, _ = align_icp(est_pcd, ref_pcd)
#     print("ICP alignment complete.")
#
#     print("Evaluating reconstruction (threshold = 0.5 m)...")
#     accuracy, completion, chamfer = chamfer_metrics(
#         np.asarray(est_aligned.points),
#         np.asarray(ref_pcd.points),
#         threshold=0.5
#     )
#
#     print(f"\n=== Evaluation Results ===")
#     print(f"Accuracy (m):        {accuracy:.4f}")
#     print(f"Completion (m):      {completion:.4f}")
#     print(f"Chamfer Distance (m): {chamfer:.4f}")
#
