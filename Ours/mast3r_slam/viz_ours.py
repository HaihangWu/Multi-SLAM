import open3d as o3d
import numpy as np
from plyfile import PlyData


def load_ply(filename):
    """Load a .ply file into Open3D point cloud."""
    ply_data = PlyData.read(filename)

    # Access the 'vertex' structured array
    vertices = ply_data['vertex']

    # Extract XYZ coordinates and convert them into a NumPy array (Nx3)
    points = np.array([vertices['x'], vertices['y'], vertices['z']]).T  # Shape: (N, 3)

    # Extract RGB colors for each point (Normalize them to [0, 1])
    colors = np.array([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0  # Shape: (N, 3)

    # Check the shape of points and colors
    print(f"Points shape: {points.shape}")
    print(f"Colors shape: {colors.shape}")

    # Ensure that both arrays are valid and have matching dimensions
    if points.shape[0] != colors.shape[0]:
        raise ValueError(f"Mismatch between points and colors: {points.shape[0]} != {colors.shape[0]}")

    # Create Open3D PointCloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    return point_cloud


def merge_point_clouds(pcd_list):
    """Merge multiple point clouds into one."""
    merged_pcd = o3d.geometry.PointCloud()

    for pcd in pcd_list:
        # Merge points and colors
        merged_pcd.points.extend(pcd.points)
        merged_pcd.colors.extend(pcd.colors)

    return merged_pcd


def visualize_point_cloud(pcd, window_size=(512, 512)):
    """Visualize the merged point cloud with a specified window size."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Merged Point Cloud", width=window_size[0], height=window_size[1])

    # Add point cloud to the visualizer
    vis.add_geometry(pcd)

    # Capture and save the screenshot
    #vis.capture_screen_image("merged_point_cloud.png")
    #print("Screenshot saved as 'merged_point_cloud.png'")

    # Display the point cloud interactively
    vis.run()
    vis.destroy_window()


# File paths for the results.ply of all three agents
ply_files = [
    #r"C:\Users\hthh1\Downloads\old\room0_agent_0\results.ply",
    #r"C:\Users\hthh1\Downloads\old\room0_agent_1\results.ply",
    r"C:\Users\hthh1\Downloads\old\room0_agent_2\results.ply",
]

# Load the point clouds
pcd_list = [load_ply(file) for file in ply_files]

# Merge the point clouds
merged_pcd = merge_point_clouds(pcd_list)

# Visualize the merged point cloud with window size 512x512
visualize_point_cloud(merged_pcd, window_size=(512, 512))



# import open3d as o3d
# import numpy as np
# from plyfile import PlyData, PlyElement
#
#
# def load_ply(filename):
#     """Load a .ply file into Open3D point cloud."""
#     ply_data = PlyData.read(filename)
#
#     # Access the 'vertex' structured array
#     vertices = ply_data['vertex']
#
#     # Extract XYZ coordinates and convert them into a NumPy array (Nx3)
#     points = np.array([vertices['x'], vertices['y'], vertices['z']]).T  # Shape: (N, 3)
#
#     # Extract RGB colors for each point (Normalize them to [0, 1])
#     colors = np.array([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0  # Shape: (N, 3)
#
#     # Check the shape of points and colors
#     print(f"Points shape: {points.shape}")
#     print(f"Colors shape: {colors.shape}")
#
#     # Ensure that both arrays are valid and have matching dimensions
#     if points.shape[0] != colors.shape[0]:
#         raise ValueError(f"Mismatch between points and colors: {points.shape[0]} != {colors.shape[0]}")
#
#     # Create Open3D PointCloud object
#     point_cloud = o3d.geometry.PointCloud()
#     point_cloud.points = o3d.utility.Vector3dVector(points)
#     point_cloud.colors = o3d.utility.Vector3dVector(colors)
#
#     return point_cloud
#
#
# def visualize_and_save_point_cloud(pcd,
#                                    image_filename=r"C:\Users\hthh1\Downloads\old\room0_agent_2\point_cloud_image.png"):
#     """Visualize and save the point cloud as an image with a specific window size."""
#     vis = o3d.visualization.Visualizer()
#
#     # Create a window with a specific size (512x512)
#     vis.create_window(width=512, height=512)
#
#     # Add point cloud to the visualizer
#     vis.add_geometry(pcd)
#
#     # Capture and save the screenshot
#     vis.capture_screen_image(image_filename)
#     print(f"Screenshot saved as {image_filename}")
#
#     # Display the point cloud interactively
#     vis.run()
#     vis.destroy_window()
#
#
# # Load the point cloud and visualize it
# pcd = load_ply(r"C:\Users\hthh1\Downloads\old\room0_agent_2\results.ply")
# visualize_and_save_point_cloud(pcd)
