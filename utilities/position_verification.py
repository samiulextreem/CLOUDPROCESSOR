import open3d as o3d
import numpy as np
import pandas as pd
import os

def create_red_cube(center, size=0.01):
    """Create a yellow cube mesh at the given center position"""
    cube = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
    # Move cube so that center is at the specified position
    cube.translate(np.array(center) - np.array([size/2, size/2, size/2]))
    # Color the cube yellow
    cube.paint_uniform_color([1.0, 1.0, 0.0])  # Yellow color in RGB
    return cube

# Get the parent directory path (CLOUDPROCESSOR folder)
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Read positions from CSV file
positions_csv_path = os.path.join(parent_dir, "output", "position_final.csv")
positions_df = pd.read_csv(positions_csv_path)
print(f"Loaded {len(positions_df)} positions from {positions_csv_path}")

# Load the original point cloud
ply_path = os.path.join(parent_dir, "data_source", "d435.ply")
pcd = o3d.io.read_point_cloud(ply_path)
print(f"Loaded point cloud with {len(pcd.points)} points")

# Create a list to store all geometries (point cloud + cubes)
geometries = [pcd]

# Create red cubes at each position
for index, row in positions_df.iterrows():
    position = [row['X'], row['Y'], row['Z']]
    print(f"Creating yellow cube at position: X={position[0]:.6f}, Y={position[1]:.6f}, Z={position[2]:.6f}")
    
    # Create a yellow cube at this position (half the previous size)
    cube = create_red_cube(position, size=0.01)  # 1cm cube size (half of previous 2cm)
    geometries.append(cube)

# Combine all geometries into one point cloud
combined_pcd = o3d.geometry.PointCloud()
combined_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
combined_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors))

# Convert cube meshes to point clouds and add them
for i, geometry in enumerate(geometries[1:]):  # Skip the original point cloud
    # Sample points from the cube mesh
    cube_pcd = geometry.sample_points_uniformly(number_of_points=1000)
    
    # Convert to numpy arrays and combine
    original_points = np.asarray(combined_pcd.points)
    original_colors = np.asarray(combined_pcd.colors)
    cube_points = np.asarray(cube_pcd.points)
    cube_colors = np.asarray(cube_pcd.colors)
    
    # Combine arrays
    combined_points = np.vstack([original_points, cube_points])
    combined_colors = np.vstack([original_colors, cube_colors])
    
    # Update the combined point cloud
    combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
    combined_pcd.colors = o3d.utility.Vector3dVector(combined_colors)

# Save the combined point cloud
output_path = os.path.join(parent_dir, "output", "verification.ply")
o3d.io.write_point_cloud(output_path, combined_pcd)
print(f"Saved verification point cloud with {len(combined_pcd.points)} points to '{output_path}'")

# Display the positions that were marked
print("\n=== Positions Marked with Yellow Cubes ===")
for index, row in positions_df.iterrows():
    print(f"Region {row['Region']}: X={row['X']:.6f}, Y={row['Y']:.6f}, Z={row['Z']:.6f}")

# Visualize the result
print("\nVisualizing the point cloud with yellow cubes...")
o3d.visualization.draw_geometries(geometries, window_name="Point Cloud with Yellow Cubes at Detected Positions")
