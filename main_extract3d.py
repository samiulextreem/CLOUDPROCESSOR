import open3d as o3d
import numpy as np
import pandas as pd

# 1. Load masked points from CSV
masked_points_df = pd.read_csv("masked_3d_points.csv")

# 2. Prepare data for Open3D
filtered_points = masked_points_df[["x", "y", "z"]].values
filtered_colors = masked_points_df[["r", "g", "b"]].values

# 3. Create Open3D point cloud
pcd_filtered = o3d.geometry.PointCloud()
pcd_filtered.points = o3d.utility.Vector3dVector(filtered_points)
pcd_filtered.colors = o3d.utility.Vector3dVector(filtered_colors)

# 4. Save as PLY file
output_ply_path = "filtered_points.ply"
o3d.io.write_point_cloud(output_ply_path, pcd_filtered)
print(f"Saved filtered point cloud to {output_ply_path}")

# 5. Visualize (optional)
o3d.visualization.draw_geometries([pcd_filtered])