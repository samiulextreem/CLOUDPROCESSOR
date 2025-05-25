import open3d as o3d
import numpy as np

# Load the point cloud with RGB
pcd = o3d.io.read_point_cloud("astra_with_red_cube.ply")

# Extract points and colors
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# Compute distances from the origin
distances = np.linalg.norm(points, axis=1)

# Filter: only keep points within 500 mm (50 cm)
mask = distances <= 500
filtered_points = points[mask]
filtered_colors = colors[mask]

# Create new point cloud with RGB
filtered_pcd = o3d.geometry.PointCloud()
filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

# Save as ASCII .ply (human-readable)
o3d.io.write_point_cloud("astra_filtered.ply", filtered_pcd, write_ascii=True)

