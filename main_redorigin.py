import numpy as np
import open3d as o3d

# 1. Load your existing point cloud
pcd = o3d.io.read_point_cloud("d435.ply")

# 2. Create a RED cube at origin
cube_size = 0.02  # Size of the cube (in meters)
resolution = 20   # Points per edge (controls density)

# Generate cube points
x = np.linspace(-cube_size/2, cube_size/2, resolution)
y = np.linspace(-cube_size/2, cube_size/2, resolution)
z = np.linspace(-cube_size/2, cube_size/2, resolution)
xx, yy, zz = np.meshgrid(x, y, z)
cube_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

# Assign RED color to all cube points (RGB [1,0,0])
cube_colors = np.tile([1.0, 0.0, 0.0], (len(cube_points), 1))  # Pure red

# 3. Combine with original point cloud
combined_points = np.vstack((
    np.asarray(pcd.points),
    cube_points
))

combined_colors = np.vstack((
    np.asarray(pcd.colors) if pcd.has_colors() else np.zeros((len(pcd.points), 3)),
    cube_colors
))

# Create final point cloud
combined_pcd = o3d.geometry.PointCloud()
combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
combined_pcd.colors = o3d.utility.Vector3dVector(combined_colors)

# 4. Save and visualize
output_path = "d435_red_cube.ply"
o3d.io.write_point_cloud(output_path, combined_pcd)
print(f"Saved red cube at origin to: {output_path}")

# Visualize with coordinate axes
coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
o3d.visualization.draw_geometries([combined_pcd, coord_frame])