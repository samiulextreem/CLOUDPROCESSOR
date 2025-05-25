import open3d as o3d
import numpy as np

# Load the point cloud
pcd = o3d.io.read_point_cloud("astra_filtered.ply")

# Extract points and colors
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# Convert colors to 0–255 for thresholding
colors_255 = (colors * 255).astype(int)

# Define "greenish" condition
greenish_mask = (colors_255[:, 1] > colors_255[:, 0] + 3) & \
                (colors_255[:, 1] > colors_255[:, 2] + 3)



# Filter points and colors
green_points = points[greenish_mask]
green_colors = colors[greenish_mask]

# Create point cloud with greenish points
green_pcd = o3d.geometry.PointCloud()
green_pcd.points = o3d.utility.Vector3dVector(green_points)
green_pcd.colors = o3d.utility.Vector3dVector(green_colors)

# Save outputs
o3d.io.write_point_cloud("astra_green_filtered.ply", green_pcd, write_ascii=True)
xyzrgb = np.hstack((green_points, (green_colors * 255).astype(int)))
np.savetxt("astra_green_filtered.txt", xyzrgb, fmt="%.2f %.2f %.2f %d %d %d", header="X Y Z (mm) R G B")

# 💡 Visualize the greenish point cloud
o3d.visualization.draw_geometries([green_pcd], window_name="Greenish Points",
                                  width=800, height=600,
                                  point_show_normal=False)


