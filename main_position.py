import open3d as o3d
import numpy as np

# Load the PLY file
pcd = o3d.io.read_point_cloud("astra_green_filtered.ply")  # change filename if needed

# Convert to NumPy array
points = np.asarray(pcd.points)

# Filter out (0, 0, 0) points
non_zero_points = points[~np.all(points == 0, axis=1)]

# Print the point positions
print("X Y Z (mm):")
for point in non_zero_points:
    print(f"{point[0]:.2f} {point[1]:.2f} {point[2]:.2f}")



print("+---------------------------------+")


# Load the PLY file
pcd = o3d.io.read_point_cloud("filtered_points.ply")  # change filename if needed

# Convert to NumPy array
points = np.asarray(pcd.points)

# Filter out (0, 0, 0) points
non_zero_points = points[~np.all(points == 0, axis=1)]

# Print the point positions
print("X Y Z (mm):")
for point in non_zero_points:
    print(f"{point[0]:.2f} {point[1]:.2f} {point[2]:.2f}")
