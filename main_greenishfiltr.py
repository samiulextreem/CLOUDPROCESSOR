"""
Green Object Filter for 3D Point Clouds

This script processes a 3D point cloud to isolate green objects (like asparagus stalks).
It uses color-based filtering to extract only points with predominantly green colors,
which helps isolate plant material from the background and other objects.

Workflow:
1. Loads a pre-filtered point cloud (typically output from a masking operation)
2. Applies a color-based filter to identify "greenish" points
3. Creates a new point cloud containing only the green points
4. Saves the filtered point cloud and visualizes the result

Input:
- filtered_masked_points.ply: A point cloud file that has already been through initial filtering

Output:
- d435_filtered.ply: A point cloud file containing only the green points
- Visualization of the green-filtered point cloud

This filtering stage is typically part of a pipeline for plant analysis,
where isolating the green parts is crucial for measuring plant attributes.
"""

import open3d as o3d
import numpy as np

# Load the point cloud (output from previous mask-based filtering)
pcd = o3d.io.read_point_cloud("filtered_masked_points.ply")

# Extract points and colors from the point cloud
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# Convert colors from [0-1] range to [0-255] for easier thresholding
colors_255 = (colors * 255).astype(int)

# Define "greenish" condition:
# - Green channel (index 1) must be higher than red channel (index 0) by at least 3
# - Green channel must be higher than blue channel (index 2) by at least 3
# This creates a mask (boolean array) where True values represent green pixels
greenish_mask = (colors_255[:, 1] > colors_255[:, 0] + 3) & \
                (colors_255[:, 1] > colors_255[:, 2] + 3)



# Filter points and colors using the greenish_mask
# This selects only the points and their corresponding colors that passed the green filter
green_points = points[greenish_mask]
green_colors = colors[greenish_mask]

# Create a new point cloud with only the greenish points
green_pcd = o3d.geometry.PointCloud()
green_pcd.points = o3d.utility.Vector3dVector(green_points)  # Set the 3D coordinates
green_pcd.colors = o3d.utility.Vector3dVector(green_colors)  # Set the corresponding colors

# Save the filtered point cloud to a PLY file (in ASCII format for better compatibility)
o3d.io.write_point_cloud("d435_filtered.ply", green_pcd, write_ascii=True)

# Create a combined array with XYZ coordinates and RGB colors (useful for other processing tools)
# Format: [x, y, z, r, g, b] where r,g,b are in [0-255] range
xyzrgb = np.hstack((green_points, (green_colors * 255).astype(int)))

# Print statistics about the filtering
print(f"Original point cloud: {len(points)} points")
print(f"Green-filtered point cloud: {len(green_points)} points")
print(f"Percentage of green points: {len(green_points)/len(points)*100:.1f}%")

# Visualize the filtered greenish point cloud in a 3D viewer
print("Visualizing green-filtered point cloud...")
o3d.visualization.draw_geometries([green_pcd], window_name="Greenish Points",
                                  width=800, height=600,
                                  point_show_normal=False)


