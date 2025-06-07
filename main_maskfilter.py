import open3d as o3d
import numpy as np
import cv2
import pandas as pd

# ----------------------------
# 1. Load and Prepare Point Cloud
# ----------------------------
pcd = o3d.io.read_point_cloud("d435.ply")
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

if colors.shape[0] == 0:
    raise ValueError("Point cloud has no colors!")

# ----------------------------
# 2. Camera Projection Setup
# ----------------------------
img_width, img_height = 1280, 720
focal_length = 500  # Adjust for field of view
cx, cy = img_width // 2, img_height // 2

# Filter points in front of camera
valid = points[:, 2] < 0
x, y, z = points[valid, 0], points[valid, 1], points[valid, 2]
colors = colors[valid]

# Perspective projection
u = (focal_length * x / z) + cx
v = (focal_length * y / z) + cy

# Clip to image bounds
u_img = np.clip(u, 0, img_width - 1).astype(int)
v_img = np.clip(v, 0, img_height - 1).astype(int)

# ----------------------------
# 3. Create Projected Image
# ----------------------------
image = np.zeros((img_height, img_width, 3), dtype=np.uint8)
depth_buffer = np.full((img_height, img_width), np.inf)
visible_points = []

for i in range(len(u_img)):
    ui, vi = u_img[i], v_img[i]
    zi = z[i]
    if zi < depth_buffer[vi, ui]:
        depth_buffer[vi, ui] = zi
        image[vi, ui] = (colors[i] * 255).astype(np.uint8)
        visible_points.append([ui, vi, x[i], y[i], z[i], *colors[i]])

# Save visible points
columns = ["u", "v", "x", "y", "z", "r", "g", "b"]
visible_points_df = pd.DataFrame(visible_points, columns=columns)
visible_points_df.to_csv("visible_points.csv", index=False)
print(f"Saved {len(visible_points)} visible points")

# Save projected image
cv2.imwrite("projected_image.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

# ----------------------------
# 4. Apply Binary Mask
# ----------------------------
mask = cv2.imread("d435mask.png", cv2.IMREAD_GRAYSCALE)
if mask is None:
    raise FileNotFoundError("mask.png not found")

# Ensure binary mask
_, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# Resize mask if needed
if binary_mask.shape != (img_height, img_width):
    binary_mask = cv2.resize(binary_mask, (img_width, img_height))

# Filter points using mask
mask_values = binary_mask[visible_points_df["v"], visible_points_df["u"]]
masked_points_df = visible_points_df[mask_values == 255]
masked_points_df.to_csv("masked_3d_points.csv", index=False)
print(f"Saved {len(masked_points_df)} masked points")

# Create masked image
masked_image = cv2.bitwise_and(image, image, mask=binary_mask)
cv2.imwrite("masked_projection.png", cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))

# ----------------------------
# 5. Create filtered point cloud with colors
# ----------------------------
# Extract 3D points and colors from masked points dataframe
filtered_points = masked_points_df[["x", "y", "z"]].values
filtered_colors = masked_points_df[["r", "g", "b"]].values

# Create new point cloud from filtered points
filtered_pcd = o3d.geometry.PointCloud()
filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)  # Add colors to the point cloud

# Save the filtered point cloud
o3d.io.write_point_cloud("filtered_masked_points.ply", filtered_pcd)
print(f"Saved filtered point cloud with {len(filtered_points)} points and colors")

# ----------------------------
# 6. Visualization
# ----------------------------
# 2D visualization
cv2.imshow("Projected Image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
cv2.imshow("Binary Mask", binary_mask)
cv2.imshow("Masked Projection", cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3D visualization of filtered point cloud
print("Visualizing filtered point cloud (with colors)...")
o3d.visualization.draw_geometries([filtered_pcd])