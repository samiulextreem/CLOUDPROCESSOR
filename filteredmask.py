import open3d as o3d
import numpy as np
import cv2
import pandas as pd
import os

# Set environment variable to fix Qt plugin issue for 3D visualization
os.environ["QT_QPA_PLATFORM"] = "xcb"

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
img_width, img_height = 1280, 720  # Original camera dimensions (16:9 aspect ratio)
# RealSense D435 calibrated parameters
fx, fy = 1125, 900  # Optimized focal lengths from calibration
cx, cy = img_width // 2, img_height // 2  # Principal point (center of image)

# Filter points in front of camera
valid = points[:, 2] < 0  # RealSense uses negative Z for points in front of camera
x, y, z = points[valid, 0], points[valid, 1], points[valid, 2]
colors = colors[valid]

# Perspective projection using proper camera intrinsics
u = img_width - ((fx * x / z) + cx)  # Flip X-axis to correct horizontal flipping
v = (fy * y / z) + cy  # This maintains the correct projection direction

# Clip to image bounds
u_img = np.clip(u, 0, img_width - 1).astype(int)
v_img = np.clip(v, 0, img_height - 1).astype(int)

# ----------------------------
# 3. Create Projected Image
# ----------------------------
image = np.zeros((img_height, img_width, 3), dtype=np.uint8)
# Initialize with negative infinity since z is negative and we want to keep largest (least negative) z
depth_buffer = np.full((img_height, img_width), -np.inf)
visible_points = []

for i in range(len(u_img)):
    ui, vi = u_img[i], v_img[i]
    zi = z[i]
    # For RealSense with negative Z values, more negative Z values are farther away
    # We need to find the largest (least negative) Z value
    if zi > depth_buffer[vi, ui]:
        depth_buffer[vi, ui] = zi
        image[vi, ui] = (colors[i] * 255).astype(np.uint8)
        visible_points.append([ui, vi, x[i], y[i], z[i], *colors[i]])

# Save visible points
columns = ["u", "v", "x", "y", "z", "r", "g", "b"]
visible_points_df = pd.DataFrame(visible_points, columns=columns)


# Save projected image
cv2.imwrite("projected_image.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


# ----------------------------
# 4. Apply Binary Mask
# ----------------------------
# mask = cv2.imread("binary_mask.png", cv2.IMREAD_GRAYSCALE)
mask = cv2.imread("final_mask.png", cv2.IMREAD_GRAYSCALE)
if mask is None:
    raise FileNotFoundError("final_mask.png not found")
    
# Print mask dimensions for debugging
print(f"Mask dimensions: {mask.shape}")
print(f"Image dimensions: {image.shape[:2]}")

# Ensure binary mask
_, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# Resize mask if needed
if binary_mask.shape != (img_height, img_width):
    print(f"Resizing mask from {binary_mask.shape} to {img_width}x{img_height}")
    binary_mask = cv2.resize(binary_mask, (img_width, img_height))

# Filter points using mask
try:
    mask_values = binary_mask[visible_points_df["v"], visible_points_df["u"]]
    masked_points_df = visible_points_df[mask_values == 255]
    
    if len(masked_points_df) == 0:
        print("Warning: No points were selected by the mask!")
        print(f"Mask has {np.sum(binary_mask == 255)} white pixels")
        print(f"Visible points count: {len(visible_points_df)}")
    else:
        masked_points_df.to_csv("masked_3d_points.csv", index=False)
        print(f"Saved {len(masked_points_df)} masked points")
except IndexError as e:
    print(f"Error applying mask: {e}")
    print("Check if the mask dimensions and visible points coordinates are compatible")
    # Continue with an empty DataFrame to avoid breaking the pipeline
    masked_points_df = pd.DataFrame(columns=columns)

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
o3d.io.write_point_cloud("points.ply", filtered_pcd)
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