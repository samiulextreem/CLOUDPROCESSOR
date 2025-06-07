import open3d as o3d
import numpy as np
import cv2
import pandas as pd  # For CSV export

# Load point cloud
pcd = o3d.io.read_point_cloud("d435.ply")
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

if colors.shape[0] == 0:
    raise ValueError("Point cloud has no colors!")

# Camera parameters
img_width, img_height = 1280, 720
focal_length = 500  # Adjust based on FOV
cx, cy = img_width // 2, img_height // 2  # Image center

# Perspective projection
x, y, z = points[:, 0], points[:, 1], points[:, 2]

# Filter points in front of the camera (z > 0)
valid = z < 0
x, y, z = x[valid], y[valid], z[valid]
colors = colors[valid]

# Project to 2D
u = (focal_length * x / z) + cx
v = (focal_length * y / z) + cy

# Convert to integer coordinates
u_img = np.clip(u, 0, img_width - 1).astype(int)
v_img = np.clip(v, 0, img_height - 1).astype(int)

# Initialize image and track visible points
image = np.zeros((img_height, img_width, 3), dtype=np.uint8)
depth_buffer = np.full((img_height, img_width), np.inf)

# Store visible points and their pixel positions
visible_points = []

for i in range(len(u_img)):
    ui, vi = u_img[i], v_img[i]
    zi = z[i]
    if zi < depth_buffer[vi, ui]:
        depth_buffer[vi, ui] = zi
        image[vi, ui] = (colors[i] * 255).astype(np.uint8)
        visible_points.append([ui, vi, x[i], y[i], z[i], *colors[i]])

# Convert to a structured NumPy array or DataFrame
columns = ["u", "v", "x", "y", "z", "r", "g", "b"]
visible_points_df = pd.DataFrame(visible_points, columns=columns)

# Save to CSV
visible_points_df.to_csv("visible_points.csv", index=False)
print(f"Saved {len(visible_points)} visible points to 'visible_points.csv'")

# Display the image
cv2.imwrite("projected_image.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
cv2.imshow("Camera View (0,0,0)", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()