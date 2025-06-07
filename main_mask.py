import cv2
import numpy as np

# Load the image
image = cv2.imread("d435_Color.png")  # Replace with your image path
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert to HSV

# Define green color range in HSV (adjust these values!)
lower_green = np.array([35, 50, 50])    # Lower bound for green (Hue=35° to 85°)
upper_green = np.array([85, 255, 255])  # Upper bound for green

# Create a binary mask (white=green, black=non-green)
mask = cv2.inRange(image_hsv, lower_green, upper_green)

# Optional: Clean up the mask (remove noise)
mask = cv2.erode(mask, None, iterations=1)  # Remove small noise
mask = cv2.dilate(mask, None, iterations=2)  # Fill gaps

# Save the mask
cv2.imwrite("d435mask.png", mask)