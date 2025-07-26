import open3d as o3d
import numpy as np
import cv2
import pandas as pd
import os

class PointCloudProcessor:
    """
    A comprehensive point cloud processing pipeline for RealSense D435 data.
    Handles point cloud filtering, mask application, position detection, and visualization.
    """
    
    def __init__(self, data_source_dir=None, output_dir=None):
        """
        Initialize the PointCloudProcessor
        
        Args:
            data_source_dir (str): Directory containing input files (ply, mask, color image)
            output_dir (str): Directory for output files
        """
        # Get the parent directory path (CLOUDPROCESSOR folder)
        self.parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Set directories
        self.data_source_dir = data_source_dir or os.path.join(self.parent_dir, "data_source")
        self.output_dir = output_dir or os.path.join(self.parent_dir, "output")
        
        # Camera parameters for RealSense D435
        self.img_width, self.img_height = 1280, 720  # Original camera dimensions (16:9 aspect ratio)
        self.fx, self.fy = 1125, 900  # Optimized focal lengths from calibration
        self.cx, self.cy = self.img_width // 2, self.img_height // 2  # Principal point (center of image)
        
        # Initialize data containers
        self.pcd = None
        self.points = None
        self.colors = None
        self.visible_points_df = None
        self.masked_points_df = None
        self.center_positions = []
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_point_cloud(self, ply_filename="d435.ply"):
        """
        Load and prepare point cloud data
        
        Args:
            ply_filename (str): Name of the PLY file in data_source directory
        """
        ply_path = os.path.join(self.data_source_dir, ply_filename)
        self.pcd = o3d.io.read_point_cloud(ply_path)
        self.points = np.asarray(self.pcd.points)
        self.colors = np.asarray(self.pcd.colors)

        if self.colors.shape[0] == 0:
            raise ValueError("Point cloud has no colors!")
        
        print(f"✓ Loaded point cloud with {len(self.points)} points")
    
    def project_to_2d(self):
        """
        Project 3D points to 2D image coordinates using camera intrinsics
        
        Returns:
            tuple: (projected_image, visible_points_dataframe)
        """
        # Filter points in front of camera
        valid = self.points[:, 2] < 0  # RealSense uses negative Z for points in front of camera
        x, y, z = self.points[valid, 0], self.points[valid, 1], self.points[valid, 2]
        colors = self.colors[valid]

        # Perspective projection using proper camera intrinsics
        u = self.img_width - ((self.fx * x / z) + self.cx)  # Flip X-axis to correct horizontal flipping
        v = (self.fy * y / z) + self.cy  # This maintains the correct projection direction

        # Clip to image bounds
        u_img = np.clip(u, 0, self.img_width - 1).astype(int)
        v_img = np.clip(v, 0, self.img_height - 1).astype(int)

        # Create projected image
        image = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        # Initialize with negative infinity since z is negative and we want to keep largest (least negative) z
        depth_buffer = np.full((self.img_height, self.img_width), -np.inf)
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
        self.visible_points_df = pd.DataFrame(visible_points, columns=columns)

        # Save projected image
        cv2.imwrite(os.path.join(self.output_dir, "projected_image.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"✓ Created 2D projection with {len(visible_points)} visible points")
        
        return image
    
    def apply_mask_and_highlight(self, mask_filename="final_mask.png", color_image_filename="d435_Color.png"):
        """
        Apply binary mask and create highlighted visualizations
        
        Args:
            mask_filename (str): Name of the mask file in data_source directory
            color_image_filename (str): Name of the color image file in data_source directory
            
        Returns:
            tuple: (highlighted_image, d435_with_red_dots)
        """
        # Load mask
        mask_path = os.path.join(self.data_source_dir, mask_filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"{mask_filename} not found at {mask_path}")

        # Load the original D435 color image
        d435_image_path = os.path.join(self.data_source_dir, color_image_filename)
        d435_image = cv2.imread(d435_image_path)
        if d435_image is None:
            raise FileNotFoundError(f"{color_image_filename} not found at {d435_image_path}")

        # Convert BGR to RGB for consistency
        d435_image_rgb = cv2.cvtColor(d435_image, cv2.COLOR_BGR2RGB)

        # Resize mask to match the d435 image dimensions if needed
        d435_height, d435_width = d435_image.shape[:2]
        mask_for_d435 = cv2.resize(mask, (d435_width, d435_height))

        # Create binary mask
        _, binary_mask_d435 = cv2.threshold(mask_for_d435, 127, 255, cv2.THRESH_BINARY)

        # Create image with red dots
        d435_with_red_dots = d435_image_rgb.copy()
        d435_with_red_dots[binary_mask_d435 == 255] = [255, 0, 0]  # Red color in RGB

        # Save the image with red dots
        cv2.imwrite(os.path.join(self.output_dir, "d435_with_red_dots.png"), cv2.cvtColor(d435_with_red_dots, cv2.COLOR_RGB2BGR))
        print(f"✓ Saved d435_Color.png with red dots overlay")

        # Print mask dimensions for debugging
        print(f"✓ Mask dimensions: {mask.shape}")

        # Ensure binary mask for projection
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Resize mask if needed
        if binary_mask.shape != (self.img_height, self.img_width):
            print(f"Resizing mask from {binary_mask.shape} to {self.img_width}x{self.img_height}")
            binary_mask = cv2.resize(binary_mask, (self.img_width, self.img_height))

        # Filter points using mask
        try:
            mask_values = binary_mask[self.visible_points_df["v"], self.visible_points_df["u"]]
            self.masked_points_df = self.visible_points_df[mask_values == 255]
            
            if len(self.masked_points_df) == 0:
                print("Warning: No points were selected by the mask!")
                print(f"Mask has {np.sum(binary_mask == 255)} white pixels")
                print(f"Visible points count: {len(self.visible_points_df)}")
            else:
                self.masked_points_df.to_csv(os.path.join(self.output_dir, "masked_3d_points.csv"), index=False)
                print(f"✓ Saved {len(self.masked_points_df)} masked points")
        except IndexError as e:
            print(f"Error applying mask: {e}")
            print("Check if the mask dimensions and visible points coordinates are compatible")
            # Continue with an empty DataFrame to avoid breaking the pipeline
            columns = ["u", "v", "x", "y", "z", "r", "g", "b"]
            self.masked_points_df = pd.DataFrame(columns=columns)

        # Get projected image from previous step (we need to call project_to_2d first)
        image = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        if self.visible_points_df is not None:
            for _, row in self.visible_points_df.iterrows():
                u, v = int(row['u']), int(row['v'])
                image[v, u] = [int(row['r']*255), int(row['g']*255), int(row['b']*255)]

        # Create highlighted image with red pixels for masked areas
        highlighted_image = image.copy()
        # Make masked pixels red
        highlighted_image[binary_mask == 255] = [255, 0, 0]  # Red color in RGB
        cv2.imwrite(os.path.join(self.output_dir, "masked_projection.png"), cv2.cvtColor(highlighted_image, cv2.COLOR_RGB2BGR))
        print(f"✓ Created highlighted projection image")
        
        return highlighted_image, d435_with_red_dots, binary_mask
    
    def detect_positions_and_create_point_cloud(self, binary_mask):
        """
        Detect center positions of masked regions and create point cloud with red highlighting
        
        Args:
            binary_mask (np.ndarray): Binary mask for highlighting
            
        Returns:
            tuple: (filtered_point_cloud, center_positions_list)
        """
        # Start with all visible points
        all_points = self.visible_points_df[["x", "y", "z"]].values
        all_colors = self.visible_points_df[["r", "g", "b"]].values.copy()

        # Highlight masked points in red
        if len(self.masked_points_df) > 0:
            try:
                mask_values = binary_mask[self.visible_points_df["v"], self.visible_points_df["u"]]
                mask_indices = mask_values == 255
                all_colors[mask_indices] = [1.0, 0.0, 0.0]  # Set to red (RGB values in range 0-1)
                print(f"✓ Highlighted {np.sum(mask_indices)} points in red")
                
                # Print center positions of masked (red) points grouped by connected components
                print("\n=== Center Positions of Masked Regions ===")
                
                # Find connected components in the binary mask to identify individual circles/regions
                num_labels, labels = cv2.connectedComponents(binary_mask)
                
                # Get the masked points (red points)
                red_points = self.visible_points_df[mask_indices]
                
                if len(red_points) > 0:
                    # List to store center positions for CSV export
                    self.center_positions = []
                    
                    # Group points by connected components
                    for label in range(1, num_labels):  # Skip label 0 (background)
                        # Create mask for this specific component
                        component_mask = (labels == label).astype(np.uint8) * 255
                        
                        # Find points belonging to this component
                        component_points = []
                        for idx, row in red_points.iterrows():
                            u, v = int(row['u']), int(row['v'])
                            if component_mask[v, u] == 255:
                                component_points.append(row)
                        
                        if component_points:
                            # Calculate center position (average of all points in this region)
                            x_coords = [p['x'] for p in component_points]
                            y_coords = [p['y'] for p in component_points]
                            z_coords = [p['z'] for p in component_points]
                            
                            center_x = np.mean(x_coords)
                            center_y = np.mean(y_coords)
                            center_z = np.mean(z_coords)
                            
                            print(f"Region {label}: X={center_x:.6f}, Y={center_y:.6f}, Z={center_z:.6f}")
                            
                            # Add to list for CSV export
                            self.center_positions.append({
                                'Region': label,
                                'X': center_x,
                                'Y': center_y,
                                'Z': center_z
                            })
                    
                    # Export center positions to CSV
                    if self.center_positions:
                        positions_df = pd.DataFrame(self.center_positions)
                        positions_df.to_csv(os.path.join(self.output_dir, "position_final.csv"), index=False)
                        print(f"✓ Exported {len(self.center_positions)} center positions to CSV")
                    else:
                        print("No center positions to export")
                
            except IndexError as e:
                print(f"Error highlighting points: {e}")

        # Create new point cloud with all points (highlighted ones in red)
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(all_points)
        filtered_pcd.colors = o3d.utility.Vector3dVector(all_colors)

        # Save the point cloud with red highlighting
        o3d.io.write_point_cloud(os.path.join(self.output_dir, "points.ply"), filtered_pcd)
        print(f"✓ Saved point cloud with {len(all_points)} total points (masked points highlighted in red)")
        
        return filtered_pcd
    
    def visualize_2d(self, image, binary_mask, highlighted_image):
        """
        Display 2D visualizations
        
        Args:
            image (np.ndarray): Projected image
            binary_mask (np.ndarray): Binary mask
            highlighted_image (np.ndarray): Highlighted projection image
        """
        cv2.imshow("Projected Image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imshow("Binary Mask", binary_mask)
        cv2.imshow("Highlighted Projection", cv2.cvtColor(highlighted_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def visualize_3d(self, point_cloud):
        """
        Display 3D point cloud visualization
        
        Args:
            point_cloud (o3d.geometry.PointCloud): Point cloud to visualize
        """
        print("Visualizing point cloud with red highlighting for masked areas...")
        o3d.visualization.draw_geometries([point_cloud])
    
    def run_position_verification(self):
        """
        Run position verification by creating yellow cubes at detected positions
        
        Returns:
            bool: True if verification completed successfully, False otherwise
        """
        try:
            # Check if position_final.csv exists
            positions_csv_path = os.path.join(self.output_dir, "position_final.csv")
            if not os.path.exists(positions_csv_path):
                print("❌ position_final.csv not found. Run detection first.")
                return False
            
            print("✓ Position file found")
            print("🔄 Creating verification visualization...")
            
            # Import and execute position verification
            import subprocess
            import sys
            
            # Get the path to the position verification script
            verification_script = os.path.join(os.path.dirname(__file__), "position_verification.py")
            
            # Execute the position verification script
            exec(open(verification_script).read())
            
            print("✅ Position verification completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error during verification: {e}")
            return False
    
    def process_complete_pipeline(self, ply_filename="d435.ply", mask_filename="final_mask.png", 
                                color_image_filename="d435_Color.png", show_visualizations=False,
                                run_verification=True):
        """
        Run the complete point cloud processing pipeline
        
        Args:
            ply_filename (str): Name of the PLY file
            mask_filename (str): Name of the mask file
            color_image_filename (str): Name of the color image file
            show_visualizations (bool): Whether to show 2D and 3D visualizations
            run_verification (bool): Whether to run position verification with yellow cubes
            
        Returns:
            dict: Results containing all generated data
        """
        print("🚀 Starting Point Cloud Processing Pipeline")
        
        # Step 1: Load point cloud
        self.load_point_cloud(ply_filename)
        
        # Step 2: Project to 2D
        projected_image = self.project_to_2d()
        
        # Step 3: Apply mask and create highlights
        highlighted_image, d435_with_red_dots, binary_mask = self.apply_mask_and_highlight(
            mask_filename, color_image_filename)
        
        # Step 4: Detect positions and create point cloud
        filtered_pcd = self.detect_positions_and_create_point_cloud(binary_mask)
        
        # Step 5: Run position verification (optional)
        verification_success = False
        if run_verification:
            verification_success = self.run_position_verification()
        
        # Step 6: Visualizations (optional)
        if show_visualizations:
            self.visualize_2d(projected_image, binary_mask, highlighted_image)
            self.visualize_3d(filtered_pcd)
        
        print("✅ Pipeline completed successfully!")
        
        # Return results
        return {
            'projected_image': projected_image,
            'highlighted_image': highlighted_image,
            'd435_with_red_dots': d435_with_red_dots,
            'binary_mask': binary_mask,
            'point_cloud': filtered_pcd,
            'visible_points': self.visible_points_df,
            'masked_points': self.masked_points_df,
            'center_positions': self.center_positions,
            'verification_completed': verification_success
        }


# For backward compatibility - create instance and run pipeline when script is executed directly
if __name__ == "__main__":
    processor = PointCloudProcessor()
    results = processor.process_complete_pipeline(show_visualizations=True)
    print("Processing completed!")





