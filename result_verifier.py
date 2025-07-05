#!/usr/bin/env python3
"""
Result Verifier: Places small red cubes at coordinates from CSV file in point cloud
This script reads coordinates from masked_3d_points.csv and places red cubes
at those locations in the d435.ply point cloud to verify the masking results.
"""

import open3d as o3d
import pandas as pd
import numpy as np

def create_small_cube(center, size=0.005, color=[1.0, 0.0, 0.0]):
    """
    Create a small cube mesh at the given center position
    
    Args:
        center: [x, y, z] coordinates for cube center
        size: Size of the cube (default 0.005 for small cubes)
        color: RGB color as [r, g, b] with values 0-1 (default red)
    
    Returns:
        Open3D TriangleMesh of the cube
    """
    cube = o3d.geometry.TriangleMesh.create_box(size, size, size)
    # Translate cube so its center is at the specified position
    cube.translate(np.array(center) - np.array([size/2, size/2, size/2]))
    cube.paint_uniform_color(color)
    return cube

def main():
    """Main function to create verification point cloud with red cubes"""
    
    # Load the original point cloud
    print("Loading original point cloud...")
    try:
        original_pcd = o3d.io.read_point_cloud("d435.ply")
        print(f"Loaded point cloud with {len(original_pcd.points)} points")
    except Exception as e:
        print(f"Error loading d435.ply: {e}")
        return
    
    # Load the CSV coordinates
    print("Loading CSV coordinates...")
    try:
        df = pd.read_csv("masked_3d_points.csv")
        print(f"Loaded {len(df)} coordinate points from CSV")
    except Exception as e:
        print(f"Error loading masked_3d_points.csv: {e}")
        return
    
    # Extract x, y, z coordinates
    coordinates = df[['x', 'y', 'z']].values
    print(f"Creating {len(coordinates)} red cubes...")
    
    # Create list to store all geometries
    geometries = [original_pcd]
    
    # Create red cubes at each coordinate
    cube_size = 0.005  # Small cube size
    for i, coord in enumerate(coordinates):
        cube = create_small_cube(coord, cube_size)
        geometries.append(cube)
    
    # Combine all geometries
    print("Combining geometries...")
    combined_mesh = o3d.geometry.TriangleMesh()
    
    # Add all cube meshes to the combined mesh
    for geom in geometries[1:]:  # Skip the point cloud, only combine cubes
        combined_mesh += geom
    
    # Create a new point cloud that includes both original points and cube vertices
    verification_pcd = o3d.geometry.PointCloud()
    
    # Add original point cloud points and colors
    original_points = np.asarray(original_pcd.points)
    original_colors = np.asarray(original_pcd.colors)
    
    # Get vertices from the combined cube mesh and set them as red
    cube_vertices = np.asarray(combined_mesh.vertices)
    cube_colors = np.tile([1.0, 0.0, 0.0], (len(cube_vertices), 1))  # Red color for all cube vertices
    
    # Combine points and colors
    all_points = np.vstack([original_points, cube_vertices])
    all_colors = np.vstack([original_colors, cube_colors])
    
    # Set points and colors to the verification point cloud
    verification_pcd.points = o3d.utility.Vector3dVector(all_points)
    verification_pcd.colors = o3d.utility.Vector3dVector(all_colors)
    
    # Save the verification point cloud
    output_file = "verification.ply"
    success = o3d.io.write_point_cloud(output_file, verification_pcd)
    
    if success:
        print(f"Successfully saved verification point cloud to {output_file}")
        print(f"Total points: {len(all_points)} (original: {len(original_points)}, cube vertices: {len(cube_vertices)})")
    else:
        print(f"Failed to save {output_file}")
        return
    
    # Optional: Visualize the result
    print("\nVisualization controls:")
    print("  Left click + drag: Rotate")
    print("  Right click + drag: Pan")
    print("  Mouse wheel: Zoom in/out")
    print("  Q: Exit the viewer")
    print("\nDisplaying verification point cloud with red cubes...")
    
    # Show both the original point cloud and the cube meshes for better visualization
    o3d.visualization.draw_geometries([original_pcd, combined_mesh], 
                                    window_name="Verification: Original Point Cloud + Red Cubes")

if __name__ == "__main__":
    main()
