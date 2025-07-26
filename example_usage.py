"""
Example demonstrating how to use the PointCloudProcessor class
"""
import sys
import os

# Add utilities to path
sys.path.append('utilities')

from utilities.filteredmask import PointCloudProcessor

def main():
    """Example of using the PointCloudProcessor class"""
    
    print("🚀 Point Cloud Processing Example")
    
    # Create an instance of the processor
    processor = PointCloudProcessor()
    
    # Option 1: Run the complete pipeline with default settings (including verification)
    print("\n=== Option 1: Complete Pipeline with Verification ===")
    try:
        results = processor.process_complete_pipeline(
            ply_filename="d435.ply",
            mask_filename="final_mask.png", 
            color_image_filename="d435_Color.png",
            show_visualizations=False,  # Set to True to see 2D/3D visualizations
            run_verification=True  # Automatically runs position verification
        )
        print("✅ Complete pipeline executed successfully!")
        print(f"   - Found {len(results['center_positions'])} center positions")
        print(f"   - Processed {len(results['visible_points'])} visible points")
        print(f"   - Masked {len(results['masked_points'])} points")
        print(f"   - Verification completed: {results['verification_completed']}")
    except Exception as e:
        print(f"❌ Error in complete pipeline: {e}")
    
    # Option 2: Run pipeline without verification, then run verification separately
    print("\n=== Option 2: Pipeline + Separate Verification ===")
    try:
        # Create a new processor instance
        processor2 = PointCloudProcessor()
        
        # Run pipeline without verification
        results = processor2.process_complete_pipeline(
            ply_filename="d435.ply",
            mask_filename="final_mask.png", 
            color_image_filename="d435_Color.png",
            show_visualizations=False,
            run_verification=False  # Skip verification in main pipeline
        )
        
        # Run verification separately
        verification_success = processor2.run_position_verification()
        print(f"✅ Separate verification completed: {verification_success}")
        
    except Exception as e:
        print(f"❌ Error in separate processing: {e}")
    
    # Option 2: Step-by-step processing
    print("\n=== Option 2: Step-by-Step Processing ===")
    try:
        # Step 1: Load point cloud
        processor.load_point_cloud("d435.ply")
        
        # Step 2: Project to 2D
        projected_image = processor.project_to_2d()
        print(f"✅ Created 2D projection: {projected_image.shape}")
        
        # Step 3: Apply mask and highlight
        highlighted_image, d435_with_red_dots, binary_mask = processor.apply_mask_and_highlight(
            "final_mask.png", "d435_Color.png"
        )
        print(f"✅ Applied mask and created highlights")
        
        # Step 4: Detect positions and create point cloud
        filtered_pcd = processor.detect_positions_and_create_point_cloud(binary_mask)
        print(f"✅ Created filtered point cloud with {len(filtered_pcd.points)} points")
        
        # Access the results
        print(f"   - Center positions: {len(processor.center_positions)}")
        print(f"   - Visible points: {len(processor.visible_points_df)}")
        print(f"   - Masked points: {len(processor.masked_points_df)}")
        
        # Optional: Show visualizations
        # processor.visualize_2d(projected_image, binary_mask, highlighted_image)
        # processor.visualize_3d(filtered_pcd)
        
    except Exception as e:
        print(f"❌ Error in step-by-step processing: {e}")
    
    print("\n🎉 Example completed!")

if __name__ == "__main__":
    main()
