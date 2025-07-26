import sys
import os

# Add utilities folder to path
sys.path.append('utilities')

def main():
    """Main function orchestrating the entire pipeline using PointCloudProcessor class"""
    print("🚀 Starting Point Cloud Processing Pipeline")
    
    # Import the PointCloudProcessor class
    try:
        from utilities.filteredmask import PointCloudProcessor
        print("✓ Successfully imported PointCloudProcessor class")
    except ImportError as e:
        print(f"❌ Could not import PointCloudProcessor class: {e}")
        return False
    
    # Check if required files exist
    required_files = ['data_source/d435.ply', 'data_source/final_mask.png', 'data_source/d435_Color.png']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
        
    print("✓ All required files found")
    
    try:
        # Create PointCloudProcessor instance
        processor = PointCloudProcessor()
        print("✓ Created PointCloudProcessor instance")
        
        # Run the complete pipeline
        print("\n" + "="*50)
        print("POINT CLOUD PROCESSING PIPELINE")
        print("="*50)
        
        results = processor.process_complete_pipeline(
            ply_filename="d435.ply",
            mask_filename="final_mask.png", 
            color_image_filename="d435_Color.png",
            show_visualizations=False,  # Set to True if you want to see visualizations
            run_verification=True  # This will automatically run position verification
        )
        
        # The verification is now integrated into the pipeline
        if results.get('verification_completed', False):
            print("✅ Position verification completed successfully!")
        else:
            print("⚠ Position verification was not completed")
        
        # Step 3: Display results
        display_results(results)
        
        print("\n🎉 Pipeline completed successfully!")
        print("Ready for analysis and visualization!")
        return True
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False

def display_results(results=None):
    """Display the results summary"""
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    
    # Check and display generated files
    output_files = {
        'output/projected_image.png': '2D projection of point cloud',
        'output/d435_with_red_dots.png': 'D435 image with red mask overlay',
        'output/masked_projection.png': 'Projected image with red highlighting',
        'output/masked_3d_points.csv': 'Filtered 3D points data',
        'output/position_final.csv': 'Center positions of detected objects',
        'output/points.ply': 'Point cloud with red highlighting',
        'output/verification.ply': 'Point cloud with yellow cube markers'
    }
    
    print("📁 Generated Files:")
    for filename, description in output_files.items():
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            size_str = f"{size/1024:.1f}KB" if size < 1024*1024 else f"{size/(1024*1024):.1f}MB"
            print(f"   ✓ {filename} ({size_str}) - {description}")
        else:
            print(f"   ✗ {filename} - {description}")
    
    # Display detection results if available
    try:
        import pandas as pd
        if os.path.exists('output/position_final.csv'):
            df = pd.read_csv('output/position_final.csv')
            print(f"\n🎯 Detection Results:")
            print(f"   Objects detected: {len(df)}")
            for _, row in df.iterrows():
                print(f"   Region {row['Region']}: X={row['X']:.6f}, Y={row['Y']:.6f}, Z={row['Z']:.6f}")
        
        # If results object is provided, show additional info
        if results:
            print(f"\n� Processing Statistics:")
            print(f"   Total visible points: {len(results['visible_points'])}")
            print(f"   Masked points: {len(results['masked_points'])}")
            print(f"   Center positions found: {len(results['center_positions'])}")
            
    except Exception as e:
        print(f"⚠ Could not display detection results: {e}")

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)