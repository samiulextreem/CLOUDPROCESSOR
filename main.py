import sys
import os

# Add utilities folder to path if it exists
if os.path.exists('utilities'):
    sys.path.append('utilities')

# Import the filteredmask functionality
try:
    import filteredmask
    print("✓ Successfully imported filteredmask module")
except ImportError:
    print("⚠ Could not import filteredmask module - running as standalone script")
    
# Import position verification functionality
try:
    import position_verification
    print("✓ Successfully imported position_verification module")
except ImportError:
    print("⚠ Could not import position_verification module")

def run_point_cloud_filtering():
    """Run the point cloud filtering pipeline"""
    print("\n" + "="*50)
    print("POINT CLOUD FILTERING PIPELINE")
    print("="*50)
    
    try:
        # Check if required files exist
        required_files = ['data_source/d435.ply', 'data_source/final_mask.png', 'data_source/d435_Color.png']
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            print(f"❌ Missing required files: {missing_files}")
            return False
            
        print("✓ All required files found")
        
        # Run the filtering process
        print("🔄 Starting point cloud filtering...")
        
        # Execute the filteredmask script from utilities folder
        filteredmask_path = os.path.join('utilities', 'filteredmask.py')
        exec(open(filteredmask_path).read())
        
        print("✅ Point cloud filtering completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during filtering: {e}")
        return False

def run_position_verification():
    """Run the position verification with yellow cubes"""
    print("\n" + "="*50)
    print("POSITION VERIFICATION")
    print("="*50)
    
    try:
        # Check if position_final.csv exists
        if not os.path.exists('output/position_final.csv'):
            print("❌ output/position_final.csv not found. Run filtering first.")
            return False
            
        print("✓ Position file found")
        print("🔄 Creating verification visualization...")
        
        # Execute the position verification script
        exec(open('position_verification.py').read())
        
        print("✅ Position verification completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        return False

def display_results():
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
    except Exception as e:
        print(f"⚠ Could not display detection results: {e}")

def main():
    """Main function orchestrating the entire pipeline"""
    print("🚀 Starting Point Cloud Processing Pipeline")
    
    # Step 1: Run point cloud filtering
    if not run_point_cloud_filtering():
        print("❌ Pipeline failed at filtering stage")
        return False
    
    # Step 2: Run position verification
    if not run_position_verification():
        print("⚠ Verification failed, but filtering completed")
    
    # Step 3: Display results
    display_results()
    
    print("\n🎉 Pipeline completed successfully!")
    print("Ready for analysis and visualization!")
    return True

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