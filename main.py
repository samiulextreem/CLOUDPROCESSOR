#!/usr/bin/env python3
"""
Point Cloud Processing Pipeline - Main Entry Point
Simple and clean interface for running the complete pipeline.
"""

import sys
import os
sys.path.append('utilities')

def main():
    """Run the complete point cloud processing pipeline"""
    print("🚀 Point Cloud Processing Pipeline")
    print("=" * 40)
    
    # Import and setup
    try:
        from utilities.filteredmask import PointCloudProcessor
        processor = PointCloudProcessor()
        print("✓ Pipeline initialized")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Check required files
    required_files = [
        'data_source/d435.ply',
        'data_source/final_mask.png', 
        'data_source/d435_Color.png'
    ]
    
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print(f"❌ Missing files: {missing}")
        return False
    
    print("✓ All input files found")
    
    # Run pipeline
    try:
        print("\n🔄 Processing...")
        results = processor.process_complete_pipeline(
            show_visualizations=False,
            run_verification=True
        )
        
        # Show results
        print(f"\n✅ Processing Complete!")
        print(f"   • Visible points: {len(results['visible_points']):,}")
        print(f"   • Masked points: {len(results['masked_points']):,}")
        print(f"   • Regions detected: {len(results['center_positions'])}")
        print(f"   • Verification: {'✓' if results.get('verification_completed') else '✗'}")
        
        # Show output files
        output_files = [
            'output/projected_image.png',
            'output/d435_with_red_dots.png', 
            'output/position_final.csv',
            'output/points.ply',
            'output/verification.ply'
        ]
        
        print(f"\n📁 Output files:")
        for file in output_files:
            if os.path.exists(file):
                size = os.path.getsize(file) / 1024
                print(f"   ✓ {file} ({size:.1f}KB)")
        
        # Show detected positions
        if results['center_positions']:
            print(f"\n🎯 Detected positions:")
            for pos in results['center_positions']:
                print(f"   • Region {pos['Region']}: ({pos['X']:.3f}, {pos['Y']:.3f}, {pos['Z']:.3f})")
        
        print(f"\n🎉 Pipeline completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Processing error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
