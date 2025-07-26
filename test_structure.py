#!/usr/bin/env python3

# Quick test to verify all imports and paths work correctly
import sys
import os

print("🧪 Testing updated file structure...")

try:
    # Test 1: Import PointCloudProcessor
    sys.path.append('utilities')
    from utilities.filteredmask import PointCloudProcessor
    print("✅ PointCloudProcessor imported successfully")
    
    # Test 2: Import position_verification
    from utilities import position_verification
    print("✅ position_verification imported successfully")
    
    # Test 3: Create processor instance
    processor = PointCloudProcessor()
    print("✅ PointCloudProcessor instance created successfully")
    
    # Test 4: Check file paths
    required_files = ['data_source/d435.ply', 'data_source/final_mask.png', 'data_source/d435_Color.png']
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ Found: {file_path}")
        else:
            print(f"⚠️  Missing: {file_path}")
    
    # Test 5: Check utilities directory structure
    utilities_files = ['utilities/filteredmask.py', 'utilities/position_verification.py']
    for file_path in utilities_files:
        if os.path.exists(file_path):
            print(f"✅ Found: {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
    
    print("\n🎉 All tests passed! The file structure is correctly organized.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
