# CLOUDPROCESSOR

## Overview
CLOUDPROCESSOR is a Python-based toolkit for processing, analyzing, and visualizing 3D point cloud data. This project provides a set of modular tools for point cloud filtering, segmentation, projection, and visualization using the Open3D library and other computer vision techniques.

## ⚠️ Model Files
Large model files (*.pth) are not included in this repository due to GitHub's file size limit (100MB). 
You can:
1. Train models locally using the provided scripts
2. Download pre-trained models from the release page (if available)
3. Request model files directly from the author

## Features
- **Point Cloud Loading & Visualization**: Load PLY files and visualize 3D point clouds
- **Mask-based Filtering**: Apply 2D masks to 3D point clouds for object segmentation
- **Color-based Filtering**: Filter points based on color properties (green, red detection)
- **Distance-based Filtering**: Remove points beyond specified distances
- **3D to 2D Projection**: Project 3D point clouds to 2D images
- **Position Detection**: Locate objects in 3D space
- **Segmentation**: Isolate specific objects within point clouds

## Requirements
- Python 3.12+
- Open3D
- NumPy
- Pandas
- OpenCV (cv2)
- Matplotlib
- SciPy

## Files Description

### Data Files
- `astra.ply`, `astra_filtered.ply`, etc. - Point cloud data files
- `Record_20250523094916.bag` - RealSense camera recording
- `image.png`, `mask.png` - Image data and mask for filtering

### Core Scripts
- `main_extract3d.py` - Extracts 3D points from masked data
- `main_mask.py` - Creates masks for green object detection
- `main_maskfilter.py` - Applies masks to filter point clouds
- `main_projection.py` - Projects 3D points to 2D space
- `main_distancefiltr.py` - Filters points by distance
- `main_greenishfiltr.py` - Filters points with green color
- `main_position.py` - Calculates object positions
- `main_redorigin.py` - Detects red objects
- `main_segmentor.py` - Segments objects in point clouds
- `HW10.py` - Signal processing demonstrations

## Usage
1. Ensure all dependencies are installed
2. Run the desired script with Python:
   ```
   python main_extract3d.py
   ```
3. Output files will be generated in the project directory

## Project Structure
The project follows a modular approach where each script handles a specific processing task. The typical workflow involves:
1. Loading point cloud data
2. Filtering or processing the data
3. Visualizing or saving the results

## License
[Specify license information here]

## Contributors
[List contributors here]
