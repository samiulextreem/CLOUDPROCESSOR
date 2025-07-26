# CLOUDPROCESSOR 🌐

## Overview
CLOUDPROCESSOR is an advanced Python-based toolkit for processing, analyzing, and visualizing 3D point cloud data from Intel RealSense cameras. This project provides a comprehensive pipeline for point cloud filtering, object detection, position verification, and spatial analysis using modern computer vision techniques.

## 🚀 Quick Start
```bash
# Clone the repository
git clone https://github.com/samiulextreem/CLOUDPROCESSOR.git
cd CLOUDPROCESSOR

# Install dependencies
pip install -r requirements.txt

# Run the main pipeline
python main.py
```

## ✨ Features
- **🎯 Complete Processing Pipeline**: One-command execution for end-to-end processing
- **📷 RealSense Integration**: Optimized for Intel RealSense D435 camera data
- **🎭 Mask-based Filtering**: Advanced 2D mask application to 3D point clouds
- **🔍 Object Detection**: Automated detection and positioning of objects in 3D space
- **📊 Position Verification**: Built-in verification system for detected positions
- **🖼️ Visualization**: Comprehensive visualization tools with projection mapping
- **📈 Data Export**: Export results in CSV and PLY formats
- **🔧 Modular Design**: Clean, maintainable code structure

## 📋 Requirements
- Python 3.12+
- Open3D
- NumPy
- Pandas
- OpenCV (cv2)
- Matplotlib
- SciPy

## 📁 Project Structure
```
CLOUDPROCESSOR/
├── main.py                    # 🎯 Main pipeline entry point
├── requirements.txt           # 📦 Dependencies
├── data_source/              # 📂 Input data
│   ├── d435.ply             # Point cloud data
│   ├── d435_Color.png       # RGB image
│   └── final_mask.png       # Segmentation mask
├── utilities/               # 🛠️ Core processing modules
│   ├── filteredmask.py      # Main processor class
│   ├── position_verification.py  # Position verification
│   └── asparagus_detector.py     # Object detection
├── dataset/                 # 📊 Training datasets
│   └── DATASET_asparagas/   # Asparagus detection dataset
└── output/                  # 📤 Generated results (auto-created)
```

## 🔧 Usage

### Basic Usage
```bash
python main.py
```
This runs the complete pipeline with default settings and provides detailed progress output.

### Advanced Usage
For custom processing, you can import and use the processor directly:
```python
from utilities.filteredmask import PointCloudProcessor

processor = PointCloudProcessor()
results = processor.process_complete_pipeline(
    show_visualizations=True,
    run_verification=True
)
```

## 📤 Output Files
The pipeline generates several output files in the `output/` directory:
- `projected_image.png` - 2D projection of the point cloud
- `d435_with_red_dots.png` - Detected positions marked on original image
- `position_final.csv` - Detected object positions (X, Y, Z coordinates)
- `points.ply` - Processed point cloud data
- `verification.ply` - Verification visualization

## 🔬 Pipeline Workflow
1. **Load Data**: Point cloud (PLY) + RGB image + segmentation mask
2. **Apply Mask**: Filter 3D points using 2D mask
3. **Detect Objects**: Identify regions of interest
4. **Calculate Positions**: Determine 3D coordinates
5. **Verify Results**: Quality assurance and validation
6. **Export Data**: Save results in multiple formats

## ⚠️ Important Notes
- **Model Files**: Large model files (*.pth) are excluded from git tracking
- **Output Folder**: Generated files are not version controlled
- **Camera Calibration**: Optimized for RealSense D435 with specific focal lengths

## 🔧 Configuration
The system is pre-configured for RealSense D435 cameras with:
- Focal Length X: 850-1125 (tunable)
- Focal Length Y: 900 (calibrated)
- Resolution: 1280x720

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author
**Samiul Haque**
- GitHub: [@samiulextreem](https://github.com/samiulextreem)

---
*Last updated: July 2025*
