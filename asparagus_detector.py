#!/usr/bin/env python3
"""
Asparagus Detection Module
-------------------------
A production-ready module for detecting asparagus in images using a pre-trained Mask R-CNN model.

This module provides a class-based and function-based interface for detecting asparagus in images.
It can be used as a standalone command-line tool or imported as a library in other Python scripts.

Example usage as a module:
    from asparagus_detector import AsparagusDetector, detect_asparagus
    
    # Class-based interface
    detector = AsparagusDetector()
    results = detector.detect('image.png', save_results=True)
    
    # Function-based interface (backward compatible)
    results = detect_asparagus('image.png')

Example usage as a command-line tool:
    python asparagus_detector.py --image image.png --output ./results --threshold 0.7
"""

import os
import sys
import cv2
import numpy as np
import torch
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add detectron2 to path if needed
DETECTRON2_PATH = os.path.abspath('./detectron2')
if os.path.exists(DETECTRON2_PATH) and DETECTRON2_PATH not in sys.path:
    sys.path.insert(0, DETECTRON2_PATH)

# Detectron2 imports
import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo

# Make sure CUDA is available or fallback to CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class AsparagusDetector:
    """
    Class to detect asparagus in images using Mask R-CNN.
    """
    
    def __init__(self, model_path=None, confidence_threshold=0.7, device=None, select_best_only=True):
        """
        Initialize the asparagus detector.
        
        Creates an instance of the AsparagusDetector class with the specified parameters.
        The detector is configured with a Mask R-CNN model specifically trained for
        asparagus detection.
        
        Args:
            model_path (str): Path to model weights file. If None, uses default path
                             in 'output/model_final.pth'.
            confidence_threshold (float): Detection confidence threshold (0-1).
                                         Higher values reduce false positives but may
                                         miss some objects. Default: 0.7
            device (str): Device to use for inference ('cuda' or 'cpu').
                         If None, uses CUDA if available, otherwise CPU.
            select_best_only (bool): If True, only returns the highest scoring detection.
                                   If False, returns all detections above threshold. Default: True
        """
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Set selection mode
        self.select_best_only = select_best_only
            
        # Set up model
        self.predictor = self._setup_model(model_path, confidence_threshold)
    
    def _setup_model(self, weights_path=None, confidence_threshold=0.7):
        """
        Set up the Mask R-CNN model for asparagus detection.
        
        This method configures a Detectron2 Mask R-CNN model with the following settings:
        - Base architecture: Mask R-CNN with ResNet-50-FPN backbone
        - Single class (asparagus) detection
        - Mask prediction enabled
        - Customizable confidence threshold
        - Customizable weights path
        
        Args:
            weights_path (str): Path to model weights file. If None, uses default path
                               in 'output/model_final.pth'.
            confidence_threshold (float): Detection confidence threshold (0-1).
                                         Higher values reduce false positives.
            
        Returns:
            Detectron2 DefaultPredictor object configured for asparagus detection
        """
        # Configure model
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.DEVICE = self.device
        
        # Configure for single class (asparagus)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.MASK_ON = True
        
        # Set weights path
        if weights_path is None:
            weights_path = os.path.join("output", "model_final.pth")
        cfg.MODEL.WEIGHTS = weights_path
        
        # Set detection threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        
        # Create output directory if needed
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        
        return DefaultPredictor(cfg)
    
    def detect(self, image_or_path, save_results=False, output_dir=None):
        """
        Detect asparagus in an image.
        
        Args:
            image_or_path: Input image (numpy array) or path to image file (str)
            save_results (bool): Whether to save visualization and mask files
            output_dir (str): Directory to save output files (if None, uses current directory)
            
        Returns:
            dict: Detection results containing:
                - num_detections: Number of asparagus instances detected
                - instances: Detectron2 instances object for further processing
                - image_shape: Shape of the input image (height, width, channels)
                - spears_positions: List of dictionaries with position data for each spear:
                  - top_pixel: (x, y) coordinates of the top point
                  - bottom_pixel: (x, y) coordinates of the bottom point
                  - ten_percent_up: (x, y) coordinates of a point 10% up from the bottom
                  - score: Detection confidence score
                - output_files: Dictionary of paths to output visualization files (if save_results=True)
                  - visualization: Path to detection visualization image
                  - mask: Path to binary mask image (if detections exist)
                  - masked_image: Path to masked original image (if detections exist)
        
        Raises:
            ValueError: If the image file could not be loaded
            RuntimeError: If detection fails
        """
        # Set output directory
        if output_dir is None:
            output_dir = os.getcwd()
        
        # Load image if path provided
        if isinstance(image_or_path, str):
            image = cv2.imread(image_or_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_or_path}")
        else:
            image = image_or_path
        
        # Run inference
        outputs = self.predictor(image)
        
        # Get instances
        instances = outputs["instances"].to("cpu")
        num_detections = len(instances)
        
        # Filter to keep only the highest scoring detection if requested
        if num_detections > 0 and instances.has("scores") and self.select_best_only:
            # Get the index of the highest scoring detection
            best_idx = instances.scores.argmax().item()
            
            # Keep only the best detection
            instances = instances[best_idx:best_idx+1]  # Keep as tensor with single element
            num_detections = 1
        
        # Calculate top and bottom pixel positions for each spear
        spears_positions = []
        if num_detections > 0 and instances.has("pred_masks"):
            for i in range(num_detections):
                # Get mask for this instance
                mask = instances.pred_masks[i].numpy()
                
                # Find non-zero (True) indices
                y_indices, x_indices = np.where(mask)
                
                if len(y_indices) > 0:
                    # Find top-most and bottom-most pixels
                    top_y = int(y_indices.min())
                    bottom_y = int(y_indices.max())
                    
                    # Find corresponding x positions (centerline)
                    # For top pixel, find the x that corresponds to the minimum y
                    top_y_indices = np.where(y_indices == top_y)[0]
                    top_x_values = x_indices[top_y_indices]
                    top_x = int(np.mean(top_x_values))
                    
                    # For bottom pixel, find the x that corresponds to the maximum y
                    bottom_y_indices = np.where(y_indices == bottom_y)[0]
                    bottom_x_values = x_indices[bottom_y_indices]
                    bottom_x = int(np.mean(bottom_x_values))
                    
                    # Calculate the point 10% up from the bottom
                    ten_percent_up_y = int(bottom_y - (bottom_y - top_y) * 0.1)
                    
                    # Find the x value at this height by interpolating between top and bottom points
                    # First check if we have indices at exactly this height
                    ten_percent_y_indices = np.where(y_indices == ten_percent_up_y)[0]
                    
                    if len(ten_percent_y_indices) > 0:
                        # We have actual points at this height, use their mean
                        ten_percent_x_values = x_indices[ten_percent_y_indices]
                        ten_percent_x = int(np.mean(ten_percent_x_values))
                    else:
                        # Interpolate between top and bottom
                        # If top_y == bottom_y (horizontal spear), use the bottom_x
                        if bottom_y == top_y:
                            ten_percent_x = bottom_x
                        else:
                            # Linear interpolation
                            percentage = (ten_percent_up_y - top_y) / (bottom_y - top_y)
                            ten_percent_x = int(top_x + percentage * (bottom_x - top_x))
                    
                    spears_positions.append({
                        "spear_id": i,
                        "top_pixel": (top_x, top_y),
                        "bottom_pixel": (bottom_x, bottom_y),
                        "ten_percent_up": (ten_percent_x, ten_percent_up_y),
                        "score": float(instances.scores[i]) if instances.has("scores") else None
                    })
        
        # Prepare results dictionary
        results = {
            "num_detections": num_detections,
            "instances": instances,
            "image_shape": image.shape,
            "spears_positions": spears_positions,
            "output_files": {}
        }
        
        # Save results if requested
        if save_results:
            self._save_visualizations(image, instances, output_dir, results)
        
        return results
    
    def _save_visualizations(self, image, instances, output_dir, results):
        """
        Create and save visualization images.
        
        This method generates and saves three types of visualization files:
        1. Detection visualization with bounding boxes and scores
        2. Binary mask of all detections
        3. Original image with background removed (only shows detected asparagus)
        
        Args:
            image: Input image (numpy array)
            instances: Detectron2 instances object containing predictions
            output_dir: Directory to save output files
            results: Results dictionary to update with file paths
        
        Returns:
            None, but updates the results dictionary with output file paths
        """
        # Create visualization
        v = Visualizer(image[:, :, ::-1], scale=1.0)
        result = v.draw_instance_predictions(instances)
        result_img = result.get_image()[:, :, ::-1]
        
        # Mark top and bottom positions on the image if available
        if results.get("spears_positions"):
            # Copy the visualization image for marking positions
            marked_img = result_img.copy()
            
            # Define colors and marker sizes
            top_color = (0, 255, 0)     # Green for top pixels
            bottom_color = (0, 0, 255)  # Red for bottom pixels
            ten_percent_color = (0, 0, 255)  # Red for 10% up point (changed to red)
            marker_size = 5
            cut_point_marker_size = 10  # Larger marker size for cut points (twice as big)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            
            # Draw markers and labels for each spear
            for spear in results["spears_positions"]:
                # Draw top pixel
                top_x, top_y = spear["top_pixel"]
                cv2.circle(marked_img, (top_x, top_y), marker_size, top_color, -1)
                cv2.putText(marked_img, f"T{spear['spear_id']}", (top_x + 5, top_y - 5), 
                           font, font_scale, top_color, font_thickness)
                
                # Draw bottom pixel
                bottom_x, bottom_y = spear["bottom_pixel"]
                cv2.circle(marked_img, (bottom_x, bottom_y), marker_size, bottom_color, -1)
                cv2.putText(marked_img, f"B{spear['spear_id']}", (bottom_x + 5, bottom_y + 15), 
                           font, font_scale, bottom_color, font_thickness)
                
                # Draw 10% up point - using larger marker size
                ten_percent_x, ten_percent_y = spear["ten_percent_up"]
                cv2.circle(marked_img, (ten_percent_x, ten_percent_y), cut_point_marker_size, ten_percent_color, -1)
                
                # Draw a line connecting the two points
                cv2.line(marked_img, (top_x, top_y), (bottom_x, bottom_y), (255, 255, 0), 1)
            
            # Save visualization with markers
            vis_path = os.path.join(output_dir, "detection_visualization.png")
            cv2.imwrite(vis_path, marked_img)
            results["output_files"]["visualization"] = vis_path
        else:
            # Save standard visualization
            vis_path = os.path.join(output_dir, "detection_visualization.png")
            cv2.imwrite(vis_path, result_img)
            results["output_files"]["visualization"] = vis_path
        
        # Process mask if detections exist
        num_detections = len(instances)
        if num_detections > 0 and instances.has("pred_masks"):
            # Generate object detection mask
            combined_mask = instances.pred_masks.any(dim=0).numpy()
            mask_img = (combined_mask * 255).astype(np.uint8)
            
            # Save the original binary mask (without any markings)
            obj_mask_path = os.path.join(output_dir, "binary_mask.png")
            cv2.imwrite(obj_mask_path, mask_img)
            results["output_files"]["mask"] = obj_mask_path
            
            # Create a separate binary mask for cut points
            if results.get("spears_positions"):
                # Create an empty mask for cut points with same dimensions as the original mask
                cut_points_mask = np.zeros_like(mask_img)
                
                # Draw the 10% up points (cut points) on this separate mask
                for spear in results["spears_positions"]:
                    # Get 10% up point coordinates
                    ten_percent_x, ten_percent_y = spear["ten_percent_up"]
                    # Draw a filled circle at the cut point position with the larger marker size
                    cv2.circle(cut_points_mask, (ten_percent_x, ten_percent_y), cut_point_marker_size, 255, -1)
                
                # Save the cut points binary mask
                cut_points_mask_path = os.path.join(output_dir, "cut_points_mask.png")
                cv2.imwrite(cut_points_mask_path, cut_points_mask)
                results["output_files"]["cut_points_mask"] = cut_points_mask_path
                
                # Also create a visualization with both masks for debugging
                # Convert object mask to color for better visualization
                mask_img_color = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
                
                # Mark the cut points on the colored mask with larger marker size
                for spear in results["spears_positions"]:
                    ten_percent_x, ten_percent_y = spear["ten_percent_up"]
                    cv2.circle(mask_img_color, (ten_percent_x, ten_percent_y), cut_point_marker_size, (0, 0, 255), -1)
                
                # Save the combined visualization
                combined_mask_path = os.path.join(output_dir, "combined_mask_visualization.png")
                cv2.imwrite(combined_mask_path, mask_img_color)
                results["output_files"]["combined_mask_visualization"] = combined_mask_path
    
    def get_model_info(self):
        """
        Get detailed information about the model.
        
        This method provides information about the model architecture, parameters,
        and the device it's running on. This is useful for debugging and understanding
        the computational resources required by the model.
        
        Returns:
            dict: Model information containing:
                - device: The device being used for inference ('cuda' or 'cpu')
                - total_parameters: Total number of parameters in the model
                - trainable_parameters: Number of trainable parameters
                - model: String representation of the model architecture
        """
        model = self.predictor.model
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            "device": self.device,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model": str(model),
        }


# Function-based interface for backward compatibility with original script
def detect_asparagus(image_path, output_dir=None, visualize=True, confidence_threshold=0.7, model_path=None, select_best_only=True):
    """
    Detect asparagus in an image (backward-compatible function interface).
    
    This function provides backward compatibility with the original asperagas_detection.py script.
    It creates a temporary AsparagusDetector instance and uses it to detect asparagus in the image.
    
    Args:
        image_path (str): Path to the input image
        output_dir (str): Directory to save output files (if None, uses current directory)
        visualize (bool): Whether to generate visualization images
        confidence_threshold (float): Detection confidence threshold (0-1)
        model_path (str): Path to model weights file (if None, uses default path)
        select_best_only (bool): If True, only returns the highest scoring detection.
                               If False, returns all detections above threshold. Default: True
        
    Returns:
        dict: Detection results and paths to output files, including:
            - num_detections: Number of asparagus instances detected
            - instances: Detectron2 instances object for further processing
            - image_shape: Shape of the input image (height, width, channels) 
            - spears_positions: List of dictionaries with position data for each spear:
              - top_pixel: (x, y) coordinates of the top point
              - bottom_pixel: (x, y) coordinates of the bottom point
              - ten_percent_up: (x, y) coordinates of a point 10% up from the bottom
              - score: Detection confidence score
            - output_files: Dictionary of paths to output files
    """
    # Set output directory
    if output_dir is None:
        output_dir = os.getcwd()
    
    # Create detector
    detector = AsparagusDetector(
        model_path=model_path, 
        confidence_threshold=confidence_threshold,
        select_best_only=select_best_only
    )
    
    # Run detection
    return detector.detect(image_path, save_results=visualize, output_dir=output_dir)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Asparagus Detection Tool")
    parser.add_argument("--image", "-i", required=True, help="Path to input image")
    parser.add_argument("--output", "-o", default=None, help="Path to output directory")
    parser.add_argument("--threshold", "-t", type=float, default=0.7, help="Detection confidence threshold (0-1)")
    parser.add_argument("--model", "-m", default=None, help="Path to model weights file")
    parser.add_argument("--info", action="store_true", help="Display model information")
    parser.add_argument("--no-visualization", dest="visualize", action="store_false", 
                      help="Disable visualization generation")
    parser.add_argument("--all-detections", action="store_true", 
                      help="Return all detections instead of just the highest scoring one")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if CUDA is available
    logging.info(f"Using device: {DEVICE}")
    
    # Validate image path
    image_path = args.image
    if not os.path.exists(image_path):
        logging.error(f"Image not found: {image_path}")
        sys.exit(1)
    
    try:
        # Create detector
        detector = AsparagusDetector(
            model_path=args.model,
            confidence_threshold=args.threshold,
            select_best_only=not args.all_detections  # Invert flag: default is to select best only
        )
        
        # Show model info if requested
        if args.info:
            model_info = detector.get_model_info()
            print("\nModel Information:")
            print(f"  Device: {model_info['device']}")
            print(f"  Total Parameters: {model_info['total_parameters']:,}")
            print(f"  Trainable Parameters: {model_info['trainable_parameters']:,}")
            print("\nModel Architecture:")
            print(model_info['model'])
        
        # Run detection
        logging.info(f"Detecting asparagus in {image_path}")
        results = detector.detect(
            image_path, 
            save_results=args.visualize, 
            output_dir=args.output
        )
        
        # Display results
        print(f"\nDetection Results:")
        if not args.all_detections:
            print(f"  Selected highest scoring asparagus (best of all detections)")
        print(f"  Detected {results['num_detections']} asparagus instances")
        print(f"  Image shape: {results['image_shape']}")
        
        # Display spear positions
        if results.get("spears_positions"):
            print("\nAsparagus Spear Positions (x, y):")
            print("  ID   Score      Top Pixel       Bottom Pixel     10% Up Point")
            print("  --   -----      ---------       ------------     -----------")
            for spear in results["spears_positions"]:
                score = spear.get("score", "N/A")
                score_str = f"{score:.3f}" if isinstance(score, float) else score
                print(f"  {spear['spear_id']:<4} {score_str:<10} {spear['top_pixel']} {spear['bottom_pixel']} {spear['ten_percent_up']}")
        
        if results.get("output_files"):
            print("\nOutput Files:")
            for file_type, file_path in results["output_files"].items():
                print(f"  - {file_type}: {file_path}")
                
    except Exception as e:
        logging.error(f"Error during detection: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
