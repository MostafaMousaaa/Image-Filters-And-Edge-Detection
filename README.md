# Computer Vision Image Processing Application

## Overview

This application is a comprehensive tool for image processing and computer vision operations. It provides an intuitive graphical user interface for applying various image processing algorithms, visualizing results, and comparing different techniques.

## Features

### 1. Image Reading and Display
- Support for RGB and grayscale images
- Zoomable image display
- Side-by-side image comparison

### 2. Noise Generation
- Gaussian noise with adjustable mean and sigma
- Salt & Pepper noise with customizable probabilities
- Uniform noise with adjustable low and high values

### 3. Image Filtering
- Average filter with variable kernel size
- Gaussian filter with adjustable sigma
- Median filter with variable kernel size

### 4. Edge Detection
- Sobel edge detector with X and Y direction visualization
- Roberts edge detector with X and Y direction visualization
- Prewitt edge detector with X and Y direction visualization
- Canny edge detector with adjustable thresholds

### 5. Histogram Analysis
- Grayscale histogram visualization
- RGB channel histograms (separate for R, G, B)
- Cumulative distribution function display for each channel

### 6. Image Enhancement
- Histogram equalization
- Image normalization

### 7. Thresholding
- Global thresholding with adjustable threshold value
- Local (adaptive) thresholding with variable block size and constant

### 8. Color Operations
- Transformation from color to grayscale
- Channel-wise histogram visualization and equalization

### 9. Frequency Domain Processing
- Low pass filtering (Ideal and Butterworth)
- High pass filtering (Ideal and Butterworth)
- Adjustable cutoff frequency

### 10. Hybrid Images
- Combining two images with adjustable alpha blending
- Preview of both source images
- Control over the blending factor

## Implementation Details

### Edge Detection

All edge detection methods (except Canny) provide visualization in three modes:
- Magnitude: Combined strength of edges in all directions
- X Direction: Horizontal edge components
- Y Direction: Vertical edge components

### Histogram Analysis

The histogram widget provides three views:
- Grayscale: Shows intensity distribution for grayscale images or luminance for color images
- RGB: Shows separate histograms for Red, Green, and Blue channels
- Distribution: Shows cumulative distribution functions used for histogram equalization

### Frequency Domain Filters

Two filter types are available:
- Low Pass: Keeps the low-frequency components (smoothing)
- High Pass: Keeps the high-frequency components (edge enhancement)

Each filter can use either:
- Ideal: Sharp cutoff at the specified frequency
- Butterworth: Gradual transition for smoother results

## Usage Instructions

1. Load an image using the "Open Image" button or File menu
2. Apply operations using the tabs on the right sidebar
3. View results in the main display area
4. Check histograms in the bottom panel
5. Save results using the "Save Image" button or File menu

## Requirements

- Python 3.8+
- PyQt6
- OpenCV
- NumPy
- Matplotlib
