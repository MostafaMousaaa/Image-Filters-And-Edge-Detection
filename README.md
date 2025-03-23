# Image Filters and Edge Detection Application

A comprehensive computer vision application that provides a visual interface for image processing, filtering, edge detection, and active contour models. This project was developed as part of the Computer Vision course assignments.

## Features

### Assignment 1: Image Processing and Filtering

- **Image Loading and Visualization**: Load and display images with interactive controls
- **Noise Generators**: Add Gaussian, Salt & Pepper, and Uniform noise to images
- **Low-Pass Filters**:
  - Average (Mean) Filter
  - Gaussian Filter
  - Median Filter
- **Edge Detection**:
  - Sobel Operator (Magnitude, X and Y directions)
  - Roberts Operator
  - Prewitt Operator 
  - Canny Edge Detector
- **Thresholding**:
  - Global Thresholding
  - Local Adaptive Thresholding
- **Frequency Domain Processing**:
  - Gaussian Low-Pass Filter
  - Butterworth High-Pass Filter
- **Hybrid Image Generation**: Combine two images with controllable alpha blending
- **Histogram Analysis**: View and analyze image histograms for both grayscale and color images

### Assignment 2: Edge Detection and Active Contours

- **Advanced Edge Detection**:
  - Canny Edge Detection with parameter control
  - Hough Line Transform for line detection
  - Hough Circle Transform for circle detection
  - Ellipse detection
  
- **Active Contour Model (Snake)**:
  - Interactive contour initialization
  - Contour evolution with Greedy Snake algorithm
  - Adjustable parameters (alpha, beta, gamma)
  - Contour metrics calculation (perimeter, area)
  - Chain code representation

## Performance Optimizations

- **C++ Integration**: Critical algorithms implemented in C++ for performance
- **Optimized Data Structures**: Efficient memory usage for large images
- **Multi-threaded Processing**: Parallel processing for computationally intensive operations

## Installation

### Prerequisites
- Python 3.8+
- OpenCV
- PyQt6
- NumPy
- C++ compiler (for building the C++ modules)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Image-Filters-And-Edge-Detection.git
cd Image-Filters-And-Edge-Detection
```

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

3. Build C++ modules (optional for improved performance):
```bash
cd image-processing-app/src/processing
python setup.py build_ext --inplace
```

## Usage

Run the application:

```bash
python image-processing-app/main.py
```

### Basic Workflow:

1. **Load an image** using the File menu or toolbar
2. **Apply filters or operations** from the sidebar tabs
3. **Adjust parameters** to fine-tune the results
4. **Save the processed image** when satisfied

### Edge Detection and Active Contours:

1. **Load an image** to process
2. Use the **Edge Detection panel** to detect edges, lines, and shapes
3. Use the **Active Contour panel** to:
   - Initialize a contour by clicking points on the image
   - Evolve the contour to fit object boundaries
   - Calculate metrics (perimeter, area)
   - View the chain code representation

## Project Structure
