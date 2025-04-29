'''
This module implements optimal and spectral thresholding algorithms for image segmentation:

1. Global Optimal Iterative Thresholding:
   Takes a grayscale image and iteratively finds the optimal threshold
   Returns the threshold value, binary image, and count of pixels above threshold

2. Local Optimal Iterative Thresholding:
   Takes a grayscale image and a block size
   Applies thresholding locally to each block using the global method
   Returns the binary image

3. Global Spectral Thresholding:
   Takes a grayscale image and segments it based on histogram valley detection
   Returns the threshold values, binary image, and count of pixels above threshold

4. Local Spectral Thresholding:
   Takes a grayscale image and a block size
   Applies spectral thresholding locally to each block
   Returns the binary image

5. Multi-level Spectral Segmentation:
   Takes a grayscale image and number of desired levels
   Segments the image into multiple levels using spectral thresholding
   Returns a multi-level segmented image
'''

import numpy as np
import cv2
from typing import Tuple, Any, List
from scipy.signal import find_peaks, savgol_filter

def global_optimal_iterative_thresholding(input_img: np.ndarray) -> Tuple[int, np.ndarray, int]:
    """
    Apply global optimal iterative thresholding to an image.

    Parameters:
    input_img (np.ndarray): Input grayscale image.

    Returns:
    Tuple[int, np.ndarray, int]: Tuple containing the optimal threshold value,
                                  the binary image, and the number of pixels above the threshold.
    """
    # Ensure the input image is valid and in grayscale
    if input_img is None:
        raise ValueError("Input image is None.")
    if len(input_img.shape) != 2:
        raise ValueError("Input image must be a grayscale image.")

    # Initialize variables
    threshold = 0
    max_iter = 100
    epsilon = 1e-5

    # Initial guess for the threshold
    T = np.mean(input_img)

    for _ in range(max_iter):
        # Calculate the mean of pixels below and above the current threshold
        below_mean = input_img[input_img < T].mean() if np.any(input_img < T) else 0
        above_mean = input_img[input_img >= T].mean() if np.any(input_img >= T) else 0

        # Update the threshold
        new_T = (below_mean + above_mean) / 2

        # Check for convergence
        if abs(new_T - T) < epsilon:
            break

        T = new_T

    # Create binary image based on the final threshold
    binary_image = (input_img >= T).astype(np.uint8) * 255

    # Count pixels above the threshold
    num_pixels_above_threshold = np.sum(binary_image > 0)

    return int(T), binary_image, num_pixels_above_threshold


def local_optimal_iterative_thresholding(input_img: np.ndarray, block_dim: int) -> np.ndarray:
    """
    Apply local optimal iterative thresholding to an image.

    Parameters:
    input_img (np.ndarray): Input grayscale image.
    block_dim (int): Block dimension (e.g., 5 means 5 x 5).

    Returns:
    np.ndarray: Binary image after local thresholding.
    """
    # Ensure the input image is valid and in grayscale
    if input_img is None:
        raise ValueError("Input image is None.")
    if len(input_img.shape) != 2:
        raise ValueError("Input image must be a grayscale image.")

    # Get the dimensions of the input image
    rows, cols = input_img.shape

    # Create an output binary image
    binary_image = np.zeros_like(input_img, dtype=np.uint8)

    # Iterate over the image in blocks
    for i in range(0, rows, block_dim):
        for j in range(0, cols, block_dim):
            # Define the block boundaries
            block = input_img[i:i + block_dim, j:j + block_dim]

            # Calculate the optimal threshold for the current block
            T = global_optimal_iterative_thresholding(block)[0]

            # Apply the threshold to the block
            binary_image[i:i + block_dim, j:j + block_dim] = (block >= T).astype(np.uint8) * 255

    return binary_image


def global_spectral_thresholding(input_img: np.ndarray, num_thresholds: int = 1) -> Tuple[List[int], np.ndarray, int]:
    """
    Apply global spectral thresholding based on histogram valley detection.
    
    Parameters:
    input_img (np.ndarray): Input grayscale image.
    num_thresholds (int): Number of thresholds to return (for multi-modal segmentation)
    
    Returns:
    Tuple[List[int], np.ndarray, int]: Tuple containing list of optimal threshold values,
                                  the binary image (using first threshold), and the number of pixels above the first threshold.
    """
    # Ensure the input image is valid and in grayscale
    if input_img is None:
        raise ValueError("Input image is None.")
    if len(input_img.shape) != 2:
        raise ValueError("Input image must be a grayscale image.")
    
    # Calculate histogram
    hist, bins = np.histogram(input_img.flatten(), 256, [0, 256])
    
    # Smooth the histogram to remove noise
    hist_smooth = savgol_filter(hist, 15, 2)
    
    # Normalize the histogram
    hist_norm = hist_smooth / np.sum(hist_smooth)
    
    # Calculate histogram variance for adaptive window sizing
    hist_variance = np.var(hist_norm)
    window_size = max(5, min(50, int(hist_variance * 1000)))
    
    # Find valleys (local minima) in the smoothed histogram
    # Valleys are where spectral thresholds often exist
    peaks, _ = find_peaks(-hist_norm)
    
    # If no valleys found, use the otsu method as fallback
    if len(peaks) == 0:
        print("No valleys found")
    
    # Evaluate the "deepness" of each valley by its neighboring peaks
    valley_scores = []
    for peak in peaks:
        # Skip boundaries
        if peak <= 5 or peak >= 250:
            continue
            
        # Calculate the "deepness" with adaptive window size
        left_max = np.max(hist_norm[max(0, peak-window_size):peak])
        right_max = np.max(hist_norm[peak+1:min(255, peak+window_size+1)])
        valley_score = hist_norm[peak] / ((left_max + right_max) / 2)
        valley_scores.append((peak, valley_score))
    
    # Find the best thresholds based on deepest valleys
    thresholds = []
    if valley_scores:
        # Sort by score (smaller is better as it means deeper valley)
        valley_scores.sort(key=lambda x: x[1])
        # Take the top N valleys as thresholds
        thresholds = [v[0] for v in valley_scores[:min(num_thresholds, len(valley_scores))]]
        thresholds.sort()  # Sort thresholds in ascending order
    else:
        # If no valid valleys, use the overall minimum
        thresholds = [peaks[np.argmin(hist_norm[peaks])]]
    
    # Create binary image using the first threshold
    first_threshold = thresholds[0]
    binary_image = (input_img >= first_threshold).astype(np.uint8) * 255
    
    # Count pixels above the first threshold
    pixels_above = np.sum(binary_image > 0)
    
    return thresholds, binary_image, pixels_above


def local_spectral_thresholding(input_img: np.ndarray, block_dim: int, num_thresholds: int = 1) -> np.ndarray:
    """
    Apply local spectral thresholding to an image.
    
    Parameters:
    input_img (np.ndarray): Input grayscale image.
    block_dim (int): Block dimension (e.g., 5 means 5 x 5).
    num_thresholds (int): Number of thresholds to use within each block.
    
    Returns:
    np.ndarray: Binary image after local spectral thresholding.
    """
    # Ensure the input image is valid and in grayscale
    if input_img is None:
        raise ValueError("Input image is None.")
    if len(input_img.shape) != 2:
        raise ValueError("Input image must be a grayscale image.")
    
    # Get the dimensions of the input image
    rows, cols = input_img.shape
    
    # Create an output binary image
    binary_image = np.zeros_like(input_img, dtype=np.uint8)
    
    # Iterate over the image in blocks
    for i in range(0, rows, block_dim):
        for j in range(0, cols, block_dim):
            # Define the block boundaries
            block = input_img[i:i + block_dim, j:j + block_dim]
            
            # Skip very small blocks at the edges
            if block.size <= 10:
                binary_image[i:i + block_dim, j:j + block_dim] = 0
                continue
                
            try:
                # Calculate the spectral thresholds for the current block
                thresholds, _, _ = global_spectral_thresholding(block, num_thresholds)
                
                # Use first threshold for binary image
                first_threshold = thresholds[0] if thresholds else np.mean(block)
                binary_image[i:i + block_dim, j:j + block_dim] = (block >= first_threshold).astype(np.uint8) * 255
            except Exception as e:
                # If there's an error processing the block, use a simple mean threshold
                T = np.mean(block)
                binary_image[i:i + block_dim, j:j + block_dim] = (block >= T).astype(np.uint8) * 255
    
    return binary_image


def multi_level_spectral_segmentation(input_img: np.ndarray, num_levels: int = 3) -> np.ndarray:
    """
    Segment image into multiple levels using spectral thresholding.
    
    Parameters:
    input_img (np.ndarray): Input grayscale image.
    num_levels (int): Number of levels in the segmented image (num_thresholds + 1)
    
    Returns:
    np.ndarray: Multi-level segmented image where each region has a different intensity
    """
    # Ensure the input image is valid and in grayscale
    if input_img is None:
        raise ValueError("Input image is None.")
    if len(input_img.shape) != 2:
        raise ValueError("Input image must be a grayscale image.")
    
    # Get thresholds using spectral thresholding
    thresholds, _, _ = global_spectral_thresholding(input_img, num_levels - 1)
    
    # Create a multi-level image
    segmented_img = np.zeros_like(input_img)
    
    if len(thresholds) == 0:
        # If no thresholds found, just return the original image normalized to 0-255
        return cv2.normalize(input_img, None, 0, 255, cv2.NORM_MINMAX)
    
    # Sort thresholds for consistent level assignment
    thresholds.sort()
    
    # Create a multi-level segmentation
    # First, set everything to the lowest level (0)
    segmented_img.fill(0)
    
    # For each threshold, assign pixels above that threshold to the corresponding level
    intensity_step = 255 // num_levels
    
    for i, thresh in enumerate(thresholds):
        # Calculate intensity for this level (evenly spaced from 0 to 255)
        intensity = intensity_step * (i + 1)
        
        # Assign pixels above this threshold to this intensity level
        segmented_img[input_img >= thresh] = intensity
    
    # Highest intensity level for areas above the highest threshold
    if len(thresholds) > 0:
        segmented_img[input_img >= thresholds[-1]] = 255
    
    return segmented_img


def local_multi_level_segmentation(input_img: np.ndarray, block_dim: int, num_levels: int = 3) -> np.ndarray:
    """
    Apply local multi-level spectral segmentation to an image.
    
    Parameters:
    input_img (np.ndarray): Input grayscale image.
    block_dim (int): Block dimension (e.g., 5 means 5 x 5).
    num_levels (int): Number of levels in the segmented image.
    
    Returns:
    np.ndarray: Multi-level segmented image.
    """
    # Ensure the input image is valid and in grayscale
    if input_img is None:
        raise ValueError("Input image is None.")
    if len(input_img.shape) != 2:
        raise ValueError("Input image must be a grayscale image.")
    
    # Get the dimensions of the input image
    rows, cols = input_img.shape
    
    # Create an output segmented image
    segmented_image = np.zeros_like(input_img, dtype=np.uint8)
    
    # Iterate over the image in blocks
    for i in range(0, rows, block_dim):
        for j in range(0, cols, block_dim):
            # Define the block boundaries
            block = input_img[i:i + block_dim, j:j + block_dim]
            block_rows = min(block_dim, rows - i)
            block_cols = min(block_dim, cols - j)
            
            # Skip very small blocks at the edges
            if block.size <= 10:
                continue
                
            try:
                # Create a multi-level segmentation for this block
                block_segmented = multi_level_spectral_segmentation(block, num_levels)
                segmented_image[i:i + block_rows, j:j + block_cols] = block_segmented[:block_rows, :block_cols]
            except Exception as e:
                # If segmentation fails, use a simple binary threshold
                T = np.mean(block)
                binary_block = (block >= T).astype(np.uint8) * 255
                segmented_image[i:i + block_rows, j:j + block_cols] = binary_block[:block_rows, :block_cols]
    
    return segmented_image


