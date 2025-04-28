'''
This module implements optimal thresholding algorithms for image segmentation:

1-Global Optimal Iterative Thresholding:
   Takes a grayscale image and iteratively finds the optimal threshold
   Returns the threshold value, binary image, and count of pixels above threshold

2-Local Optimal Iterative Thresholding:
   Takes a grayscale image and a block size
   Applies thresholding locally to each block using the global method
   Returns the binary image
'''

import numpy as np
import cv2
from typing import Tuple, Any

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


