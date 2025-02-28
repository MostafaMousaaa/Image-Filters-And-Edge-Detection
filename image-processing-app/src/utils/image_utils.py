import cv2
import numpy as np
import os

def load_image(file_path):
    """
    Load an image from the specified file path.
    
    Args:
        file_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: Loaded image as a NumPy array, or None if loading fails
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
        
    try:
        # Read the image using OpenCV
        image = cv2.imread(file_path)
        
        # Check if image was successfully loaded
        if image is None:
            print(f"Failed to load image: {file_path}")
            return None
            
        return image
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        return None

def save_image(filepath, image):
    cv2.imwrite(filepath, image)

def convert_to_grayscale(image):
    """
    Convert an image to grayscale.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: Grayscale image
    """
    if image is None:
        return None
        
    # If already grayscale, return as is
    if len(image.shape) == 2:
        return image
        
    # Convert to grayscale
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def normalize_image(image):
    """
    Normalize the image to the range [0, 255].
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: Normalized image
    """
    if image is None:
        return None
    
    # Normalize to [0, 255]
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)

def equalize_histogram(image):
    """
    Equalize the histogram of an image.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: Image with equalized histogram
    """
    if image is None:
        return None
    
    # Check if image is grayscale
    if len(image.shape) == 2:
        return cv2.equalizeHist(image)
    
    # For color images, convert to YUV and equalize Y channel
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv_image[:,:,0] = cv2.equalizeHist(yuv_image[:,:,0])
    return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)