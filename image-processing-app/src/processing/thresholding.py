import numpy as np
import cv2

def global_threshold(image, threshold):
    """
    Apply global thresholding to an image
    
    Args:
        image: Input image
        threshold: Threshold value (0-255)
    
    Returns:
        Binary image where pixels >= threshold are set to 255 and others to 0
    """
    # Convert to grayscale if the image is color
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create output binary image
    binary_image = np.zeros_like(image)
    
    # Apply thresholding
    binary_image[image >= threshold] = 255
    
    return binary_image

def local_threshold(image, block_size, constant):
    """
    Apply local adaptive thresholding to an image using mean value of neighborhood
    
    Args:
        image: Input image
        block_size: Size of the neighborhood area (must be odd)
        constant: Constant subtracted from the mean
    
    Returns:
        Binary image where pixels >= local_threshold are set to 255 and others to 0
    """
    # Convert to grayscale if the image is color
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Ensure block size is odd
    if block_size % 2 == 0:
        block_size += 1
    
    # Get image dimensions
    rows, cols = image.shape
    
    # Create output binary image
    binary_image = np.zeros_like(image)
    
    # Apply padding to handle boundary
    pad = block_size // 2
    padded_image = np.pad(image, pad, mode='reflect')
    
    # For each pixel in the image
    for i in range(rows):
        for j in range(cols):
            # Get the neighborhood
            neighborhood = padded_image[i:i+block_size, j:j+block_size]
            
            # Calculate the local threshold
            local_threshold = np.mean(neighborhood) - constant
            
            # Apply thresholding
            if image[i, j] >= local_threshold:
                binary_image[i, j] = 255
    
    return binary_image

def local_threshold_optimized(image, block_size, constant):
    """
    Apply local adaptive thresholding to an image using mean value of neighborhood
    This is an optimized version that uses integral images for faster computation
    
    Args:
        image: Input image
        block_size: Size of the neighborhood area (must be odd)
        constant: Constant subtracted from the mean
    
    Returns:
        Binary image where pixels >= local_threshold are set to 255 and others to 0
    """
    # Convert to grayscale if the image is color
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Ensure block size is odd
    if block_size % 2 == 0:
        block_size += 1
    
    # Get image dimensions
    rows, cols = image.shape
    
    # Create output binary image
    binary_image = np.zeros_like(image)
    
    # Compute integral image
    integral_image = np.cumsum(np.cumsum(image.astype(np.float32), axis=0), axis=1)
    
    # Padding for integral image
    pad = block_size // 2
    
    # For each pixel in the image
    for i in range(rows):
        for j in range(cols):
            # Calculate coordinates for the neighborhood box
            y_start = max(0, i - pad)
            y_end = min(rows - 1, i + pad)
            x_start = max(0, j - pad)
            x_end = min(cols - 1, j + pad)
            
            # Count pixels in the neighborhood
            count = (y_end - y_start + 1) * (x_end - x_start + 1)
            
            # Calculate sum using integral image
            # Formula: sum = I(x2,y2) - I(x2,y1-1) - I(x1-1,y2) + I(x1-1,y1-1)
            # Where I is the integral image
            sum_val = integral_image[y_end, x_end]
            
            if y_start > 0:
                sum_val -= integral_image[y_start-1, x_end]
            
            if x_start > 0:
                sum_val -= integral_image[y_end, x_start-1]
            
            if y_start > 0 and x_start > 0:
                sum_val += integral_image[y_start-1, x_start-1]
            
            # Calculate mean value
            mean = sum_val / count
            
            # Apply thresholding
            if image[i, j] >= mean - constant:
                binary_image[i, j] = 255
    
    return binary_image

def otsu_threshold(image):
    """
    Determine optimal threshold using Otsu's method and apply it
    
    Args:
        image: Input image
    
    Returns:
        Binary image and the calculated threshold value
    """
    # Convert to grayscale if the image is color
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate histogram
    hist = np.zeros(256, dtype=np.float32)
    for i in range(256):
        hist[i] = np.sum(image == i)
    
    # Normalize histogram
    hist = hist / np.sum(hist)
    
    pixel_range = np.arange(256)
    
    # Calculate cumulative sums
    cumsum = np.cumsum(hist)
    
    # Calculate cumulative means
    cumulative_mean = np.cumsum(hist * pixel_range)
    
    # Compute between-class variance
    global_mean = cumulative_mean[-1]
    between_variance = np.zeros(256)
    
    for t in range(256):
        if cumsum[t] == 0 or cumsum[t] == 1:
            continue
        
        weight_background = cumsum[t]
        weight_foreground = 1.0 - weight_background
        
        if weight_background == 0 or weight_foreground == 0:
            continue
            
        mean_background = cumulative_mean[t] / weight_background
        mean_foreground = (global_mean - cumulative_mean[t]) / weight_foreground
        
        between_variance[t] = weight_background * weight_foreground * \
                             (mean_background - mean_foreground) ** 2
    
    # Find the threshold that maximizes between-class variance
    optimal_threshold = np.argmax(between_variance)
    
    # Apply the threshold
    binary_image = global_threshold(image, optimal_threshold)
    
    return binary_image, optimal_threshold