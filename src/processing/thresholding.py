import numpy as np
import cv2

def global_threshold(image, threshold):
    thresholded_image = np.zeros_like(image) # same shape and size as image but all the values are zeros

    if len(image.shape) == 2:  # Grayscale image
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                if image[row][col] > threshold: # Apply binary thresholding, where the value is set to either 255 (white) or 0 (black)
                    thresholded_image[row][col] = 255
                else:
                    thresholded_image[row][col] = 0

    elif len(image.shape) == 3:  # Colored image (BGR)
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                for component in range(image.shape[2]):  # Iterate through channels (BGR)
                    if image[row][col][component] > threshold:
                        thresholded_image[row][col][component] = 255
                    else:
                        thresholded_image[row][col][component] = 0

    return thresholded_image

def local_threshold(image, block_size):
    if ((block_size < 3) or (block_size > 99) or (block_size % 2 == 0)): # Ensure block size is odd so that we always have a center pixel in our block
        return "Block Size must be an odd number between 3 & 99"
    
    # Mean based thresholding, where for each block, the mean is the threshold value
    thresholded_image = np.zeros_like(image)
    height = image.shape[0]
    width = image.shape[1]

    # Used to compute the neighbors around the center pixel (pixel at row i and column j)
    half_block = block_size // 2

    # Iterate over each pixel in the image (excluding borders)
    for row in range(height):
        for col in range(width):
            # Get the neighborhood region (block) around the current pixel
            y_min = max(row - half_block, 0) # top boundary
            y_max = min(row + half_block + 1, height) # bottom boundary (1 is added as ending is exclusive in array slicing)
            x_min = max(col - half_block, 0) # left boundary
            x_max = min(col + half_block + 1, width) # right boundary

            if len(image.shape) == 2: # Grayscale image
                # Extract the local block
                local_block = image[y_min:y_max, x_min:x_max]

                # Compute the mean of the local block
                local_mean = np.mean(local_block)

                if image[row][col] > local_mean:
                    thresholded_image[row][col] = 255
                else:
                    thresholded_image[row][col] = 0
            
            elif len(image.shape) == 3:  # Colored image (BGR)
                # Extract the local block for each channel
                local_block_b = image[y_min:y_max, x_min:x_max, 0] # Blue channel
                local_block_g = image[y_min:y_max, x_min:x_max, 1] # Green channel
                local_block_r = image[y_min:y_max, x_min:x_max, 2]  # Red channel

                # Compute the mean for each color channel
                local_mean_b = np.mean(local_block_b)
                local_mean_g = np.mean(local_block_g)
                local_mean_r = np.mean(local_block_r)
                local_means = [local_mean_b, local_mean_g, local_mean_r]

                for channel in range(3):
                    if image[row][col][channel] > local_means[channel]:
                        thresholded_image[row][col][channel] = 255
                    else:
                        thresholded_image[row][col][channel] = 0

    return thresholded_image

"""
def local_threshold_optimized(image, block_size, constant):
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
 """
