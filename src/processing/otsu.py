import numpy as np

def global_otsu_threshold(image):
    
    # Calculate histogram
    freqs, bins = np.histogram(image.flatten(), 256, [0, 256])  # distributing intensities from 0 to 255 "[0, 256]" on 256 bins, so each bin takes 1 value
    
    # Calculate probability for each intensity level
    p_i = freqs / freqs.sum()
    
    # Initialize variables
    max_variance = 0
    optimal_threshold = 0
    
    # For each possible threshold T, calculate variance
    for t in range(1, 256):
        # P(k) - probability of class occurrence
        p_0 = np.sum(p_i[:t])
        p_1 = np.sum(p_i[t:])
        
        # Avoid division by zero
        if p_0 == 0 or p_1 == 0:
            continue
        
        # Calculate class means
        m_0 = np.sum(np.arange(t) * p_i[:t]) / p_0
        m_1 = np.sum(np.arange(t, 256) * p_i[t:]) / p_1
        
        # Calculate global mean
        m_g = p_0 * m_0 + p_1 * m_1
        
        # Calculate between-class variance
        variance = p_0 * (m_0 - m_g)**2 + p_1 * (m_1 - m_g)**2
        
        # Update threshold if we found a higher variance
        if variance > max_variance:
            max_variance = variance
            optimal_threshold = t
    
    # Apply threshold to image
    thresholded_image = (image > optimal_threshold).astype(np.uint8) * 255
    
    return optimal_threshold, thresholded_image



def local_otsu_threshold(image, window_size = 35, step = None):
    if step is None:
        step = window_size // 2
    
    # Create output image initialized with zeros
    output = np.zeros_like(image)
    # Create a counter array to track how many times each pixel is processed
    counts = np.zeros_like(image, dtype=float)
    
    # Half window size for easier calculations
    half_window = window_size // 2
    
    # For each window position in the image
    for y in range(0, image.shape[0], step):
        for x in range(0, image.shape[1], step):
            # Define window boundaries with proper edge handling
            y_start = max(0, y - half_window)
            y_end = min(image.shape[0], y + half_window + 1)
            x_start = max(0, x - half_window)
            x_end = min(image.shape[1], x + half_window + 1)
            
            # Extract the window
            window = image[y_start:y_end, x_start:x_end]
            
            # Apply Otsu to the window
            threshold, _ = global_otsu_threshold(window)
            
            # Apply threshold to the same window region
            region = image[y_start:y_end, x_start:x_end]
            thresholded_region = (region > threshold).astype(np.uint8) * 255
            
            # Accumulate thresholded values in output
            output[y_start:y_end, x_start:x_end] += thresholded_region
            # Count how many times each pixel is processed
            counts[y_start:y_end, x_start:x_end] += 1
    
    # Average the accumulated thresholds where pixels are processed multiple times
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(output, counts)
        result = np.nan_to_num(result)
    
    # Convert result to binary (0 or 255) based on most frequent value at each pixel
    return (result > 127.5).astype(np.uint8) * 255
    