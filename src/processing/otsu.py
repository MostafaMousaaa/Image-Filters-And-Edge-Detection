import numpy as np

def otsu_threshold(image):
    
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