import numpy as np
import cv2

def apply_low_pass_filter(image, method, cutoff_frequency=30):
    # Convert to grayscale if color image
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    if method == 'Ideal':
        return ideal_low_pass_filter(image, cutoff_frequency)
    elif method == 'Butterworth':
        return butterworth_low_pass_filter(image, cutoff_frequency)
    else:
        raise ValueError("Unsupported filter type. Use 'ideal' or 'butterworth'.")

def ideal_low_pass_filter(image, cutoff_frequency):
    # Extract shape - fixed to handle both grayscale and color images
    if len(image.shape) == 3:
        rows, cols, _ = image.shape
    else:
        rows, cols = image.shape
        
    crow, ccol = rows // 2, cols // 2
    
    # Create the DFT of the image
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # Create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows, cols, 2), np.float32)  # Changed to include the complex component
    
    # Create a circle for the low-pass filter
    circle_mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(circle_mask, (ccol, crow), cutoff_frequency, 1, thickness=-1)
    
    # Apply the circle mask to both real and imaginary parts
    mask[:, :, 0] = circle_mask  # Real part
    mask[:, :, 1] = circle_mask  # Imaginary part
    
    # Apply mask and inverse DFT
    dft_shift_masked = dft_shift * mask
    f_ishift = np.fft.ifftshift(dft_shift_masked)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    
    # Normalize to 8-bit
    img_normalized = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(img_normalized)

def butterworth_low_pass_filter(image, cutoff_frequency, order=2):
    # Extract shape - fixed to handle both grayscale and color images
    if len(image.shape) == 3:
        rows, cols, _ = image.shape
    else:
        rows, cols = image.shape
        
    crow, ccol = rows // 2, cols // 2
    
    # Create meshgrid for distance calculation
    x = np.arange(0, rows)
    y = np.arange(0, cols)
    x, y = np.meshgrid(x, y)
    
    # Calculate distance from center
    d = np.sqrt((x - crow) ** 2 + (y - ccol) ** 2)
    
    # Create Butterworth filter
    h = 1 / (1 + (d / cutoff_frequency) ** (2 * order))
    
    # Create the DFT of the image
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # Create a mask with the Butterworth filter
    mask = np.zeros((rows, cols, 2), np.float32)
    mask[:, :, 0] = h  # Real part
    mask[:, :, 1] = h  # Imaginary part
    
    # Apply mask and inverse DFT
    dft_shift_masked = dft_shift * mask
    f_ishift = np.fft.ifftshift(dft_shift_masked)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    
    # Normalize to 8-bit
    img_normalized = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(img_normalized)

def apply_high_pass_filter(image, method, cutoff_frequency=30):
    # Convert to grayscale if color image
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    if method == 'Ideal':
        return ideal_high_pass_filter(image, cutoff_frequency)
    elif method == 'Butterworth':
        return butterworth_high_pass_filter(image, cutoff_frequency)
    else:
        raise ValueError("Unsupported filter type. Use 'ideal' or 'butterworth'.")

def ideal_high_pass_filter(image, cutoff_frequency):
    # Extract shape - fixed to handle both grayscale and color images
    if len(image.shape) == 3:
        rows, cols, _ = image.shape
    else:
        rows, cols = image.shape
        
    crow, ccol = rows // 2, cols // 2
    
    # Create the DFT of the image
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # Create a mask first, center square is 0, remaining all ones
    mask = np.ones((rows, cols, 2), np.float32)  # Changed to include the complex component
    
    # Create a circle for the high-pass filter (inverse of low-pass)
    circle_mask = np.ones((rows, cols), np.uint8)
    cv2.circle(circle_mask, (ccol, crow), cutoff_frequency, 0, thickness=-1)
    
    # Apply the circle mask to both real and imaginary parts
    mask[:, :, 0] = circle_mask  # Real part
    mask[:, :, 1] = circle_mask  # Imaginary part
    
    # Apply mask and inverse DFT
    dft_shift_masked = dft_shift * mask
    f_ishift = np.fft.ifftshift(dft_shift_masked)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    
    # Normalize to 8-bit
    img_normalized = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(img_normalized)

def butterworth_high_pass_filter(image, cutoff_frequency, order=2):
    # Extract shape - fixed to handle both grayscale and color images
    if len(image.shape) == 3:
        rows, cols, _ = image.shape
    else:
        rows, cols = image.shape
        
    crow, ccol = rows // 2, cols // 2
    
    # Create meshgrid for distance calculation
    x = np.arange(0, rows)
    y = np.arange(0, cols)
    x, y = np.meshgrid(x, y)
    
    # Calculate distance from center
    d = np.sqrt((x - crow) ** 2 + (y - ccol) ** 2)
    
    # Create Butterworth high-pass filter (inverse of low-pass)
    # Avoid division by zero
    h = np.zeros_like(d)
    non_zero = d > 0
    h[non_zero] = 1 / (1 + (cutoff_frequency / d[non_zero]) ** (2 * order))
    
    # Create the DFT of the image
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # Create a mask with the Butterworth filter
    mask = np.zeros((rows, cols, 2), np.float32)
    mask[:, :, 0] = h  # Real part
    mask[:, :, 1] = h  # Imaginary part
    
    # Apply mask and inverse DFT
    dft_shift_masked = dft_shift * mask
    f_ishift = np.fft.ifftshift(dft_shift_masked)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    
    # Normalize to 8-bit
    img_normalized = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(img_normalized)