import numpy as np
import cv2

def gaussian_low_pass_filter(image, sigma=10):
    # Check if the image is colored or grayscale
    is_gray = len(image.shape) == 2

    rows = image.shape[0]
    cols = image.shape[1]
    center_x, center_y = rows // 2, cols // 2  # Center of the image in the frequency domain (location of low frequencies)

    # Create a grid of coordinates for frequency domain(u, v)
    x = np.arange(0, rows, 1)
    y = np.arange(0, cols, 1)
    X, Y = np.meshgrid(y - center_y, x - center_x) # Creates two 2D arrays (X => horizontal(column) coordinate of each pixel, Y => vertical(row) coordinate of each pixel)
    # the center of the frequency domain is shifted to the middle of the image by subtracting center_y and center_x
    
    D = np.sqrt(X**2 + Y**2)  # Computes the distance from the center (frequency domain)
    gaussian_filter = np.exp(- (D**2) / (2 * sigma**2)) # Gaussian filter equation, sigma is standard deviation and it controls cuttoff frequency (directly proportional)

    if is_gray:
        dftImage = np.fft.fft2(image)  # Computes 2D DFT of the image
        
        dftShiftedImage = np.fft.fftshift(dftImage) # Shifts the zero frequency component to the center of the image

        filteredDftImg = dftShiftedImage * gaussian_filter # Apply the Gaussian filter in the frequency domain (convolution in spatial domain is multiplication in frequency domain)

        filteredUnshiftedDftImg = np.fft.ifftshift(filteredDftImg) # Shifts back to original corners (inverse shift)

        inverseDftImg = np.fft.ifft2(filteredUnshiftedDftImg) # Computes the inverse DFT to go back to the spatial domain

        filtered_image = np.abs(inverseDftImg) # Takes the real part of the inverse DFT (because the result is complex)

        filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8) # clips the values and converts back to uint8
    else:
        # Separate the channels in the color image (BGR)
        bgr_image_filtered = np.zeros_like(image, dtype=np.float32)

        for i in range(3):
            channel = image[:, :, i]
            # Repeat the process for each channel separately
            channel_dft = np.fft.fft2(channel)
            channel_dft_shift = np.fft.fftshift(channel_dft)
            channel_filtered_dft = channel_dft_shift * gaussian_filter
            channel_filtered_dft_shifted_back = np.fft.ifftshift(channel_filtered_dft)
            channel_inverse_dft = np.fft.ifft2(channel_filtered_dft_shifted_back)
            bgr_image_filtered[:, :, i] = np.abs(channel_inverse_dft)

        # Clip the values to valid range [0, 255] and convert to uint8
        filtered_image = np.clip(bgr_image_filtered, 0, 255).astype(np.uint8)
    

    return filtered_image

"""def apply_low_pass_filter(image, filter_type='ideal', cutoff_frequency=30):
    if filter_type == 'ideal':
        return ideal_low_pass_filter(image, cutoff_frequency)
    elif filter_type == 'butterworth':
        return butterworth_low_pass_filter(image, cutoff_frequency)
    else:
        raise ValueError("Unsupported filter type. Use 'ideal' or 'butterworth'.")

def ideal_low_pass_filter(image, cutoff_frequency):
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), cutoff_frequency, 1, thickness=-1)

    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    dft_shift *= mask

    f_ishift = np.fft.ifftshift(dft_shift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    return np.uint8(img_back)

def butterworth_low_pass_filter(image, cutoff_frequency, order=2):
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    x = np.arange(0, rows)
    y = np.arange(0, cols)
    x, y = np.meshgrid(x, y)
    d = np.sqrt((x - crow) ** 2 + (y - ccol) ** 2)
    h = 1 / (1 + (d / cutoff_frequency) ** (2 * order))

    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    dft_shift *= h

    f_ishift = np.fft.ifftshift(dft_shift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    return np.uint8(img_back)
"""

def apply_high_pass_filter(image, filter_type='ideal', cutoff_frequency=30):
    if filter_type == 'ideal':
        return ideal_high_pass_filter(image, cutoff_frequency)
    elif filter_type == 'butterworth':
        return butterworth_high_pass_filter(image, cutoff_frequency)
    else:
        raise ValueError("Unsupported filter type. Use 'ideal' or 'butterworth'.")

def ideal_high_pass_filter(image, cutoff_frequency):
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), cutoff_frequency, 0, thickness=-1)

    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    dft_shift *= mask

    f_ishift = np.fft.ifftshift(dft_shift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    return np.uint8(img_back)

def butterworth_high_pass_filter(image, cutoff_frequency, order=2):
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    x = np.arange(0, rows)
    y = np.arange(0, cols)
    x, y = np.meshgrid(x, y)
    d = np.sqrt((x - crow) ** 2 + (y - ccol) ** 2)
    h = 1 / (1 + (cutoff_frequency / d) ** (2 * order))
    h[d == 0] = 0  # Avoid division by zero

    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    dft_shift *= h

    f_ishift = np.fft.ifftshift(dft_shift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    return np.uint8(img_back)