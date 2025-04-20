import numpy as np
import cv2

def gaussian_low_pass_filter(image, sigma=30):
    # Check if the image is colored or grayscale
    is_gray = len(image.shape) == 2

    rows = image.shape[0]
    cols = image.shape[1]
    center_x, center_y = rows // 2, cols // 2  # Center of the image in the frequency domain (location of low frequencies)

    # Create a grid of coordinates for frequency domain(u, v)
    x = np.arange(0, rows, 1)
    y = np.arange(0, cols, 1)
    X, Y = np.meshgrid(y - center_y, x - center_x) # Creates two 2D arrays (X => horizontal(column) coordinate of each pixel, Y => vertical(row) coordinate of each pixel)
    # The center of the frequency domain is shifted to the middle of the image by subtracting center_y and center_x
    # Therfore X & Y represent the horizontal and vertical distances of a pixel from the center (the origin) in the frequency domain

    D = np.sqrt(X**2 + Y**2)  # Computes the distance from the center (frequency domain)

    gaussian_filter = np.exp(- (D**2) / (2 * sigma**2)) # Gaussian filter equation, sigma is standard deviation and it controls cuttoff frequency (directly proportional)
    # At very low D(u, v) (0 for example) the filter's equation equals 1, therefore low frequencies pass as they are located at the center

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

def butterworth_high_pass_filter(image, cutoff_frequency=10, order=2):
    # Check if the image is grayscale or color
    is_gray = len(image.shape) == 2

    # Get the image size
    rows = image.shape[0]
    cols = image.shape[1]
    center_x, center_y = rows // 2, cols // 2  # Frequency domain center

    # Create a grid of coordinates for the frequency domain (u, v)
    x = np.arange(0, rows, 1)
    y = np.arange(0, cols, 1)
    X, Y = np.meshgrid(y - center_y, x - center_x)  # Shifted grid to center the frequencies

    # Calculate the distance from the center (frequency domain)
    D = np.sqrt(X**2 + Y**2)

    H = 1 - (1 / (1 + (D / cutoff_frequency)**(2 * order)))  # Butterworth high-pass filter transfer function

    if is_gray:
        dft_image = np.fft.fft2(image) # Computes the 2D DFT of the image

        dft_shifted = np.fft.fftshift(dft_image)  # Shift the zero frequency component to the center
        
        filtered_dft = dft_shifted * H  # Apply the Butterworth high-pass filter
        
        # Inverse shift and compute the inverse DFT to obtain the filtered image
        filtered_dft_shifted_back = np.fft.ifftshift(filtered_dft)
        inverse_dft = np.fft.ifft2(filtered_dft_shifted_back)
        
        # Take the real part and clip to valid range
        filtered_image = np.abs(inverse_dft)
        filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)

    else:
        # For color images, apply the filter to each channel separately
        bgr_image_filtered = np.zeros_like(image, dtype=np.float32)
        
        for i in range(3):
            # Process each channel (B, G, R)
            channel = image[:, :, i]
            
            # Compute the 2D DFT of the channel
            channel_dft = np.fft.fft2(channel)
            channel_dft_shifted = np.fft.fftshift(channel_dft)
            
            # Apply the Butterworth high-pass filter
            channel_filtered_dft = channel_dft_shifted * H
            
            # Inverse shift and compute the inverse DFT to obtain the filtered channel
            channel_filtered_dft_shifted_back = np.fft.ifftshift(channel_filtered_dft)
            channel_inverse_dft = np.fft.ifft2(channel_filtered_dft_shifted_back)
            
            # Store the filtered channel
            bgr_image_filtered[:, :, i] = np.abs(channel_inverse_dft)
        
        # Clip the values to the valid range and convert back to uint8
        filtered_image = np.clip(bgr_image_filtered, 0, 255).astype(np.uint8)

    return filtered_image