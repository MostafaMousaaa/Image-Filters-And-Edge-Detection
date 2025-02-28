import numpy as np
import cv2

def apply_low_pass_filter(image, filter_type='ideal', cutoff_frequency=30):
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