from skimage import filters
import numpy as np

def global_threshold(image, threshold):
    """Apply global thresholding to the image."""
    binary_image = image > threshold
    return binary_image.astype(np.uint8)

def local_threshold(image, block_size, constant):
    """Apply local thresholding to the image using the mean."""
    local_thresh = filters.threshold_local(image, block_size, offset=constant)
    binary_image = image > local_thresh
    return binary_image.astype(np.uint8)