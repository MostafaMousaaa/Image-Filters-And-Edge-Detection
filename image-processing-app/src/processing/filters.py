from scipy.ndimage import gaussian_filter, uniform_filter, median_filter
import numpy as np
import cv2

def average_filter(image, size=3):
    return uniform_filter(image, size=size)

def gaussian_filter_custom(image, sigma=1):
    return gaussian_filter(image, sigma=sigma)

def median_filter_custom(image, size=3):
    return median_filter(image, size=size)

def apply_low_pass_filter(image, filter_type='average', size=3, sigma=1):
    if filter_type == 'average':
        return average_filter(image, size)
    elif filter_type == 'gaussian':
        return gaussian_filter_custom(image, sigma)
    elif filter_type == 'median':
        return median_filter_custom(image, size)
    else:
        raise ValueError("Unknown filter type: {}".format(filter_type))