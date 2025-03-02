from scipy.ndimage import gaussian_filter, uniform_filter, median_filter
import numpy as np
import cv2

def average_filter(image, size=3):
    noisy_img = np.copy(image)
        
    avg_kernel = 1/9 * np.array([[1, 1, 1], 
                                    [1, 1, 1], 
                                    [1, 1, 1]])
    for curr_channel in range(noisy_img.shape[2]):  # loop over the 3 channels as it's a colored img
        # looping over the image pixels except the boundaries to update pixel values using the avg kernel
        for row in range(1, noisy_img.shape[0] - 1):   
            for col in range (1, noisy_img.shape[1] - 1):
                used_range = noisy_img[ row-1 : row+2 , col-1 : col+2, curr_channel] # slicing to take the pixel and its 8 neighbouring pixels in each loop.
                filtered_pixel = np.sum(avg_kernel * used_range)  # getting the new pixel value
                
                noisy_img[row][col][curr_channel] = filtered_pixel  # updating the current pixel in the noisy image
    return noisy_img
    

def gaussian_filter_custom(image, sigma=1):
    noisy_img = np.copy(image)
        
    gaussian_kernel = (1/16) * np.array([[1, 2, 1], 
                                            [2, 4, 2], 
                                            [1, 2, 1]])
    for curr_channel in range(noisy_img.shape[2]):  # loop over the 3 channels as it's a colored img
        # looping over the image pixels except the boundaries to update pixel values using the gaussian kernel
        for row in range(1, noisy_img.shape[0] - 1):   
            for col in range (1, noisy_img.shape[1] - 1):
                used_range = noisy_img[ row-1 : row+2 , col-1 : col+2, curr_channel] # slicing to take the pixel and its 8 neighbouring pixels in each loop.
                filtered_pixel = np.sum(gaussian_kernel * used_range)  # getting the new pixel value
                
                noisy_img[row][col][curr_channel] = np.clip(0, 255, filtered_pixel)  # updating the current pixel in the noisy image
        return noisy_img

def median_filter_custom(image, size=3):
    noisy_img = np.copy(image)
        
    for curr_channel in range(noisy_img.shape[2]):  # loop over the 3 channels as it's a colored img
        for row in range(1, noisy_img.shape[0] - 1):   
            for col in range (1, noisy_img.shape[1] - 1):
                used_range = noisy_img[ row-1 : row+2 , col-1 : col+2, curr_channel]  # taking a 3x3 range
                flattened_used_range = used_range.flatten() # flattening to sort and get the median
                sorted_flattened_used_range = np.sort(flattened_used_range)
                middle_value = sorted_flattened_used_range[len(sorted_flattened_used_range)//2] # the median is the middle value of the sorted array
                noisy_img[row][col][curr_channel] = middle_value # updating the current pixel's value in the noisy img

    return noisy_img

def apply_low_pass_filter(image, filter_type='Average', size=3, sigma=1):
    if filter_type == 'Average':
        return average_filter(image, size)
    elif filter_type == 'Gaussian': 
        return gaussian_filter_custom(image, sigma)
    elif filter_type == 'Median':
        return median_filter_custom(image, size)
    else:
        raise ValueError("Unknown filter type: {}".format(filter_type))