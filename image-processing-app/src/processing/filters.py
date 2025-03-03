from scipy.ndimage import gaussian_filter, uniform_filter, median_filter
import numpy as np
import cv2

def average_filter(image, size=3):
    noisy_img = np.copy(image)
    half_kernel_size = size // 2
    avg_kernel = np.ones((size, size))
    avg_kernel /= np.sum(avg_kernel)
    print(f"average_kernel: {avg_kernel}")
    
    for curr_channel in range(noisy_img.shape[2]):  # loop over the 3 channels as it's a colored img
        # looping over the image pixels except the boundaries to update pixel values using the avg kernel
        for row in range(half_kernel_size, noisy_img.shape[0] - half_kernel_size):   
            for col in range (half_kernel_size, noisy_img.shape[1] - half_kernel_size):
                used_range = noisy_img[ row-half_kernel_size : row+half_kernel_size+1 , col-half_kernel_size : col+half_kernel_size+1, curr_channel] # slicing to take the pixel and its 8 neighbouring pixels in each loop.
                filtered_pixel = np.sum(avg_kernel * used_range)  # getting the new pixel value
                
                noisy_img[row][col][curr_channel] = filtered_pixel  # updating the current pixel in the noisy image
    return noisy_img
    

def gaussian_filter_custom(image, size, sigma=1):
    noisy_img = np.copy(image)            
        
    #gaussian_kernel = np.zeros((3, 3))
    half_kernel_size = size // 2  # Half-size for symmetry
    x, y = np.meshgrid(np.arange(-half_kernel_size, half_kernel_size+1), np.arange(-half_kernel_size, half_kernel_size+1))
    
    # apply the Gaussian formula
    gaussian_kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    # normalize so the sum is 1 by dividing by the sum of pixel values
    gaussian_kernel /= np.sum(gaussian_kernel)
    print(f"gaussian kernel: {gaussian_kernel}")
    
    for curr_channel in range(noisy_img.shape[2]):  # loop over the 3 channels as it's a colored img
        # looping over the image pixels except the boundaries to update pixel values using the gaussian kernel
        for row in range(half_kernel_size, noisy_img.shape[0] - half_kernel_size):   
            for col in range (half_kernel_size, noisy_img.shape[1] - half_kernel_size):
                used_range = noisy_img[ row-half_kernel_size : row+half_kernel_size+1 , col-half_kernel_size : col+half_kernel_size+1, curr_channel] # slicing to take the pixel and its neighbouring pixels in each loop.
                filtered_pixel = np.sum(gaussian_kernel * used_range)  # getting the new pixel value
                
                noisy_img[row][col][curr_channel] = np.clip(0, 255, filtered_pixel)  # updating the current pixel in the noisy image
    return noisy_img

def median_filter_custom(image, size=3):
    noisy_img = np.copy(image)
    half_kernel_size = size // 2
     
    for curr_channel in range(noisy_img.shape[2]):  # loop over the 3 channels as it's a colored img
        for row in range(half_kernel_size, noisy_img.shape[0] - half_kernel_size):   
            for col in range (half_kernel_size, noisy_img.shape[1] - half_kernel_size):
                used_range = noisy_img[ row - half_kernel_size : row+half_kernel_size+1 , col-half_kernel_size : col+half_kernel_size+1, curr_channel]  # taking a size x size range
                flattened_used_range = used_range.flatten() # flattening to sort and get the median
                sorted_flattened_used_range = np.sort(flattened_used_range)
                middle_value = sorted_flattened_used_range[len(sorted_flattened_used_range)//2] # the median is the middle value of the sorted array
                noisy_img[row][col][curr_channel] = middle_value # updating the current pixel's value in the noisy img

    return noisy_img

def apply_low_pass_filter(image, filter_type='Average', size=3, sigma=1):
    if filter_type == 'Average':
        return average_filter(image, size)
    elif filter_type == 'Gaussian': 
        return gaussian_filter_custom(image, size, sigma)
    elif filter_type == 'Median':
        return median_filter_custom(image, size)
    else:
        raise ValueError("Unknown filter type: {}".format(filter_type))