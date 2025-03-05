import numpy as np

def global_threshold(image, threshold):
    thresholded_image = np.zeros_like(image) # same shape and size as image but all the values are zeros

    if len(image.shape) == 2:  # Grayscale image
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                if image[row][col] > threshold: # Apply binary thresholding, where the value is set to either 255 (white) or 0 (black)
                    thresholded_image[row][col] = 255
                else:
                    thresholded_image[row][col] = 0

    elif len(image.shape) == 3:  # Colored image (BGR)
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                for component in range(image.shape[2]):  # Iterate through channels (BGR)
                    if image[row][col][component] > threshold:
                        thresholded_image[row][col][component] = 255
                    else:
                        thresholded_image[row][col][component] = 0

    return thresholded_image

def local_threshold(image, block_size):
    if ((block_size < 3) or (block_size > 99) or (block_size % 2 == 0)): # Ensure block size is odd so that we always have a center pixel in our block
        return "Block Size must be an odd number between 3 & 99"
    
    # Mean based thresholding, where for each block, the mean is the threshold value
    thresholded_image = np.zeros_like(image)
    height = image.shape[0]
    width = image.shape[1]

    # Used to compute the neighbors around the center pixel (pixel at row i and column j)
    half_block = block_size // 2

    # Iterate over each pixel in the image (excluding borders)
    for row in range(height):
        for col in range(width):
            # Get the neighborhood region (block) around the current pixel
            y_min = max(row - half_block, 0) # top boundary
            y_max = min(row + half_block + 1, height) # bottom boundary (1 is added as ending is exclusive in array slicing)
            x_min = max(col - half_block, 0) # left boundary
            x_max = min(col + half_block + 1, width) # right boundary

            if len(image.shape) == 2: # Grayscale image
                # Extract the local block
                local_block = image[y_min:y_max, x_min:x_max]

                # Compute the mean of the local block
                local_mean = np.mean(local_block)

                if image[row][col] > local_mean:
                    thresholded_image[row][col] = 255
                else:
                    thresholded_image[row][col] = 0
            
            elif len(image.shape) == 3:  # Colored image (BGR)
                # Extract the local block for each channel
                local_block_b = image[y_min:y_max, x_min:x_max, 0] # Blue channel
                local_block_g = image[y_min:y_max, x_min:x_max, 1] # Green channel
                local_block_r = image[y_min:y_max, x_min:x_max, 2]  # Red channel

                # Compute the mean for each color channel
                local_mean_b = np.mean(local_block_b)
                local_mean_g = np.mean(local_block_g)
                local_mean_r = np.mean(local_block_r)
                local_means = [local_mean_b, local_mean_g, local_mean_r]

                for channel in range(3):
                    if image[row][col][channel] > local_means[channel]:
                        thresholded_image[row][col][channel] = 255
                    else:
                        thresholded_image[row][col][channel] = 0

    return thresholded_image
