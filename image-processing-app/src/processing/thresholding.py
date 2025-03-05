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

def local_threshold(image, block_size, constant):
    return