import numpy as np
import cv2

def add_gaussian_noise(image, mean=0, sigma=25):
    noisy_image = np.copy(image)
    noise = np.random.normal(mean, sigma, noisy_image.shape).astype(np.int16)  # generate random values around the mean value with spreading controlled by sigma for each pixel.
                                                                             # values are mostly concentrated around the mean value like the normal distribution curve (bell shaped) with some values far from it (histogram is bell shaped)
                                                                             # max and min values that can occur are (mean + 3*sigma , mean - 3*sigma)
    noisy_image = cv2.add(noisy_image.astype(np.int16), noise)  # Convert image pixels also to be of 16 bits each to be like the noise and to avoid overflow while adding
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8) # returning image pixels to 8 bits each again and clipping values outside 0 - 255 range
    
    return noisy_image
    
def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    
    noisy_image = np.copy(image)
    total_pixels = noisy_image.shape[0] * noisy_image.shape[1]
        # adding salt
    num_salt = int(total_pixels * salt_prob)  # number of salt pixels
    salt_coords = [np.random.randint(0, noisy_image.shape[0], num_salt), np.random.randint(0, noisy_image.shape[1], num_salt)] # list of 2 np arrays, first one for row coordinates for the salt and second one is for corresponding column coordinates
    noisy_image[salt_coords[0], salt_coords[1]] = 255  # changing values of pixes at salt coordinates into white
    # adding pepper the same way
    num_pepper = int(total_pixels * pepper_prob)
    pepper_coords = [np.random.randint(0, noisy_image.shape[0], num_pepper), np.random.randint(0, noisy_image.shape[1], num_pepper)] # list of 2 np arrays, first one for row coordinates for the pepper and second one is for corresponding column coordinates
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0
    
    return noisy_image


# uniform noise doesn't simulate real world scenarios, but is usually used for testing
def add_uniform_noise(image, low=0, high=50):
    noisy_image = np.copy(image)
    uniform_noise = np.random.uniform(low, high, noisy_image.shape).astype(np.uint8)  # generates random values between 'low' and 'high' for each pixel
    noisy_image = cv2.add(image, uniform_noise) # adds noise to the image and ensures values remain within 0-255, if values in a pixel exceeded 255 while adding, it's clipped to 255(white)
    return noisy_image