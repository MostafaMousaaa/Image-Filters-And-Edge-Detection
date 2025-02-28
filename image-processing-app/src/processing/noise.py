import numpy as np
import cv2

def add_gaussian_noise(image, mean=0, sigma=25):
    """Add Gaussian noise to an image."""
    gaussian_noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, gaussian_noise)
    return noisy_image

def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    """Add salt and pepper noise to an image."""
    noisy_image = np.copy(image)
    total_pixels = image.size
    num_salt = np.ceil(salt_prob * total_pixels)
    num_pepper = np.ceil(pepper_prob * total_pixels)

    # Add salt noise
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[coords[0], coords[1], :] = 255

    # Add pepper noise
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords[0], coords[1], :] = 0

    return noisy_image

def add_uniform_noise(image, low=0, high=50):
    """Add uniform noise to an image."""
    uniform_noise = np.random.uniform(low, high, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, uniform_noise)
    return noisy_image