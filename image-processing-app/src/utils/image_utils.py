import cv2
import numpy as np
import os

def load_image(file_path):
    """
    Load an image from the specified file path.
    
    Args:
        file_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: Loaded image as a NumPy array, or None if loading fails
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
        
    try:
        # Read the image using OpenCV
        image = cv2.imread(file_path)
        
        # Check if image was successfully loaded
        if image is None:
            print(f"Failed to load image: {file_path}")
            return None
            
        return image
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        return None

def save_image(filepath, image):
    cv2.imwrite(filepath, image)


class ImageProcessing:
    """Custom image processing functions implemented from scratch"""
    
    @staticmethod
    def convert_to_grayscale(image):
        if image is None:
        return None
        
        # If image is already grayscale, return it as it is
        if len(image.shape) == 2:
            return image

        grayScaledImage = np.zeros((image.shape[0], image.shape[1]), dtype=image.dtype) # Creates a zero array with the same height and width as image (but no channels)

        for row in range(image.shape[0]):
                for col in range(image.shape[1]):
                        grayIntensity = round(0.114 * image[row][col][0] + 0.587 * image[row][col][1] + 0.299 * image[row][col][2])
                        grayScaledImage[row][col] = grayIntensity

        return grayScaledImage

    @staticmethod
    def normalize_image(image, target_min=0, target_max=255):
        """
        Normalize image intensity values to a specified range
        
        Args:
            image (np.ndarray): Input image
            target_min: Minimum value in output image
            target_max: Maximum value in output image
            
        Returns:
            np.ndarray: Normalized image
        """
        if image is None:
            return None
        
        # Create a copy to avoid modifying the original
        normalized = image.copy().astype(np.float32)
        
        # Get current min and max values
        current_min = np.min(normalized)
        current_max = np.max(normalized)
        
        # Check if normalization is needed
        if current_min == current_max:
            return np.full_like(image, target_min, dtype=np.uint8)
        
        # Apply min-max normalization
        normalized = (normalized - current_min) * ((target_max - target_min) / (current_max - current_min)) + target_min
        
        # Clip values to ensure they are in range and convert to uint8
        return np.clip(normalized, target_min, target_max).astype(np.uint8)

    @staticmethod
    def frequency_of_grey_levels(image):
        """
        Calculate the histogram (frequency of occurrence of each gray level)
        
        Args:
            image: Input grayscale image
            
        Returns:
            np.ndarray: Array containing frequency count for each gray level (0-255)
        """
        # Convert to grayscale if color image
        if len(image.shape) == 3:
            image = ImageProcessing.convert_to_grayscale(image)
            
        # Initialize frequency array
        freq = np.zeros(shape=(256,), dtype=np.int32)
        
        # Count occurrences of each gray level
        for gray_level in range(256):
            freq[gray_level] = np.sum(image == gray_level)
            
        return freq

    @staticmethod
    def equalize_histogram(image):
        """
        Equalize the histogram of an image to improve contrast
        
        Args:
            image: Input image
            
        Returns:
            np.ndarray: Image with equalized histogram
        """
        if image is None:
            return None
        
        # Check if image is color or grayscale
        is_color = len(image.shape) == 3
        
        if is_color:
            # Process color image by equalizing in YUV/YCrCb space
            # Convert to YCrCb
            ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            
            # Extract Y channel
            y_channel = ycrcb_image[:,:,0]
            
            # Equalize Y channel
            y_equalized = ImageProcessing._equalize_single_channel(y_channel)
            
            # Replace Y channel
            ycrcb_image[:,:,0] = y_equalized
            
            # Convert back to BGR
            return cv2.cvtColor(ycrcb_image, cv2.COLOR_YCrCb2BGR)
        else:
            # Process grayscale image directly
            return ImageProcessing._equalize_single_channel(image)

    @staticmethod
    def _equalize_single_channel(channel):
        """
        Helper function to equalize a single channel using histogram equalization
        
        Args:
            channel: Single channel image
            
        Returns:
            np.ndarray: Equalized channel
        """
        # Get histogram frequencies
        freq = ImageProcessing.frequency_of_grey_levels(channel)
        
        # Calculate probability mass function (PMF)
        pmf = freq / channel.size
        
        # Calculate cumulative distribution function (CDF)
        cdf = np.cumsum(pmf) * 255.0
        
        # Create lookup table for mapping
        lookup_table = np.clip(np.round(cdf), 0, 255).astype(np.uint8)
        
        # Apply mapping to get equalized image
        equalized_channel = lookup_table[channel]
        
        return equalized_channel


# Replace OpenCV functions with our custom implementations
def convert_to_grayscale(image):
    return ImageProcessing.convert_to_grayscale(image)

def normalize_image(image):
    return ImageProcessing.normalize_image(image)

def equalize_histogram(image):
    return ImageProcessing.equalize_histogram(image)