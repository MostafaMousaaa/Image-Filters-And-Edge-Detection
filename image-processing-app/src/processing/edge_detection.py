from PyQt6.QtGui import QImage, QPixmap
import cv2
import numpy as np
import math

class EdgeDetection:
    """Custom edge detection class with implementations from scratch"""
    
    @staticmethod
    def convolve(image: np.ndarray, gx: np.ndarray, gy: np.ndarray):
        """
        Apply convolution with x and y kernels
        
        Args:
            image: Input image
            gx: Kernel for x direction
            gy: Kernel for y direction
        
        Returns:
            Tuple of gradient images in x and y directions
        """
        if(len(image.shape) == 3):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Extract shape - fixed to handle both grayscale and color images
        if len(image.shape) == 3:
            rows, cols, _ = image.shape
        else:
            rows, cols = image.shape
            
        kernel_rows, kernel_cols = gx.shape
        output_rows, output_cols = rows, cols
        if kernel_rows % 2 == 0:
            output_rows -= 1
            output_cols -= 1
        else:
            output_rows -= 2
            output_cols -= 2
        i_x = np.zeros((output_rows, output_cols), dtype=np.float32)
        i_y = np.zeros((output_rows, output_cols), dtype=np.float32)

        for i in range(0, rows-kernel_rows+1):
            for j in range(0, cols-kernel_cols+1):
                square = image[i:i+kernel_rows, j:j+kernel_cols]
                i_x[i, j] = np.sum(square * gx)
                i_y[i, j] = np.sum(square * gy)
        
        return i_x, i_y
    
    @staticmethod
    def Sobel(image: np.ndarray, direction: str='mag', kSize: int=3):
        """
        Apply Sobel edge detection
        
        Args:
            image: Input image
            direction: 'x' for x-direction, 'y' for y-direction, 'mag' for magnitude
            kSize: Kernel size (3, 5, or 7)
            
        Returns:
            Edge detected image
        """
        g_x, g_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32), np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        
        if kSize == 5:
            g_x, g_y = (
                np.array([
                    [-2, -1, 0, 1, 2],
                    [-3, -2, 0, 2, 3],
                    [-4, -3, 0, 3, 4],
                    [-3, -2, 0, 2, 3],
                    [-2, -1, 0, 1, 2]], dtype=np.float32),
                np.array([
                    [-2, -3, -4, -3, -2],
                    [-1, -2, -3, -2, -1],
                    [0, 0, 0, 0, 0],
                    [1, 2, 3, 2, 1],
                    [2, 3, 4, 3, 2]], dtype=np.float32)
            )
        elif kSize == 7:
            g_x, g_y = (
                np.array([
                    [-3, -2, -1, 0, 1, 2, 3],
                    [-4, -3, -2, 0, 2, 3, 4],
                    [-5, -4, -3, 0, 3, 4, 5],
                    [-6, -5, -4, 0, 4, 5, 6],
                    [-5, -4, -3, 0, 3, 4, 5],
                    [-4, -3, -2, 0, 2, 3, 4],
                    [-3, -2, -1, 0, 1, 2, 3]], dtype=np.float32),
                np.array([
                    [-3, -4, -5, -6, -5, -4, -3],
                    [-2, -3, -4, -5, -4, -3, -2],
                    [-1, -2, -3, -4, -3, -2, -1],
                    [0, 0, 0, 0, 0, 0, 0],
                    [1, 2, 3, 4, 3, 2, 1],
                    [2, 3, 4, 5, 4, 3, 2],
                    [3, 4, 5, 6, 5, 4, 3]], dtype=np.float32)
            )
                
        i_x, i_y = EdgeDetection.convolve(image=image, gx=g_x, gy=g_y)
        
        if direction.lower() == 'x':
            return EdgeDetection.normalize_to_uint8(i_x)
        elif direction.lower() == 'y':
            return EdgeDetection.normalize_to_uint8(i_y)
        else:
            magnitude = np.sqrt(np.square(i_x) + np.square(i_y))
            return EdgeDetection.normalize_to_uint8(magnitude)
    
    @staticmethod
    def prewitt(image: np.ndarray, direction: str='mag'):
        """
        Apply Prewitt edge detection
        
        Args:
            image: Input image
            direction: 'x' for x-direction, 'y' for y-direction, 'mag' for magnitude
            
        Returns:
            Edge detected image
        """
        g_x, g_y = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32), np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

        i_x, i_y = EdgeDetection.convolve(image, g_x, g_y)
        
        if direction.lower() == 'x':
            return EdgeDetection.normalize_to_uint8(i_x)
        elif direction.lower() == 'y':
            return EdgeDetection.normalize_to_uint8(i_y)
        else:
            magnitude = np.sqrt(np.square(i_x) + np.square(i_y))
            return EdgeDetection.normalize_to_uint8(magnitude)
     
    @staticmethod
    def roberts(image: np.ndarray, direction: str='mag'):
        """
        Apply Roberts edge detection
        
        Args:
            image: Input image
            direction: 'x' for x-direction, 'y' for y-direction, 'mag' for magnitude
            
        Returns:
            Edge detected image
        """
        g_x, g_y = np.array([[1, 0], [0, -1]], dtype=np.float32), np.array([[0, 1], [-1, 0]], dtype=np.float32)
        i_x, i_y = EdgeDetection.convolve(image, g_x, g_y)
        
        if direction.lower() == 'x':
            return EdgeDetection.normalize_to_uint8(i_x)
        elif direction.lower() == 'y':
            return EdgeDetection.normalize_to_uint8(i_y)
        else:
            magnitude = np.sqrt(np.square(i_x) + np.square(i_y))
            return EdgeDetection.normalize_to_uint8(magnitude)
    

    @staticmethod
    def Canny(image: np.ndarray, low_threshold: int=50, high_threshold: int=150, ksize: int=5, sigma: float=1.4):
        """
        Apply Canny edge detection from scratch
        
        Args:
            image: Input image
            low_threshold: Low threshold for hysteresis
            high_threshold: High threshold for hysteresis
            ksize: Gaussian kernel size for noise reduction
            sigma: Gaussian kernel standard deviation
            
        Returns:
            Edge detected image
        """
        # Step 1: Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Step 2: Noise reduction using Gaussian filter
        blurred = EdgeDetection.apply_gaussian_filter(gray, ksize, sigma)
        
        # Step 3: Gradient calculation using Sobel
        # Define Sobel kernels
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        
        # Apply Sobel kernels
        gradient_x, gradient_y = EdgeDetection.convolve(blurred, sobel_x, sobel_y)
        
        # Calculate gradient magnitude and direction
        gradient_magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
        gradient_direction = np.arctan2(gradient_y, gradient_x)
        
        # Normalize gradient magnitude to 0-255
        gradient_magnitude = 255.0 * gradient_magnitude / np.max(gradient_magnitude)
        
        # Step 4: Non-maximum suppression
        suppressed = EdgeDetection.non_max_suppression(gradient_magnitude, gradient_direction)
        
        # Step 5: Double thresholding
        strong_edges, weak_edges = EdgeDetection.double_threshold(suppressed, low_threshold, high_threshold)
        
        # Step 6: Edge tracking by hysteresis
        final_edges = EdgeDetection.edge_tracking(strong_edges, weak_edges)
        
        return final_edges.astype(np.uint8)
    
    @staticmethod
    def normalize_to_uint8(img):
        """
        Normalize image values to uint8 range [0, 255]
        
        Args:
            img: Input floating point image
            
        Returns:
            8-bit unsigned integer image
        """
        img_min = np.min(img)
        img_max = np.max(img)
        
        if img_max - img_min > 0:
            normalized = 255.0 * (img - img_min) / (img_max - img_min)
        else:
            normalized = np.zeros_like(img)
            
        return normalized.astype(np.uint8)
    
    @staticmethod
    def gaussian_kernel(size=5, sigma=1.4):
        """
        Create a Gaussian kernel
        
        Args:
            size: Size of the kernel (must be odd)
            sigma: Standard deviation of the Gaussian
            
        Returns:
            Gaussian kernel
        """
        # Ensure size is odd
        if size % 2 == 0:
            size += 1
            
        # Create a coordinate grid
        half_size = size // 2
        x, y = np.meshgrid(np.arange(-half_size, half_size + 1), np.arange(-half_size, half_size + 1))
        
        # Calculate the Gaussian function
        normal = 1 / (2.0 * np.pi * sigma**2)
        kernel = normal * np.exp(-(x**2 + y**2) / (2 * sigma**2))
        
        # Normalize the kernel
        return kernel / np.sum(kernel)
    
    @staticmethod
    def apply_gaussian_filter(image, kernel_size=5, sigma=1.4):
        """
        Apply Gaussian filter using custom convolution
        
        Args:
            image: Input image
            kernel_size: Size of the Gaussian kernel
            sigma: Standard deviation of the Gaussian
            
        Returns:
            Blurred image
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create Gaussian kernel
        kernel = EdgeDetection.gaussian_kernel(kernel_size, sigma)
        
        # Get image dimensions - fixed to handle both grayscale and color images
        if len(image.shape) == 3:
            rows, cols, _ = image.shape
        else:
            rows, cols = image.shape
            
        kernel_rows, kernel_cols = kernel.shape
        
        # Calculate output dimensions
        output_rows = rows - kernel_rows + 1
        output_cols = cols - kernel_cols + 1
        
        # Create output array
        output = np.zeros((output_rows, output_cols), dtype=np.float32)
        
        # Apply convolution
        for i in range(output_rows):
            for j in range(output_cols):
                output[i, j] = np.sum(image[i:i+kernel_rows, j:j+kernel_cols] * kernel)
                
        return output
    
    @staticmethod
    def non_max_suppression(gradient_magnitude, gradient_direction):
        """
        Apply non-maximum suppression to the gradient magnitude
        
        Args:
            gradient_magnitude: Magnitude of the gradient
            gradient_direction: Direction of the gradient in radians
            
        Returns:
            Result of non-maximum suppression
        """
        # Extract shape - fixed to handle both grayscale and color images
        if len(gradient_magnitude.shape) == 3:
            rows, cols, _ = gradient_magnitude.shape
        else:
            rows, cols = gradient_magnitude.shape
            
        output = np.zeros((rows, cols), dtype=np.float32)
        
        # Convert angle to degrees and take absolute value
        angle = np.degrees(gradient_direction) % 180
        
        # For each pixel, check if it's a local maximum along the gradient direction
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                # Check pixel's neighbors based on gradient direction
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    # East-West direction (horizontal)
                    neighbors = [gradient_magnitude[i, j+1], gradient_magnitude[i, j-1]]
                elif 22.5 <= angle[i, j] < 67.5:
                    # Northeast-Southwest direction
                    neighbors = [gradient_magnitude[i-1, j+1], gradient_magnitude[i+1, j-1]]
                elif 67.5 <= angle[i, j] < 112.5:
                    # North-South direction (vertical)
                    neighbors = [gradient_magnitude[i-1, j], gradient_magnitude[i+1, j]]
                elif 112.5 <= angle[i, j] < 157.5:
                    # Northwest-Southeast direction
                    neighbors = [gradient_magnitude[i-1, j-1], gradient_magnitude[i+1, j+1]]
                
                # If the current pixel is a local maximum, keep it
                if gradient_magnitude[i, j] >= max(neighbors):
                    output[i, j] = gradient_magnitude[i, j]
                
        return output
    
    @staticmethod
    def double_threshold(image, low_threshold, high_threshold):
        """
        Apply double thresholding to the image
        
        Args:
            image: Input image
            low_threshold: Low threshold value
            high_threshold: High threshold value
            
        Returns:
            Image with strong and weak edges
        """
        # Create output array for strong and weak edges
        strong_edges = np.zeros_like(image)
        weak_edges = np.zeros_like(image)
        
        # Apply thresholding
        strong_i, strong_j = np.where(image >= high_threshold)
        weak_i, weak_j = np.where((image < high_threshold) & (image >= low_threshold))
        
        # Set pixel values
        strong_edges[strong_i, strong_j] = 255
        weak_edges[weak_i, weak_j] = 50  # Weak edges are set to a mid-gray value
        
        return strong_edges, weak_edges
    
    @staticmethod
    def edge_tracking(strong_edges, weak_edges):
        """
        Track edges by hysteresis - connecting weak edges to strong edges
        
        Args:
            strong_edges: Image with strong edges (255)
            weak_edges: Image with weak edges (50)
            
        Returns:
            Final edge image
        """
        # Extract shape - fixed to handle both grayscale and color images
        if len(strong_edges.shape) == 3:
            rows, cols, _ = strong_edges.shape
        else:
            rows, cols = strong_edges.shape
            
        # Create a combined image with both strong and weak edges
        edges = strong_edges.copy()
        weak_i, weak_j = np.where(weak_edges == 50)
        
        # Define 8-connected neighborhood offsets
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        # For each weak edge pixel
        for i, j in zip(weak_i, weak_j):
            # Check if any of the 8 neighboring pixels is a strong edge
            for di, dj in neighbors:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols and edges[ni, nj] == 255:
                    # If connected to a strong edge, make this weak edge strong
                    edges[i, j] = 255
                    break
        
        # Set any remaining weak edges to 0
        edges[edges != 255] = 0
        
        return edges


# Interface functions that use our custom EdgeDetection class
def sobel_edge_detection(image, direction=None):
    """
    Apply Sobel edge detection using custom implementation
    
    Args:
        image: Input image
        direction: None for magnitude, 'x' for x-direction, 'y' for y-direction
    
    Returns:
        Edge detected image
    """
    dir_param = 'mag'
    if direction == 'x':
        dir_param = 'x'
    elif direction == 'y':
        dir_param = 'y'
    
    return EdgeDetection.Sobel(image, direction=dir_param)

def roberts_edge_detection(image, direction=None):
    """
    Apply Roberts edge detection using custom implementation
    
    Args:
        image: Input image
        direction: None for magnitude, 'x' for x-direction, 'y' for y-direction
    
    Returns:
        Edge detected image
    """
    dir_param = 'mag'
    if direction == 'x':
        dir_param = 'x'
    elif direction == 'y':
        dir_param = 'y'
    
    return EdgeDetection.roberts(image, direction=dir_param)

def prewitt_edge_detection(image, direction=None):
    """
    Apply Prewitt edge detection using custom implementation
    
    Args:
        image: Input image
        direction: None for magnitude, 'x' for x-direction, 'y' for y-direction
    
    Returns:
        Edge detected image
    """
    dir_param = 'mag'
    if direction == 'x':
        dir_param = 'x'
    elif direction == 'y':
        dir_param = 'y'
    
    return EdgeDetection.prewitt(image, direction=dir_param)

def canny_edge_detection(image, low_threshold=50, high_threshold=150):
    """
    Apply Canny edge detection using custom implementation
    
    Args:
        image: Input image
        low_threshold: Low threshold for the hysteresis procedure
        high_threshold: High threshold for the hysteresis procedure
    
    Returns:
        Edge detected image
    """
    return EdgeDetection.Canny(image, low_threshold=low_threshold, high_threshold=high_threshold)

def display_image(image: np.ndarray) -> QPixmap:
    """Convert OpenCV image to QPixmap for display"""
    height, width, channel = image.shape
    bytes_per_line = 3 * width
    q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(q_image)