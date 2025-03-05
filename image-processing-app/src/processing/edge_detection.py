from PyQt6.QtGui import QImage, QPixmap
import cv2
import numpy as np

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
    def gaussian_blur(image, ksize=(5, 5), sigma=1.4):
        """Apply Gaussian blur"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # For now, we'll use OpenCV for Gaussian blur
        # In a full implementation, we would create the kernel ourselves
        return cv2.GaussianBlur(image, ksize, sigma)
    
    @staticmethod
    def Canny(image: np.ndarray, low_threshold: int=50, high_threshold: int=150, ksize: tuple=(5, 5), sigma: float=1.4):
        """
        Apply simplified Canny edge detection
        
        Args:
            image: Input image
            low_threshold: Low threshold for hysteresis
            high_threshold: High threshold for hysteresis
            ksize: Gaussian kernel size for noise reduction
            sigma: Gaussian kernel standard deviation
            
        Returns:
            Edge detected image
        """
        # For demonstration purposes, we'll use OpenCV's Canny implementation
        # A full implementation would include:
        # 1. Gaussian blur
        # 2. Gradient calculation using Sobel
        # 3. Non-maximum suppression
        # 4. Double threshold
        # 5. Edge tracking by hysteresis
        
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        # Step 1: Noise reduction (blur)
        blurred = EdgeDetection.gaussian_blur(image, ksize, sigma)
        
        # Use OpenCV's Canny for steps 2-5
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        
        return edges
    
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
    Apply Canny edge detection
    
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