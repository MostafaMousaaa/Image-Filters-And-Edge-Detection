from PyQt6.QtGui import QImage, QPixmap
import cv2
import numpy as np

def sobel_edge_detection(image, direction=None):
    """
    Apply Sobel edge detection
    
    Args:
        image: Input image
        direction: None for magnitude, 'x' for x-direction, 'y' for y-direction
    
    Returns:
        Edge detected image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply Sobel operator
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Based on direction parameter
    if direction == 'x':
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        return abs_grad_x
    elif direction == 'y':
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        return abs_grad_y
    else:
        # Default: compute the magnitude
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        return grad

def roberts_edge_detection(image, direction=None):
    """
    Apply Roberts edge detection
    
    Args:
        image: Input image
        direction: None for magnitude, 'x' for x-direction, 'y' for y-direction
    
    Returns:
        Edge detected image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Define Roberts operators
    roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    
    # Apply operators using filter2D
    grad_x = cv2.filter2D(gray, -1, roberts_x)
    grad_y = cv2.filter2D(gray, -1, roberts_y)
    
    # Based on direction parameter
    if direction == 'x':
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        return abs_grad_x
    elif direction == 'y':
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        return abs_grad_y
    else:
        # Compute magnitude
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        return grad

def prewitt_edge_detection(image, direction=None):
    """
    Apply Prewitt edge detection
    
    Args:
        image: Input image
        direction: None for magnitude, 'x' for x-direction, 'y' for y-direction
    
    Returns:
        Edge detected image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Define Prewitt operators
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    
    # Apply operators using filter2D
    grad_x = cv2.filter2D(gray, -1, prewitt_x)
    grad_y = cv2.filter2D(gray, -1, prewitt_y)
    
    # Based on direction parameter
    if direction == 'x':
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        return abs_grad_x
    elif direction == 'y':
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        return abs_grad_y
    else:
        # Compute magnitude
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        return grad

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
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    
    return edges

def display_image(image: np.ndarray) -> QPixmap:
    height, width, channel = image.shape
    bytes_per_line = 3 * width
    q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(q_image)