from PyQt6.QtGui import QImage, QPixmap
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from ..processing.filters import gaussian_filter_custom

def sobel_edge_detection(image):
    image = image.astype(np.float32)
    gx_sobel = np.array([[-1, 0, 1],
                [-2, 0, 2], 
                [-1, 0, 1]], dtype=np.float32)  # horizontal gradient kernel
    gy_sobel = np.array([[-1, -2, -1], 
                [0, 0, 0], 
                [1, 2, 1]], dtype=np.float32) # vertical gradient kernel
    
    gx_img = ndimage.convolve(np.copy(image), gx_sobel)
    gy_img = ndimage.convolve(np.copy(image), gy_sobel)
    
    
    g_mag_img = np.hypot(gx_img, gy_img)
    g_mag_img /= np.max(g_mag_img)   # normalization for pixel values
    g_mag_img = (g_mag_img * 255).astype(np.uint8)  # Scale to 0-255 as QImage() expects the image to be 8 bit integer values to process the pixels correctly
                                                    # while here, g_mag_img is of float values from 0 to 1, so we made this scaling and conversion
    g_directions = np.arctan2(gy_img, gx_img) #direction mat used in non maximal suppression (edge thinning)
    # plt.imshow( g_mag_img, cmap= plt.get_cmap('gray'))
    # plt.show()
    return g_mag_img, g_directions

def non_maximal_suppression(g_mag_image, dir_mat):
    for i in range(1, g_mag_image.shape[0] - 1):
        for j in range(1, g_mag_image.shape[1] - 1):
            # direction 0
            if (dir_mat[i][j]/np.pi) * 180 >= -22.5 and (dir_mat[i][j]/np.pi) * 180 < 22.5 or np.abs((dir_mat[i][j]/np.pi) * 180) >= 157.5:
                if g_mag_image [i][j] < g_mag_image [i][j+1] or g_mag_image[i][j] < g_mag_image[i][j-1]:
                    g_mag_image[i][j] = 0
            # direction 45 
            elif (dir_mat[i][j]/np.pi) * 180 >= 22.5 and (dir_mat[i][j]/np.pi) * 180 < 67.5 or (dir_mat[i][j]/np.pi) * 180 >= -157.5 and (dir_mat[i][j]/np.pi) * 180 < -112.5:
                if g_mag_image [i][j] < g_mag_image [i+1][j-1] or g_mag_image[i][j] < g_mag_image[i-1][j+1]:
                    g_mag_image[i][j] = 0
            # direction 90
            elif (dir_mat[i][j]/np.pi) * 180 >= 67.5 and (dir_mat[i][j]/np.pi) * 180 < 112.5 or (dir_mat[i][j]/np.pi) * 180 >= -112.5 and (dir_mat[i][j]/np.pi) * 180 < -67.5:
                if g_mag_image [i][j] < g_mag_image [i-1][j] or g_mag_image[i][j] < g_mag_image[i+1][j]:
                    g_mag_image[i][j] = 0
            # direction 135
            elif (dir_mat[i][j]/np.pi) * 180 >= 112.5 and (dir_mat[i][j]/np.pi) * 180 < 157.5 or (dir_mat[i][j]/np.pi) * 180 >= -67.5 and (dir_mat[i][j]/np.pi) * 180 < -22.5:
                if g_mag_image [i][j] < g_mag_image [i+1][j+1] or g_mag_image[i][j] < g_mag_image[i-1][j-1]:
                    g_mag_image[i][j] = 0
            
    nms_img = g_mag_image
    return nms_img
    

def hysterisis_thersholding(nms_img, low_ratio = 0.15, high_ratio = 0.4):
    high_thres = np.max(nms_img) * high_ratio
    low_thres = high_thres * low_ratio
    
    for row in range(1, nms_img.shape[0]-1):
        for col in range(1, nms_img.shape[1]-1):
            if nms_img[row][col] < low_thres:
                nms_img[row][col] = 0
            elif nms_img[row][col] >= high_thres:
                nms_img[row][col] = 255
            else:
                if nms_img[row-1][col-1] >= high_thres or nms_img[row-1][col] >= high_thres or nms_img[row-1][col+1] >= high_thres or nms_img[row][col-1] >= high_thres or nms_img[row][col+1] >= high_thres or nms_img[row+1][col-1] >= high_thres or nms_img[row+1][col] >= high_thres or nms_img[row+1][col+1] >= high_thres:
                    nms_img[row][col] = 255
                else:
                    nms_img[row][col] = 0
                    
    canny_img = nms_img
    return canny_img           
                     

def canny_edge_detection(image):
    smoothed_img = gaussian_filter_custom(image, size = 3, sigma = 1)  # smoothing
    g_mag_img, g_directions = sobel_edge_detection(smoothed_img)  # edge detection using gradients
    nms_img = non_maximal_suppression(g_mag_img, g_directions)    # edge thinning
    canny_img = hysterisis_thersholding(nms_img)   # double thresholding and hysterisis thresholding
    return canny_img




################################################################################################    
    
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



def display_image(image: np.ndarray) -> QPixmap:
    height, width, channel = image.shape
    bytes_per_line = 3 * width
    q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(q_image)