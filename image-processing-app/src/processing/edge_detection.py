from PyQt6.QtGui import QImage, QPixmap
import cv2
import numpy as np
from scipy import ndimage
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
    def Sobel(image):
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
    def Canny(image: np.ndarray, low_threshold: int=30, high_threshold: int=100, ksize: int=5, sigma: float=1.4):
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Step 1: Noise reduction using Gaussian filter
        blurred_img = EdgeDetection.apply_gaussian_filter(gray, ksize, sigma)
        
        # Step 2: Gradient calculation using Sobel
        g_mag_img, g_dir_mat = EdgeDetection.sobel_edge_detection(blurred_img)
        
        # Step 3: Non-maximum suppression
        nms_img = EdgeDetection.non_max_suppression(g_mag_img, g_dir_mat)
        
        # Step 4 & 5: Double thresholding and hysterisis thresholding (edge tracking)
        canny_img = EdgeDetection.double_threshold(nms_img, low_threshold, high_threshold)
        
        return canny_img
    
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
    def apply_gaussian_filter(image, size=5, sigma=1.4):
        print("entered gaussian smoothing")
        noisy_img = np.copy(image)            
        
        half_kernel_size = size // 2  # Half-size for symmetry
        x, y = np.meshgrid(np.arange(-half_kernel_size, half_kernel_size+1), np.arange(-half_kernel_size, half_kernel_size+1))
        
        # apply the Gaussian formula
        gaussian_kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        
        # normalize so the sum is 1 by dividing by the sum of pixel values
        gaussian_kernel /= np.sum(gaussian_kernel)
        # print(f"gaussian kernel: {gaussian_kernel}")
        
        if len(image.shape) == 3 and image.shape[2] == 3 :  #rgb image
            for curr_channel in range(noisy_img.shape[2]):  # loop over the 3 channels as it's a colored img
                # looping over the image pixels except the boundaries to update pixel values using the gaussian kernel
                for row in range(half_kernel_size, noisy_img.shape[0] - half_kernel_size):   
                    for col in range (half_kernel_size, noisy_img.shape[1] - half_kernel_size):
                        used_range = noisy_img[ row-half_kernel_size : row+half_kernel_size+1 , col-half_kernel_size : col+half_kernel_size+1, curr_channel] # slicing to take the pixel and its neighbouring pixels in each loop.
                        filtered_pixel = np.sum(gaussian_kernel * used_range)  # getting the new pixel value
                        
                        noisy_img[row][col][curr_channel] = np.clip(0, 255, filtered_pixel)  # updating the current pixel in the noisy image
        else:
            for row in range(half_kernel_size, noisy_img.shape[0] - half_kernel_size):   
                    for col in range (half_kernel_size, noisy_img.shape[1] - half_kernel_size):
                        used_range = noisy_img[ row-half_kernel_size : row+half_kernel_size+1 , col-half_kernel_size : col+half_kernel_size+1] 
                        filtered_pixel = np.sum(gaussian_kernel * used_range)   
                        noisy_img[row][col] = np.clip(0, 255, filtered_pixel)
                        
        return noisy_img
    
    def sobel_edge_detection(image):
        print("entered sobel edge detection")
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
    
    @staticmethod
    def non_max_suppression(g_mag_image, dir_mat):
        print("entered nms")
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

    
    @staticmethod
    def double_threshold(nms_img, low_threshold, high_threshold):
        print("entered double thresholding")
        # high_thres = np.max(nms_img) * high_ratio
        # low_thres = high_thres * low_ratio
        
        for row in range(1, nms_img.shape[0]-1):
            for col in range(1, nms_img.shape[1]-1):
                if nms_img[row][col] < low_threshold:
                    nms_img[row][col] = 0
                elif nms_img[row][col] >= high_threshold:
                    nms_img[row][col] = 255
                else:
                    if nms_img[row-1][col-1] >= high_threshold or nms_img[row-1][col] >= high_threshold or nms_img[row-1][col+1] >= high_threshold or nms_img[row][col-1] >= high_threshold or nms_img[row][col+1] >= high_threshold or nms_img[row+1][col-1] >= high_threshold or nms_img[row+1][col] >= high_threshold or nms_img[row+1][col+1] >= high_threshold:
                        nms_img[row][col] = 255
                    else:
                        nms_img[row][col] = 0
                        
        canny_img = nms_img
        return canny_img
    
    # @staticmethod
    # def edge_tracking(strong_edges, weak_edges):
    #     """
    #     Track edges by hysteresis - connecting weak edges to strong edges
        
    #     Args:
    #         strong_edges: Image with strong edges (255)
    #         weak_edges: Image with weak edges (50)
            
    #     Returns:
    #         Final edge image
    #     """
    #     # Extract shape - fixed to handle both grayscale and color images
    #     if len(strong_edges.shape) == 3:
    #         rows, cols, _ = strong_edges.shape
    #     else:
    #         rows, cols = strong_edges.shape
            
    #     # Create a combined image with both strong and weak edges
    #     edges = strong_edges.copy()
    #     weak_i, weak_j = np.where(weak_edges == 50)
        
    #     # Define 8-connected neighborhood offsets
    #     neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
    #     # For each weak edge pixel
    #     for i, j in zip(weak_i, weak_j):
    #         # Check if any of the 8 neighboring pixels is a strong edge
    #         for di, dj in neighbors:
    #             ni, nj = i + di, j + dj
    #             if 0 <= ni < rows and 0 <= nj < cols and edges[ni, nj] == 255:
    #                 # If connected to a strong edge, make this weak edge strong
    #                 edges[i, j] = 255
    #                 break
        
    #     # Set any remaining weak edges to 0
    #     edges[edges != 255] = 0
        
    #     return edges


# Interface functions that use our custom EdgeDetection class
def sobel_edge_detection(image, direction=None):
    
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
    return EdgeDetection.Canny(image, low_threshold=low_threshold, high_threshold=high_threshold)

def display_image(image: np.ndarray) -> QPixmap:
    """Convert OpenCV image to QPixmap for display"""
    height, width, channel = image.shape
    bytes_per_line = 3 * width
    q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(q_image)