import cv2
import numpy as np
def Hessian_matrix(img):
    '''
    Compute the Hessian matrix components for the image.
    returns S_xx, S_yy, S_xy
    S_xx: Second derivative in x direction
    S_yy: Second derivative in y direction
    S_xy: Mixed derivative
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  #  apply Gaussian blur to reduce noise
    I_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)  # Gradient in x direction
    I_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)  # Gradient in y direction
    I_xx = I_x * I_x  
    I_yy = I_y * I_y  
    I_xy = I_x * I_y  
    # Gaussian smoothing
    S_xx = cv2.GaussianBlur(I_xx, (5, 5), 0)   
    S_yy = cv2.GaussianBlur(I_yy, (5, 5), 0)  
    S_xy = cv2.GaussianBlur(I_xy, (5, 5), 0) 
    return S_xx, S_yy, S_xy


def Harris(img, k=0.04, th=0.01):
    
    S_xx, S_yy, S_xy = Hessian_matrix(img)  # Compute Hessian matrix components
 
   
    detM = (S_xx * S_yy) - (S_xy ** 2)
    traceM = S_xx + S_yy
    R = detM - k * (traceM ** 2)

    # Thresholding to detect corners
    threshold = th * R.max()
    corner_image = img.copy()
    corner_image[R > threshold] = [0, 0, 255]  # Red mark for corners

    return corner_image

    
def lambda_minus(img,k,th):
    print(k)
    print(th)
    
    

    S_xx, S_yy, S_xy = Hessian_matrix(img)  # Compute Hessian matrix components
    
    # Compute eigenvalues
    lambda1 = (S_xx + S_yy) / 2 + np.sqrt(((S_xx - S_yy) / 2) ** 2 + S_xy ** 2)
    lambda2 = (S_xx + S_yy) / 2 - np.sqrt(((S_xx - S_yy) / 2) ** 2 + S_xy ** 2)

    # Minimum eigenvalue
    lambda_min = np.minimum(lambda1, lambda2)
    # Thresholding to detect corners
    threshold = th * lambda_min.max()
    corner_map = (lambda_min > threshold)

    # Visualize detected corners
    corner_image = img.copy()
    corner_image[corner_map] = [0, 0, 255]  # Red mark for corners
    return corner_image
