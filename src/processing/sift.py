import math
import cv2
import numpy as np
from .filters import gaussian_filter_custom

def generateSiftDescriptors(img, octaveLayersNum, sigma, keypointThreshold, edgeThreshold):
    if img is None:
        return []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Scale Space Construction
    k = math.sqrt(2)
    scaleLevels = [sigma, sigma * k, sigma * 2, sigma * 2 * k, sigma * 2 * k * k]
    octaveLevels = [img]

    for _ in range(octaveLayersNum - 1):
        prevImage = octaveLevels[-1]
        height = prevImage.shape[0]
        width = prevImage.shape[1]
        octaveLevels.append(cv2.resize(prevImage, (height // 2, width // 2))) # Saving different image octaves by downscaling the image by half
    
    diffOfGaussians = [] # List of lists, where each list contains multiple 2D matrices representing difference of gaussians whose sigma is different
    for octaveImage in octaveLevels:
        gaussians = []
        for sigmaLevel in scaleLevels:
            kernelSize = 2 * math.ceil(3 * sigmaLevel) + 1
            gaussians.append(cv2.GaussianBlur(src=octaveImage, ksize=(kernelSize, kernelSize), sigmaX=sigmaLevel, sigmaY=sigmaLevel))
            #gaussians.append(gaussian_filter_custom(image=octaveImage, size=kernelSize, sigma=sigmaLevel))

        currOctaveDOG = []
        for i in range(1, len(gaussians)):
            currOctaveDOG.append(cv2.subtract(gaussians[i], gaussians[i - 1]))
        diffOfGaussians.append(currOctaveDOG)
    
    # Scale Space Extrema Detection
    keypoints = []
    octaveIdx = 0
    for DOG in diffOfGaussians: # Loops through octaves
        for scaleIdx in range(1, len(DOG) - 1): # Loop through all difference of gaussians in the same octave
            lowerScale = DOG[scaleIdx - 1]
            currScale = DOG[scaleIdx]
            higherScale = DOG[scaleIdx + 1]
            
            height = currScale.shape[0]
            width = currScale.shape[1]

            sigmaOrientation = sigma * (math.sqrt(2) ** (octaveIdx + (scaleIdx + 0.5) / float(octaveLayersNum)))
            
            for y in range(1, height - 1):
                for x in range(1, width - 1):
                    pixelValue = currScale[y, x]

                    # Calculating 3 * 3 neighborhood for all 3 scales (26 neighbors at most for current pixel we are checking) (result is 1D array of 27 Elements)
                    neighborhood = np.concatenate([lowerScale[y - 1 : y + 2, x - 1 : x + 2].flatten(),
                                                   currScale[y - 1 : y + 2, x - 1 : x + 2].flatten(),
                                                   higherScale[y - 1 : y + 2, x - 1 : x + 2].flatten()])

                    if (pixelValue == np.max(neighborhood)) or (pixelValue == np.min(neighborhood)):
                        if abs(pixelValue) >= keypointThreshold: # First threshold which is contrast threshold (ignores low contrast keypoints that are suspect to noise)

                            # Edges are strong in one direction, while Corners (and blobs) are strong and stable in all directions, making them useful as SIFT keypoints
                            # Edge threshold is done by calculating hessian matrix (uses second derivatives which calculate curvature (change of gradient))

                            Dxx = currScale[y, x + 1] + currScale[y, x - 1] - 2 * pixelValue # second derivative in x direction(how much image changes horizontally)
                            Dyy = currScale[y + 1, x] + currScale[y - 1, x] - 2 * pixelValue # second derivative in y direction(how much image changes vertically)
                            # How much image changes diagonally
                            Dxy = ((currScale[y + 1, x + 1] - currScale[y + 1, x - 1]) - (currScale[y - 1, x + 1] - currScale[y - 1, x - 1])) / 4.0

                            trace = Dxx + Dyy # sum of diagonals of hessian matrix represents total curvature strength (summation of eigenvalues)
                            determinant = Dxx * Dyy - Dxy * Dxy # represents curvature strength in both directions (large positive means good corner) (product of eigenvalues)

                            if determinant > 0: # candidate for valid key point
                                curvatureRatio = (trace ** 2) / determinant # if ratio is small, then eigenvalues are both similar, therefore good corner
                                thresholdRatio = ((edgeThreshold + 1) ** 2) / edgeThreshold

                                if curvatureRatio <= thresholdRatio:
                                    keypoints.append((x, y, sigmaOrientation, octaveIdx))
        octaveIdx += 1
    
    return keypoints


def compute_gradients_opencv(image):
    dx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    
    grad_mag = np.sqrt(dx**2 + dy**2)
    grad_dir = np.degrees(np.arctan2(dy, dx) % (360))  # Convert to 0 - 360 range
    
    return grad_mag, grad_dir

def get_orientation_histogram(grad_mag, grad_dir, center, scale, bins=36):
    
    hist = np.zeros(bins)
    bin_width = 360 / bins   # each bin is of width 10
    # print(f"binwidth = {bin_width}")
    radius = int(6 * scale)  # Typically 6σ
    
    x, y = center
    x_min = max(0, int(x - radius))
    x_max = min(grad_mag.shape[1], int(x + radius + 1))
    y_min = max(0, int(y - radius))
    y_max = min(grad_mag.shape[0], int(y + radius + 1))
    
    for yi in range(y_min, y_max):
        for xi in range(x_min, x_max):
            dx = xi - x
            dy = yi - y
            dist_sq = dx**2 + dy**2
            
            if dist_sq > radius**2:
                continue
                
            # Gaussian weight based on distance (σ = 1.5*scale)
            weight = np.exp(-dist_sq / (2 * (1.5 * scale)**2))
            mag = grad_mag[yi, xi] * weight
            
            # Bin the orientation
            orientation = grad_dir[yi, xi] 
            bin_idx = int(orientation / bin_width) % bins
            hist[bin_idx] += mag
    
    return hist

def get_dominant_orientations(hist, peak_ratio=0.8):
    max_val = np.max(hist)
    bin_width = 360 / len(hist)
    orientations = []
    
    for i in range(len(hist)):  # looping on each bin to see if it's a dominant orientation
        # Check if this is a peak
        prev = hist[i-1] if i > 0 else hist[-1]
        next = hist[i+1] if i < len(hist)-1 else hist[0]
        
        if hist[i] > prev and hist[i] > next and hist[i] >= peak_ratio * max_val:
            orientations.append(i * bin_width)  
    
    return orientations

def get_rotated_coords(x, y, cos_t, sin_t):
    return x * cos_t - y * sin_t, x * sin_t + y * cos_t

def get_sift_descriptor(grad_mag, grad_dir, kpt, orientation):
    x, y, scale, _ = kpt
    cos_t, sin_t = np.cos(np.radians(np.copy(orientation))), np.sin(np.radians(np.copy(orientation)))
    
    # Fixed descriptor parameters
    desc_width = 16
    n_subregions = 4
    n_bins = 8
    sigma = 6.4  # used for weighting grad_magnitudes to make pixels near the center (feature) more effective than the far ones. (6.4 = 0.4 * desc width which is 16)
    desc = np.zeros(n_subregions * n_subregions * n_bins)  # initializing 128 D descriptor
    
    # to form the 16x16 region where the feature is centered 
    grid = np.arange(-8, 8)  
    
    for i, y_rot in enumerate(grid):
        for j, x_rot in enumerate(grid):
            # Rotate coordinates
            x_r = x_rot*cos_t - y_rot*sin_t
            y_r = x_rot*sin_t + y_rot*cos_t
            
            # Sample image
            xi, yi = int(x + x_r), int(y + y_r)
            if not (0 <= xi < grad_mag.shape[1] and 0 <= yi < grad_mag.shape[0]):
                continue
                
            # Simplified binning (no Gaussian weighting)
            angle = (grad_dir[yi, xi] - orientation) % (360)
            sub_x = j // 4  # Integer division for subregions
            sub_y = i // 4
            bin_idx = int(angle / (360/n_bins))
            
            # Accumulate magnitudes
            desc_idx = (sub_y*n_subregions + sub_x)*n_bins + bin_idx  # for sub_y = 0, sub_x = 1 (second subregion on the right)
                                                                      # desc_idx = (0 + 1) * 8 + bin_idx = 8 + bin_idx as we concatinate bins
                                                                      # desc_idx is the idx of the bin in the 128 descriptor (0 -> 127) 
            mag_weight = np.exp(-(x_r**2 + y_r**2)/ (2 * (sigma**2)))
            desc[desc_idx] += (mag_weight * grad_mag[yi, xi])  # using weighted magnitudes
    
    # normalization for becoming illumnation invariant
    desc = desc / (np.linalg.norm(desc) + 1e-6)
    desc = np.clip(desc, 0, 0.2)  # Clip large values
    desc = desc / max(np.linalg.norm(desc), 1e-6)  # Renormalize
    
    return desc

def extract_sift_descriptors(image, keypoints):
    oriented_keypoints = []
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.float32)
    
    # Compute gradients mags and directions
    grad_mag, grad_dir = compute_gradients_opencv(image)  # grad directions are in degrees
    
    descriptors = []
    
    for kpt in keypoints:
        x, y, scale, _ = kpt
        
        # Get orientation histogram for the feature
        hist = get_orientation_histogram(grad_mag, grad_dir, (x, y), scale)  # we get orientations around the feature with a certain radius
        
        # Get dominant orientations for the feature
        orientations = get_dominant_orientations(hist)
        
        # Create descriptor for each orientation for the feature (if there are many dominant orientations)
        for orientation in orientations:
            oriented_keypoints.append(cv2.KeyPoint(x, y, 10, orientation))
            desc = get_sift_descriptor(grad_mag, grad_dir, kpt, orientation)
            descriptors.append(desc)
    
    return np.array(descriptors), np.array(oriented_keypoints)
