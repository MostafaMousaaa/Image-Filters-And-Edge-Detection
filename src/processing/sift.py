import math
import cv2
import numpy as np
from .filters import gaussian_filter_custom
'''
For given set of images (grayscale and color)
A) Tasks to implement
• Extract the unique features in all images using Harris
operator and λ-. Report computation times to generate these
points.
• Generate feature descriptors using scale invariant features
(SIFT). Report computation time.
• Match the image set features using sum of squared
differences (SSD) and normalized cross correlations. Report
matching computation time.
'''
def generateSiftDescriptors(img, octaveLayersNum, sigma, keypointThreshold, edgeThreshold):
    if img is None:
        return []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Scale Space Construction
    k = 2 ** (1 / octaveLayersNum)
    scaleLevels = []
    for i in range(octaveLayersNum + 3):
        scaleLevels.append(sigma * (k ** i))
    octaveLevels = [img]

    for _ in range(octaveLayersNum - 1):
        prevImage = octaveLevels[-1]
        height = prevImage.shape[0]
        width = prevImage.shape[1]

        kernelSize = 2 * math.ceil(3 * scaleLevels[-1]) + 1
        blurred = cv2.GaussianBlur(prevImage, ksize=(kernelSize, kernelSize), sigmaX=scaleLevels[-1], sigmaY=scaleLevels[-1])
        #blurred = gaussian_filter_custom(image=prevImage, size=kernelSize, sigma=scaleLevels[-1]))
        octaveLevels.append(cv2.resize(blurred, (width // 2, height // 2))) # Saving different image octaves by downscaling the image by half
    
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

            sigmaOrientation = sigma * (2 ** octaveIdx) * (k ** scaleIdx)            
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
                                    keypoints.append((x, y, sigmaOrientation, octaveIdx, scaleIdx))
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
    radius = int(6 * scale)  # typically 6 sigma
    
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
                
            # Gaussian weight based on distance (sigma = 1.5*scale)
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



def get_sift_descriptor(grad_mag, grad_dir, kpt, orientation):
    x, y, scale, _, _ = kpt
    cos_t, sin_t = np.cos(np.radians(np.copy(orientation))), np.sin(np.radians(np.copy(orientation)))
    
    # Fixed descriptor parameters
    desc_width = 16
    n_subregions = 4
    n_bins = 8
    sigma = 6.4  # used for weighting grad_magnitudes to make pixels near the center (feature) more effective than the far ones. (6.4 = 0.4 * desc width which is 16)
    desc = np.zeros(n_subregions * n_subregions * n_bins)  # initializing 128 D descriptor
    
    # to form the 16x16 region where the feature is centered 
    grid = np.arange(-8, 8)  
    
    for i, y_grid in enumerate(grid):
        for j, x_grid in enumerate(grid):
            # Rotate grid coordinates with the dominant orientation angle
            x_grid_rotated = x_grid*cos_t - y_grid*sin_t
            y_grid_rotated = x_grid*sin_t + y_grid*cos_t
            
            # Sample image
            xi, yi = int(x + x_grid_rotated), int(y + y_grid_rotated)
            if not (0 <= xi < grad_mag.shape[1] and 0 <= yi < grad_mag.shape[0]):
                continue
                
            # making our reference orientation towards the zero angle 
            angle = (grad_dir[yi, xi] - orientation) % (360)
            sub_x = j // 4  # Integer division for subregions
            sub_y = i // 4
            bin_idx = int(angle / (360/n_bins))
            
            # Accumulate magnitudes
            desc_idx = (sub_y*n_subregions + sub_x)*n_bins + bin_idx  # for sub_y = 0, sub_x = 1 (second subregion on the right)
                                                                      # desc_idx = (0 + 1) * 8 + bin_idx = 8 + bin_idx as we concatinate bins
                                                                      # desc_idx is the idx of the bin in the 128 descriptor (0 -> 127) 
            mag_weight = np.exp(-(x_grid_rotated**2 + y_grid_rotated**2)/ (2 * (sigma**2)))
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
        x, y, scale, octaveIdx, scaleIdx = kpt
        
        # Get orientation histogram for the feature
        hist = get_orientation_histogram(grad_mag, grad_dir, (x, y), scale)  # we get orientations around the feature with a certain radius
        
        # Get dominant orientations for the feature
        orientations = get_dominant_orientations(hist)
        
        # Create descriptor for each orientation for the feature (if there are many dominant orientations)
        for orientation in orientations:
            oriented_keypoints.append(cv2.KeyPoint(x, y, scale * 2, orientation, response=0, octave=(octaveIdx << 8) | (scaleIdx & 255), class_id=-1))
            desc = get_sift_descriptor(grad_mag, grad_dir, kpt, orientation)
            descriptors.append(desc)
    
    return np.array(descriptors), np.array(oriented_keypoints)

def match_descriptors_ssd(descriptors1, descriptors2, threshold=None):
    """
    Match descriptors using Sum of Squared Differences (SSD).
    Lower values indicate better matches.
    
    Args:
        descriptors1: Descriptors from the first image
        descriptors2: Descriptors from the second image
        threshold: Optional threshold for filtering matches
        
    Returns:
        List of matches (indices of matched descriptors)
    """
    matches = []
    
    # For each descriptor in the first image
    for i, desc1 in enumerate(descriptors1):
        best_match_idx = -1
        min_distance = float('inf')
        
        # Find the closest descriptor in the second image
        for j, desc2 in enumerate(descriptors2):
            # Calculate SSD (sum of squared differences)
            distance = np.sum((desc1 - desc2) ** 2)
            
            if distance < min_distance:
                min_distance = distance
                best_match_idx = j
        # print(min_distance)
        # Apply threshold if provided
        if threshold is None or min_distance < threshold:
            matches.append((i, best_match_idx, min_distance))
        print(len(matches))
    return matches

def match_descriptors_ncc(descriptors1, descriptors2, threshold= None):
    """
    Match descriptors using Normalized Cross Correlation (NCC).
    Higher values indicate better matches (closer to 1).
    
    Args:
        descriptors1: Descriptors from the first image
        descriptors2: Descriptors from the second image
        threshold: Optional threshold for filtering matches
        
    Returns:
        List of matches (indices of matched descriptors)
    """
    matches = []
    
    # For each descriptor in the first image
    for i, desc1 in enumerate(descriptors1):
        best_match_idx = -1
        max_correlation = -1
        
        # Find the closest descriptor in the second image
        for j, desc2 in enumerate(descriptors2):
            # Calculate normalized cross correlation
            # Formula: (A·B) / (||A|| * ||B||)
            dot_product = np.sum(desc1 * desc2)
            norm_product = np.linalg.norm(desc1) * np.linalg.norm(desc2)
            
            # Avoid division by zero
            if norm_product > 0:
                correlation = dot_product / norm_product
            else:
                correlation = 0
                
            if correlation > max_correlation:
                max_correlation = correlation
                best_match_idx = j
        #print(max_correlation)
        # Apply threshold if provided (for NCC, we want values above threshold)
        if threshold is None or max_correlation > threshold:
            matches.append((i, best_match_idx, max_correlation))
    
    return matches

def filter_and_sort_matches(matches, max_matches=50, reverse=False):
    """
    Filter and sort matches for visualization.

    Args:
        matches: List of matches (i, j, score).
        max_matches: Maximum number of matches to return.
        reverse: Sorting order. False for ascending, True for descending.

    Returns:
        Filtered and sorted matches.
    """
    matches = sorted(matches, key=lambda x: x[2], reverse=reverse)
    return matches[:max_matches]

def match_descriptors(descriptors1, descriptors2, method='ssd', threshold=None):
    """
    Match descriptors using the specified method.

    Args:
        descriptors1: Descriptors from the first image
        descriptors2: Descriptors from the second image
        method: 'ssd' for Sum of Squared Differences or 'ncc' for Normalized Cross Correlation
        threshold: Optional threshold for filtering matches

    Returns:
        List of matches with their scores
    """
    start_time = cv2.getTickCount()

    if method.lower() == 'ssd':
        matches = match_descriptors_ssd(descriptors1, descriptors2, threshold = 0.5)  # 0.5
        reverse = False  # Ascending: lower SSD is better
    elif method.lower() == 'ncc':
        matches = match_descriptors_ncc(descriptors1, descriptors2, threshold = 0.69)  # 0.69
        reverse = True   # Descending: higher NCC is better
    else:
        raise ValueError(f"Unknown matching method: {method}. Use 'ssd' or 'ncc'.")

    matches = filter_and_sort_matches(matches, reverse=reverse)

    end_time = cv2.getTickCount()
    matching_time = (end_time - start_time) / cv2.getTickFrequency()

    print(f"Matching computation time ({method}): {matching_time:.4f} seconds")
    return matches

def draw_matches(img1, keypoints1, img2, keypoints2, matches, max_matches=50):
    """
    Draw matches between two images.
    
    Args:
        img1: First image
        keypoints1: Keypoints from the first image
        img2: Second image
        keypoints2: Keypoints from the second image
        matches: List of matches (i, j, score)
        max_matches: Maximum number of matches to draw
        
    Returns:
        Image with drawn matches
    """
    # Sort matches by score (assuming third element is the score)
    # For SSD, lower is better; for NCC, higher is better
    # We'll assume the caller has sorted matches appropriately
    matches = matches[:max_matches]
    
    # Create output image
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    output = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    
    # Place images side by side
    if len(img1.shape) == 2:
        output[:h1, :w1, :] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    else:
        output[:h1, :w1, :] = img1
        
    if len(img2.shape) == 2:
        output[:h2, w1:w1+w2, :] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    else:
        output[:h2, w1:w1+w2, :] = img2
    
    # Draw matches
    for idx1, idx2, _ in matches:
        # Get keypoint coordinates
        x1, y1 = int(keypoints1[idx1].pt[0]), int(keypoints1[idx1].pt[1])
        x2, y2 = int(keypoints2[idx2].pt[0]), int(keypoints2[idx2].pt[1])
        
        # Draw circles at keypoints
        cv2.circle(output, (x1, y1), 4, (0, 255, 0), 1)
        cv2.circle(output, (x2 + w1, y2), 4, (0, 255, 0), 1)
        
        # Draw line between matches
        cv2.line(output, (x1, y1), (x2 + w1, y2), (0, 255, 255), 1)
    
    return output
