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