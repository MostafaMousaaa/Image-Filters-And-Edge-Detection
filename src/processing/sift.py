import cv2
import math
from filters import gaussian_filter_custom

def generateSiftDescriptors(img, octaveLayersNum, sigma, keypointThreshold, edgeThreshold):
    if img is None:
        return None
    
    # Scale Space Construction
    k = math.sqrt(2)
    scaleLevels = [sigma, sigma * k, sigma * 2, sigma * 2 * k, sigma * 2 * k * k]
    octaveLevels = [img]

    for _ in range(octaveLayersNum - 1):
        prevImage = octaveLevels[-1]
        height, width = prevImage.shape
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
    