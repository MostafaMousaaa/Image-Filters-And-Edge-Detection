import cv2

def create_hybrid_image(image1, image2, alpha=0.5):
    """
    Create a hybrid image by combining two images with a specified alpha blending factor.
    
    Parameters:
    - image1: The first input image (numpy array).
    - image2: The second input image (numpy array).
    - alpha: The blending factor (0.0 to 1.0).
    
    Returns:
    - hybrid_image: The resulting hybrid image (numpy array).
    """
    # Ensure both images are the same size
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same dimensions.")
    
    # Blend the images
    hybrid_image = cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)
    
    return hybrid_image

def load_and_process_images(image_path1, image_path2):
    """
    Load two images from the specified paths and create a hybrid image.
    
    Parameters:
    - image_path1: Path to the first image.
    - image_path2: Path to the second image.
    
    Returns:
    - hybrid_image: The resulting hybrid image (numpy array).
    """
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)
    
    if image1 is None or image2 is None:
        raise FileNotFoundError("One or both image paths are invalid.")
    
    return create_hybrid_image(image1, image2)

def save_hybrid_image(hybrid_image, output_path):
    """
    Save the hybrid image to the specified output path.
    
    Parameters:
    - hybrid_image: The hybrid image to save (numpy array).
    - output_path: The path where the hybrid image will be saved.
    """
    cv2.imwrite(output_path, hybrid_image)