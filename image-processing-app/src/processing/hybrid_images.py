from ..processing.frequency_domain import gaussian_low_pass_filter
from ..processing.frequency_domain import butterworth_high_pass_filter
import numpy as np

def create_hybrid_image(image1, image2, alpha=0.5):
    lowPassFilteredImg = gaussian_low_pass_filter(image1)
    highPassFilteredImg = butterworth_high_pass_filter(image2, 10, 3)
    hybrid_image = alpha * lowPassFilteredImg + (1 - alpha) * highPassFilteredImg
    hybrid_image = np.clip(hybrid_image, 0, 255).astype(np.uint8)
    return hybrid_image