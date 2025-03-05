import numpy as np
class Histogram:


    @staticmethod
    def draw_histogram(image:np.ndarray):
        freq=Histogram.frequency_of_grey_levels(image)
        cdf=freq.copy()
        cdf=cdf.cumsum()
        Histogram.__draw(freq, cdf)

    @staticmethod
    def __draw(freq,cdf):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        # Plot the first subplot
        axes[0].bar(np.arange(0, 256, 1), freq, label='pdf', color='blue', linestyle='-', linewidth=2)
        axes[0].set_title('Grayscale Histogram (Bar Chart)')
        axes[0].set_xlabel('Pixel Intensity')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].grid(True)
        axes[0].set_xlim([0, 255])

        # Plot the second subplot
        axes[1].bar(np.arange(0, 256, 1), cdf, label='cdf', color='red', linestyle='--', linewidth=2)
        axes[1].set_title('Grayscale Histogram (Bar Chart)')
        axes[1].set_xlabel('Pixel Intensity')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        axes[1].grid(True)
        axes[1].set_xlim([0, 255])

        # Adjust layout for better spacing
        plt.tight_layout()

        # Display the plots
        plt.show()

    @staticmethod
    def frequency_of_grey_levels(image: np.ndarray):
        freq = np.zeros(shape=(256,), dtype=np.int32)
        for grey_level in range(256):
            freq[grey_level] = (image == grey_level).sum()
        return freq

    @staticmethod
    def equalize(image: np.ndarray):
        output_image = image.copy()
        freq = Histogram.frequency_of_grey_levels(image)

        pmf = freq / image.size
        cdf = pmf.cumsum() * 255.0

        new_level = np.clip(np.rint(cdf), 0, 255).astype(np.uint8)
        print(new_level)
        lookup_table = new_level.copy()
        output_image = lookup_table[image]
        # Histogram.draw_histogram(output_image)

        return output_image
    @staticmethod
    def normalize_image(image, target_min=0, target_max=255):
        
        img_min, img_max = image.min(), image.max()

       
        if img_max - img_min == 0:
            return np.full_like(image, target_min, dtype=np.uint8)

        
        normalized = (image - img_min) / (img_max - img_min) * (target_max - target_min) + target_min

        return normalized.astype(np.uint8)