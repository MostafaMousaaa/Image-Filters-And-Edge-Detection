import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import cv2

class HistogramWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)  # Reduce margins to maximize space
        
        # Set minimum size for better visibility
        self.setMinimumHeight(250)  # Increased minimum height
        
        # Create a tab widget for the different histogram views
        self.tab_widget = QTabWidget()
        self.layout.addWidget(self.tab_widget)
        
        # Create tabs for different histogram displays
        self.grayscale_tab = QWidget()
        self.rgb_tab = QWidget()
        self.distribution_tab = QWidget()
        
        # Setup the tabs
        self.setup_grayscale_tab()
        self.setup_rgb_tab()
        self.setup_distribution_tab()
        
        # Add tabs to widget
        self.tab_widget.addTab(self.grayscale_tab, "Grayscale")
        self.tab_widget.addTab(self.rgb_tab, "RGB")
        self.tab_widget.addTab(self.distribution_tab, "Distribution")
        
        # Image data
        self.image_data = None
        self.is_color = False
        
        # Set size policy to expand
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
    def setup_grayscale_tab(self):
        layout = QVBoxLayout(self.grayscale_tab)
        layout.setContentsMargins(0, 0, 0, 0)  # Reduce margins
        
        # Increased figure size for better visibility
        self.grayscale_figure = Figure(figsize=(8, 4), dpi=80)  # Optimized size
        self.grayscale_canvas = FigureCanvas(self.grayscale_figure)
        self.grayscale_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.grayscale_canvas)
    
    def setup_rgb_tab(self):
        layout = QVBoxLayout(self.rgb_tab)
        layout.setContentsMargins(0, 0, 0, 0)  # Reduce margins
        
        # Increased figure size with more height for three subplots
        self.rgb_figure = Figure(figsize=(8, 6), dpi=80)  # Optimized size
        self.rgb_canvas = FigureCanvas(self.rgb_figure)
        self.rgb_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.rgb_canvas)
    
    def setup_distribution_tab(self):
        layout = QVBoxLayout(self.distribution_tab)
        layout.setContentsMargins(0, 0, 0, 0)  # Reduce margins
        
        # Increased figure size with more height for distribution plots
        self.dist_figure = Figure(figsize=(8, 6), dpi=80)  # Optimized size
        self.dist_canvas = FigureCanvas(self.dist_figure)
        self.dist_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.dist_canvas)
    
    def set_image_data(self, image_data):
        if image_data is None:
            return
                    
        self.image_data = image_data

        if len(self.image_data.shape) == 3:
            self.update_rgb_histogram()
        else:
            self.update_grayscale_histogram()
            
        # self.update_distribution_function()
        pass
    
    def update_grayscale_histogram(self):
        if self.image_data is None:
            return
            
        # Clear the figure
        self.grayscale_figure.clear()
        
        # Create subplot with adjusted size for labels
        ax = self.grayscale_figure.add_subplot(111)
        
        if len(self.image_data.shape) == 2:
            self.gray_histogram = [0] * 256
            for row in range(self.image_data.shape[0]):
                for col in range(self.image_data.shape[1]):
                    self.gray_histogram[self.image_data[row][col]] += 1
            ax.bar(range(256), self.gray_histogram, color='black')
            ax.set_title('Histogram', fontsize=12)
            ax.set_xlabel('Pixel Value', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
        
        # Adjust figure to make room for labels
        self.grayscale_figure.tight_layout()
        self.grayscale_canvas.draw()
    
    def update_rgb_histogram(self):
        if self.image_data is None:
            return
            
        # Clear the figure
        self.rgb_figure.clear()
        
        # Initialize histograms for each channel (Blue, Green, Red)
        self.blue_histogram = [0] * 256
        self.green_histogram = [0] * 256
        self.red_histogram = [0] * 256
        color_histograms = [self.blue_histogram, self.green_histogram, self.red_histogram]
        
        axs = self.rgb_figure.subplots(3, 1, sharex=True)

        for row in range(self.image_data.shape[0]):
                for col in range(self.image_data.shape[1]):
                    for channel in range(self.image_data.shape[2]):
                        color_histograms[channel][self.image_data[row][col][channel]] += 1
        
        # Blue channel
        axs[0].bar(range(256), self.blue_histogram, color='blue')
        axs[0].set_title('Blue Channel', fontsize=12)
        axs[0].set_ylabel('Frequency', fontsize=10)
        
        # Green channel
        axs[1].bar(range(256), self.green_histogram, color='green')
        axs[1].set_title('Green Channel', fontsize=12)
        axs[1].set_ylabel('Frequency', fontsize=10)
        
        # Red channel
        axs[2].bar(range(256), self.red_histogram, color='red')
        axs[2].set_title('Red Channel', fontsize=12)
        axs[2].set_xlabel('Pixel Value', fontsize=10)
        axs[2].set_ylabel('Frequency', fontsize=10)
        
        self.rgb_figure.subplots_adjust(hspace=0.5, top=0.95, bottom=0.1, left=0.15, right=0.95)
        self.rgb_canvas.draw()
    
    def update_distribution_function(self):
        if self.image_data is None:
            return
            
        # Clear the figure
        self.dist_figure.clear()
        
        if len(self.image_data.shape) == 3:
            # Split image into BGR channels
            b_channel = self.image_data[:, :, 0]
            g_channel = self.image_data[:, :, 1]
            r_channel = self.image_data[:, :, 2]

            # Create subplots for each channel's CDF
            axs = self.dist_figure.subplots(3, 1, sharex=True)
            
            def calculate_cdf(histogram, channel):
                total_pixels = channel.size  # Total number of pixels in the channel
                
                # Calculate CDF manually (cumulative sum)
                cdf = [0] * 256
                cumulative_sum = 0
                
                # Build the cumulative sum (CDF)
                for i in range(256):
                    cumulative_sum += histogram[i]
                    cdf[i] = cumulative_sum / total_pixels  # Normalize by total pixels
                
                return cdf

            # Red channel CDF
            cdf_b = calculate_cdf(self.blue_histogram, b_channel)
            axs[0].plot(cdf_b, color='blue')
            axs[0].set_title('Blue Channel CDF', fontsize=12)
            axs[0].set_ylabel('Cumulative %', fontsize=10)
            axs[0].grid(True, alpha=0.3)

            # Green channel CDF
            cdf_g = calculate_cdf(self.green_histogram, g_channel)
            axs[1].plot(cdf_g, color='green')
            axs[1].set_title('Green Channel CDF', fontsize=12)
            axs[1].set_ylabel('Cumulative %', fontsize=10)
            axs[1].grid(True, alpha=0.3)

            # Blue channel CDF
            cdf_r = calculate_cdf(self.red_histogram, r_channel)
            axs[2].plot(cdf_r, color='red')
            axs[2].set_title('Red Channel CDF', fontsize=12)
            axs[2].set_xlabel('Pixel Value', fontsize=10)
            axs[2].set_ylabel('Cumulative %', fontsize=10)
            axs[2].grid(True, alpha=0.3)

            # Adjust layout to fit everything nicely
            self.dist_figure.subplots_adjust(hspace=0.5, top=0.95, bottom=0.1, left=0.15, right=0.95)
            
        else:
            # Grayscale CDF
            ax = self.dist_figure.add_subplot(111)
            
            total_pixels = self.image_data.size  # Total number of pixels in the image
                        
            # Calculate CDF manually (cumulative sum)
            cdf = [0] * 256
            cumulative_sum = 0
            
            # Build the cumulative sum (CDF)
            for i in range(256):
                cumulative_sum += self.gray_histogram[i]
                cdf[i] = cumulative_sum / total_pixels  # Normalize by total pixels
            
            # Plot the CDF for grayscale image
            ax.plot(cdf, color='black')
            ax.set_title('Grayscale CDF', fontsize=12)
            ax.set_xlabel('Pixel Value', fontsize=10)
            ax.set_ylabel('Cumulative %', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            self.dist_figure.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.95)

        # Redraw the canvas
        self.dist_canvas.draw()
