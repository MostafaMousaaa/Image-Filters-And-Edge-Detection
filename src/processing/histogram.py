from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout
from PyQt6.QtGui import QPainter, QColor
import numpy as np
import cv2

class HistogramWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.histogram = None
        self.cumulative_distribution = None
        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout()
        self.histogram_label = QLabel("Histogram")
        self.layout.addWidget(self.histogram_label)
        self.setLayout(self.layout)

    def set_image(self, image):
        self.image = image
        self.calculate_histogram()

    def calculate_histogram(self):
        if self.image is not None:
            # Convert to grayscale if the image is colored
            if len(self.image.shape) == 3:
                gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = self.image

            # Calculate histogram
            self.histogram, _ = np.histogram(gray_image.flatten(), bins=256, range=[0, 256])
            self.cumulative_distribution = np.cumsum(self.histogram)

            # Update the UI
            self.update()

    def paintEvent(self, event):
        if self.histogram is not None:
            painter = QPainter(self)
            width = self.width()
            height = self.height()

            # Normalize histogram for display
            normalized_histogram = (self.histogram / self.histogram.max()) * height

            # Draw histogram
            for x in range(len(normalized_histogram)):
                painter.drawLine(x, height, x, height - normalized_histogram[x])

            # Draw cumulative distribution
            if self.cumulative_distribution is not None:
                normalized_cdf = (self.cumulative_distribution / self.cumulative_distribution.max()) * height
                for x in range(len(normalized_cdf)):
                    painter.setPen(QColor(255, 0, 0))  # Red color for CDF
                    painter.drawLine(x, height, x, height - normalized_cdf[x])