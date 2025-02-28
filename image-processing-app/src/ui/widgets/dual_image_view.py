from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QSlider, QGroupBox, QFrame, QSplitter,
                            QSizePolicy)
from PyQt6.QtGui import QPixmap, QImage, QIcon
from PyQt6.QtCore import Qt, pyqtSignal, QSize

import cv2
import numpy as np

class ImagePanel(QWidget):
    """A widget for displaying an image with a title and load button"""
    
    def __init__(self, title="Image", parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # Title
        self.title = QLabel(title)
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        # Image display
        self.image_frame = QFrame()
        # Fix: Set frame shape and shadow separately
        self.image_frame.setFrameShape(QFrame.Shape.Panel)
        self.image_frame.setFrameShadow(QFrame.Shadow.Sunken)
        self.image_layout = QVBoxLayout(self.image_frame)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.setMinimumSize(200, 200)
        self.image_label.setStyleSheet("background-color: #f0f0f0;")
        
        self.image_layout.addWidget(self.image_label)
        
        # Load button
        self.load_button = QPushButton("Load Image")
        
        # Add widgets to layout
        self.layout.addWidget(self.title)
        self.layout.addWidget(self.image_frame)
        self.layout.addWidget(self.load_button)
        
    def set_image(self, cv_image):
        """Set the image from OpenCV image format"""
        if cv_image is None:
            return
            
        pixmap = self.convert_cv_to_pixmap(cv_image)
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)
    
    def convert_cv_to_pixmap(self, cv_image):
        """Convert OpenCV image to QPixmap"""
        # Convert to RGB if color image
        if len(cv_image.shape) == 3:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb_image.shape
            bytes_per_line = channels * width
            q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        else:
            # Grayscale image
            height, width = cv_image.shape
            q_image = QImage(cv_image.data, width, height, width, QImage.Format.Format_Grayscale8)
            
        return QPixmap.fromImage(q_image)

class DualImageView(QWidget):
    """Widget for displaying two images side by side with controls"""
    
    # Define signals
    image1_loaded = pyqtSignal(str)  # Signal emitted when first image is loaded
    image2_loaded = pyqtSignal(str)  # Signal emitted when second image is loaded
    create_hybrid_clicked = pyqtSignal(float)  # Signal emitted with alpha value when create hybrid button is clicked
    alpha_value_changed = pyqtSignal(float)  # Signal for alpha value changes
    

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        
        # Create a splitter for the two images
        self.image_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # First image area
        self.first_image_panel = ImagePanel("First Image")
        self.first_image_panel.load_button.clicked.connect(self.load_first_image_clicked)
        
        # Second image area
        self.second_image_panel = ImagePanel("Second Image")
        self.second_image_panel.load_button.clicked.connect(self.load_second_image_clicked)
        
        # Add panels to splitter
        self.image_splitter.addWidget(self.first_image_panel)
        self.image_splitter.addWidget(self.second_image_panel)
        self.image_splitter.setSizes([1, 1])  # Equal initial sizes
        
        # Add splitter to main layout
        main_layout.addWidget(self.image_splitter)
        
        # Create hybrid controls
        controls_layout = QHBoxLayout()
        
        # Alpha value slider
        alpha_layout = QVBoxLayout()
        alpha_label = QLabel("Blending Factor:")
        alpha_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(50)
        self.alpha_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.alpha_slider.setTickInterval(10)
        
        # Add this line to connect the alpha value changed signal
        self.alpha_slider.valueChanged.connect(
            lambda v: self.alpha_value_changed.emit(v/100)
        )
        
        slider_labels_layout = QHBoxLayout()
        slider_labels_layout.addWidget(QLabel("Image 1"))
        slider_labels_layout.addStretch()
        slider_labels_layout.addWidget(QLabel("Image 2"))
        
        alpha_layout.addWidget(alpha_label)
        alpha_layout.addWidget(self.alpha_slider)
        alpha_layout.addLayout(slider_labels_layout)
        
        # Create hybrid button
        self.create_hybrid_button = QPushButton("Create Hybrid Image")
        self.create_hybrid_button.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                font-weight: bold;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
        """)
        
        # Add controls to layout
        controls_layout.addLayout(alpha_layout)
        controls_layout.addWidget(self.create_hybrid_button)
        
        # Add controls layout to main layout
        main_layout.addLayout(controls_layout)
        
        # Connect signals
        self.create_hybrid_button.clicked.connect(self.create_hybrid_clicked_handler)
        
    def set_first_image(self, cv_image):
        """Set the first image from OpenCV image format"""
        self.first_image_panel.set_image(cv_image)
    
    def set_second_image(self, cv_image):
        """Set the second image from OpenCV image format"""
        self.second_image_panel.set_image(cv_image)
    
    def load_first_image_clicked(self):
        self.image1_loaded.emit("load_first")
    
    def load_second_image_clicked(self):
        self.image2_loaded.emit("load_second")
        
    def create_hybrid_clicked_handler(self):
        alpha = self.alpha_slider.value() / 100  # Convert slider value to value between 0 and 1
        self.create_hybrid_clicked.emit(alpha)
