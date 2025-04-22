from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea, QHBoxLayout, QToolButton, QSizePolicy
from PyQt6.QtGui import QPixmap, QIcon
from PyQt6.QtCore import Qt, QSize

class ImageDisplayWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.zoom_level = 1.0
        self.original_pixmap = None
        
    def setup_ui(self):
        # Main layout for image display
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create toolbar with zoom buttons
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(5, 5, 5, 5)
        
        # Zoom info
        self.zoom_label = QLabel("100%")
        self.zoom_label.setStyleSheet("font-weight: bold;")
        
        # Zoom in button
        self.zoom_in_btn = QToolButton()
        self.zoom_in_btn.setText("+")
        self.zoom_in_btn.setToolTip("Zoom In")
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        
        # Zoom out button
        self.zoom_out_btn = QToolButton()
        self.zoom_out_btn.setText("-")
        self.zoom_out_btn.setToolTip("Zoom Out")
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        
        # Reset zoom button
        self.zoom_reset_btn = QToolButton()
        self.zoom_reset_btn.setText("1:1")
        self.zoom_reset_btn.setToolTip("Reset Zoom")
        self.zoom_reset_btn.clicked.connect(self.zoom_reset)
        
        # Image info label
        self.info_label = QLabel("No image")
        self.info_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        
        # Add buttons to toolbar
        toolbar.addWidget(self.zoom_out_btn)
        toolbar.addWidget(self.zoom_label)
        toolbar.addWidget(self.zoom_in_btn)
        toolbar.addWidget(self.zoom_reset_btn)
        toolbar.addStretch()
        toolbar.addWidget(self.info_label)
        
        main_layout.addLayout(toolbar)
        
        # Create a scroll area for the image
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setStyleSheet("background-color: #f5f5f5;")
        
        # Create image label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: transparent;")
        
        # Add image label to scroll area
        self.scroll_area.setWidget(self.image_label)
        
        # Add scroll area to main layout
        main_layout.addWidget(self.scroll_area)
        
    def set_image(self, pixmap):
        """Set the image to display"""
        self.original_pixmap = pixmap
        self.update_display()
        
        # Update image info
        if pixmap:
            width = pixmap.width()
            height = pixmap.height()
            self.info_label.setText(f"Image: {width}×{height} pixels")
        else:
            self.info_label.setText("No image")
    
    def update_display(self):
        """Update the displayed image with the current zoom level"""
        if self.original_pixmap:
            if self.zoom_level == 1.0:
                self.image_label.setPixmap(self.original_pixmap)
            else:
                # Scale the pixmap based on the zoom level
                width = int(self.original_pixmap.width() * self.zoom_level)
                height = int(self.original_pixmap.height() * self.zoom_level)
                
                # Maintain aspect ratio while scaling
                scaled_pixmap = self.original_pixmap.scaled(
                    width, height, 
                    Qt.AspectRatioMode.KeepAspectRatio, 
                    Qt.TransformationMode.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)
            
            # Update the zoom label
            self.zoom_label.setText(f"{int(self.zoom_level * 100)}%")
    
    def zoom_in(self):
        """Increase zoom level"""
        if self.original_pixmap:
            self.zoom_level = min(5.0, self.zoom_level * 1.25)  # Limit max zoom to 500%
            self.update_display()
    
    def zoom_out(self):
        """Decrease zoom level"""
        if self.original_pixmap:
            self.zoom_level = max(0.1, self.zoom_level / 1.25)  # Limit min zoom to 10%
            self.update_display()
    
    def zoom_reset(self):
        """Reset zoom level to 100%"""
        if self.original_pixmap:
            self.zoom_level = 1.0
            self.update_display()