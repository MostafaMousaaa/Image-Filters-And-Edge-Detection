from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                            QSlider, QGroupBox, QSpinBox, QTabWidget, QLineEdit,
                            QFrame, QSizePolicy, QFileDialog, QProgressBar)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QImage, QFont
import os
import cv2
import numpy as np

class FaceRecognitionPanel(QWidget):
    """Panel for face recognition system"""
    
    # Define signals
    dataset_loaded = pyqtSignal(str)
    train_model_clicked = pyqtSignal(int)
    select_test_image_clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.dataset_path = None
        self.test_image_path = None
        self.components_count = 10
        self.setObjectName("faceRecognitionPanel")
        self.initUI()
    
    def initUI(self):
        main_layout = QVBoxLayout(self)
        
        # Create tab widget for different views
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        
        # Create main recognition tab
        self.main_tab = QWidget()
        self.setup_main_tab()
        self.tabs.addTab(self.main_tab, "Face Recognition")
        
        # Create eigenfaces tab
        self.eigenfaces_tab = QWidget()
        self.setup_eigenfaces_tab()
        self.tabs.addTab(self.eigenfaces_tab, "Mean Eigenfaces")
        
        # Create reconstruction tab
        self.reconstruction_tab = QWidget()
        self.setup_reconstruction_tab()
        self.tabs.addTab(self.reconstruction_tab, "Reconstruction")
        
        main_layout.addWidget(self.tabs)
    
    def setup_main_tab(self):
        layout = QVBoxLayout(self.main_tab)
        
        # --- Dataset Loading Section ---
        self.load_dataset_btn = QPushButton("Load Dataset")
        self.load_dataset_btn.setStyleSheet("background-color: #4a90e2; color: white;")
        self.load_dataset_btn.setFixedHeight(40)
        self.load_dataset_btn.clicked.connect(self.load_dataset)
        
        # Status indicator for dataset
        self.dataset_label = QLabel("No dataset loaded")
        self.dataset_label.setStyleSheet("color: #666;")
        
        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(self.load_dataset_btn)
        dataset_layout.addWidget(self.dataset_label)
        
        layout.addLayout(dataset_layout)
        
        # --- Training Section ---
        training_group = QGroupBox("Training")
        training_layout = QVBoxLayout()
        
        # Components input
        comp_layout = QHBoxLayout()
        comp_layout.addWidget(QLabel("Components:"))
        
        self.components_spin = QSpinBox()
        self.components_spin.setRange(1, 100)
        self.components_spin.setValue(10)
        self.components_spin.valueChanged.connect(self.update_components_count)
        comp_layout.addWidget(self.components_spin)
        
        training_layout.addLayout(comp_layout)
        
        # Train button
        self.train_btn = QPushButton("Train Model")
        self.train_btn.setObjectName("actionButton")
        self.train_btn.clicked.connect(lambda: self.train_model_clicked.emit(self.components_count))
        training_layout.addWidget(self.train_btn)
        
        training_group.setLayout(training_layout)
        layout.addWidget(training_group)
        
        # --- Testing Section ---
        testing_group = QGroupBox("Testing")
        testing_layout = QVBoxLayout()
        
        self.test_image_btn = QPushButton("Select test image...")
        self.test_image_btn.clicked.connect(self.select_test_image)
        testing_layout.addWidget(self.test_image_btn)
        
        testing_group.setLayout(testing_layout)
        layout.addWidget(testing_group)
        
        # --- Results Section ---
        # Make this section take up more space
        results_section = QWidget()
        results_layout = QHBoxLayout(results_section)
        
        # Input face display
        input_group = QGroupBox("Input Face")
        input_layout = QVBoxLayout()
        
        self.input_face_label = QLabel()
        self.input_face_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input_face_label.setMinimumSize(200, 200)
        self.input_face_label.setStyleSheet("background-color: #f5f5f5;")
        input_layout.addWidget(self.input_face_label)
        
        input_group.setLayout(input_layout)
        
        # Recognition results display
        result_group = QGroupBox("Recognition Result")
        result_layout = QVBoxLayout()
        
        self.result_label = QLabel("No recognition performed yet")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("background-color: #f5f5f5;")
        self.result_label.setMinimumSize(200, 200)
        
        result_layout.addWidget(self.result_label)
        
        result_group.setLayout(result_layout)
        
        results_layout.addWidget(input_group)
        results_layout.addWidget(result_group)
        
        layout.addWidget(results_section)
        
        # Add stretch to push everything to the top
        layout.addStretch()
    
    def setup_eigenfaces_tab(self):
        layout = QVBoxLayout(self.eigenfaces_tab)
        
        info_label = QLabel("This tab will show mean face and principal eigenfaces after training")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        self.eigenfaces_container = QWidget()
        self.eigenfaces_layout = QHBoxLayout(self.eigenfaces_container)
        layout.addWidget(self.eigenfaces_container)
        
        layout.addStretch()
    
    def setup_reconstruction_tab(self):
        layout = QVBoxLayout(self.reconstruction_tab)
        
        info_label = QLabel("This tab will show the reconstruction of faces using different numbers of eigenfaces")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Slider for number of components to use in reconstruction
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Number of components:"))
        
        self.recon_slider = QSlider(Qt.Orientation.Horizontal)
        self.recon_slider.setRange(1, 50)
        self.recon_slider.setValue(10)
        slider_layout.addWidget(self.recon_slider)
        
        self.recon_value_label = QLabel("10")
        slider_layout.addWidget(self.recon_value_label)
        
        layout.addLayout(slider_layout)
        
        # Container for original vs reconstructed images
        recon_container = QWidget()
        recon_layout = QHBoxLayout(recon_container)
        
        original_group = QGroupBox("Original")
        original_layout = QVBoxLayout()
        self.original_face_label = QLabel()
        self.original_face_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_face_label.setMinimumSize(200, 200)
        self.original_face_label.setStyleSheet("background-color: #f5f5f5;")
        original_layout.addWidget(self.original_face_label)
        original_group.setLayout(original_layout)
        
        recon_group = QGroupBox("Reconstructed")
        recon_inner_layout = QVBoxLayout()
        self.recon_face_label = QLabel()
        self.recon_face_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.recon_face_label.setMinimumSize(200, 200)
        self.recon_face_label.setStyleSheet("background-color: #f5f5f5;")
        recon_inner_layout.addWidget(self.recon_face_label)
        recon_group.setLayout(recon_inner_layout)
        
        recon_layout.addWidget(original_group)
        recon_layout.addWidget(recon_group)
        
        layout.addWidget(recon_container)
        
        # Error display
        error_layout = QHBoxLayout()
        error_layout.addWidget(QLabel("Reconstruction Error:"))
        self.error_label = QLabel("--")
        error_layout.addWidget(self.error_label)
        layout.addLayout(error_layout)
        
        # Connect signals
        self.recon_slider.valueChanged.connect(self.update_reconstruction_value)
        
        layout.addStretch()
    
    def load_dataset(self):
        """Open dialog to select dataset folder"""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        
        if folder_path:
            self.dataset_path = folder_path
            # Count images in the folder
            image_count = 0
            for _, _, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.pgm')):
                        image_count += 1
            
            self.dataset_label.setText(f"Dataset loaded: {os.path.basename(folder_path)} ({image_count} images)")
            # Emit the signal with folder path
            self.dataset_loaded.emit(folder_path)
    
    def select_test_image(self):
        """Open dialog to select test image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Test Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.pgm)"
        )
        
        if file_path:
            self.test_image_path = file_path
            # Load and display the image
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self.display_input_face(img)
                self.original_face_label.setPixmap(self.create_pixmap(img))
                # Emit signal after successful image loading
                self.select_test_image_clicked.emit()
    
    def update_components_count(self, value):
        """Update the number of components for PCA"""
        self.components_count = value
    
    def update_reconstruction_value(self, value):
        """Update the number of components for reconstruction"""
        self.recon_value_label.setText(str(value))
        # Here you would trigger reconstruction with the new value
    
    def display_input_face(self, img):
        """Display the input face image"""
        self.input_face_label.setPixmap(self.create_pixmap(img))
    
    def display_result_face(self, img):
        """Display the recognized face result"""
        self.result_label.setPixmap(self.create_pixmap(img))
    
    def display_eigenfaces(self, mean_face, eigenfaces):
        """Display the mean face and top eigenfaces"""
        # Clear previous content
        self.clear_layout(self.eigenfaces_layout)
        
        # Add mean face
        mean_face_frame = QFrame()
        mean_face_layout = QVBoxLayout(mean_face_frame)
        mean_face_img = QLabel()
        mean_face_img.setPixmap(self.create_pixmap(mean_face))
        mean_face_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mean_face_layout.addWidget(mean_face_img)
        mean_face_layout.addWidget(QLabel("Mean Face"))
        
        self.eigenfaces_layout.addWidget(mean_face_frame)
        
        # Add top eigenfaces (first 5)
        for i, eigenface in enumerate(eigenfaces[:5]):
            eigenface_frame = QFrame()
            eigenface_layout = QVBoxLayout(eigenface_frame)
            eigenface_img = QLabel()
            eigenface_img.setPixmap(self.create_pixmap(eigenface))
            eigenface_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
            eigenface_layout.addWidget(eigenface_img)
            eigenface_layout.addWidget(QLabel(f"Eigenface {i+1}"))
            
            self.eigenfaces_layout.addWidget(eigenface_frame)
    
    def display_reconstruction(self, original_img, reconstructed_img, error):
        """Display original vs reconstructed face and error"""
        self.original_face_label.setPixmap(self.create_pixmap(original_img))
        self.recon_face_label.setPixmap(self.create_pixmap(reconstructed_img))
        self.error_label.setText(f"{error:.4f}")
    
    def create_pixmap(self, img):
        """Convert a numpy image to QPixmap"""
        # Ensure the image is properly scaled for display
        if img is None:
            return QPixmap()
        
        # If img is a flat array, reshape it to 2D
        if len(img.shape) == 1:
            # Assume it's a square image
            size = int(np.sqrt(img.size))
            img = img.reshape(size, size)
        
        if len(img.shape) == 2:  # Grayscale image
            height, width = img.shape
            # Normalize to 0-255 if needed
            if img.dtype != np.uint8:
                img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
            
            q_img = QImage(img.data, width, height, width, QImage.Format.Format_Grayscale8)
        else:  # Color image
            height, width, channel = img.shape
            bytes_per_line = 3 * width
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            q_img = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        
        return QPixmap.fromImage(q_img)
    
    def clear_layout(self, layout):
        """Clear all widgets from layout"""
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            elif item.layout() is not None:
                self.clear_layout(item.layout())
