from PyQt6.QtWidgets import (QMainWindow, QFileDialog, QVBoxLayout, QWidget, 
                            QHBoxLayout, QDockWidget, QTabWidget, QToolBar, QLabel, 
                            QComboBox, QSlider, QPushButton, QApplication, QGroupBox,
                            QSpinBox, QDoubleSpinBox, QRadioButton, QButtonGroup,
                            QSplitter, QFrame, QMessageBox, QToolButton, QStatusBar)
from PyQt6.QtGui import QIcon, QPixmap, QImage, QAction, QFont, QKeySequence, QActionGroup
from PyQt6.QtCore import Qt, QSize, QTimer, QPropertyAnimation, QEasingCurve, QSettings

import cv2
import numpy as np
import sys
import os
import inspect

from ..ui.widgets.image_display_widget import ImageDisplayWidget
from ..ui.widgets.histogram_widget import HistogramWidget
from ..ui.widgets.filter_controls_widget import FilterControlsWidget
from ..ui.widgets.dual_image_view import DualImageView
from ..utils.image_utils import load_image, convert_to_grayscale, equalize_histogram, normalize_image
from ..processing.noise import add_gaussian_noise, add_salt_and_pepper_noise, add_uniform_noise
from ..processing.filters import apply_low_pass_filter
from ..processing.edge_detection import (sobel_edge_detection, roberts_edge_detection, 
                                        prewitt_edge_detection, canny_edge_detection)
from ..processing.thresholding import global_threshold, local_threshold
from ..processing.frequency_domain import apply_low_pass_filter as apply_freq_lpf
from ..processing.frequency_domain import apply_high_pass_filter as apply_freq_hpf
from ..processing.hybrid_images import create_hybrid_image
from ..ui.icons import icons

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Computer Vision - Image Processing App")
        self.setGeometry(50, 50, 1200, 800)
        
        # Try to set application icon
        try:
            self.setWindowIcon(QIcon("resources/app_icon.png"))
        except:
            pass
        
        # Image data
        self.original_image = None
        self.current_image = None
        self.second_image = None  # For hybrid images
        
        # Initialize UI components
        self.init_ui()
        
        # Status bar message timer
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.clear_status)
        
        # Restore previous session state if available
        self._restore_settings()
        
        # Disable image operation buttons initially
        # self.enable_image_operations(False)
        
    def init_ui(self):
        # Create central widget with layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Create a main splitter for resizable panels
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout = QVBoxLayout(self.central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.addWidget(self.main_splitter)
        
        # Create menu bar
        self.setup_menu()
        
        # Create toolbar
        self.setup_toolbar()
        
        # Create image display area
        self.setup_image_display()
        
        # Create right sidebar with tabs for different functionalities
        self.setup_sidebar()
        
        # Status bar for showing information
        self.setup_status_bar()
        
        # Set initial splitter sizes (70% for image, 30% for sidebar)
        self.main_splitter.setSizes([700, 300])

    def setup_menu(self):
        # Menu Bar
        menubar = self.menuBar()
        
        # File Menu
        file_menu = menubar.addMenu("&File")
        
        # Open action
        open_action = QAction(QIcon(":/icons/open.png"), "&Open Image", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)
        
        # Open second image (for hybrid images)
        # open_second_action = QAction(QIcon(":/icons/open.png"), "Open &Second Image", self)
        # open_second_action.triggered.connect(self.open_second_image)
        # file_menu.addAction(open_second_action)
        
        # Save action
        save_action = QAction(QIcon(":/icons/save.png"), "&Save Result", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_image)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(QApplication.quit)
        file_menu.addAction(exit_action)
        
        # Edit Menu
        edit_menu = menubar.addMenu("&Edit")
        
        # Reset to original
        reset_action = QAction(QIcon(":/icons/reset.png"), "&Reset to Original", self)
        reset_action.triggered.connect(self.reset_to_original)
        edit_menu.addAction(reset_action)
        
        # Convert to grayscale
        grayscale_action = QAction(QIcon(":/icons/grayscale.png"), "Convert to &Grayscale", self)
        grayscale_action.triggered.connect(self.convert_to_grayscale)
        edit_menu.addAction(grayscale_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)
        
        # Recent files submenu
        self.recent_files_menu = file_menu.addMenu("Recent Files")
        self._update_recent_files()

    def setup_toolbar(self):
        # Main toolbar
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        # File operations group
        file_group = QActionGroup(self)
        
        # Open image button
        open_action = QAction(QIcon(icons.OPEN), "Open Image", self)
        open_action.setToolTip("Open an image file")
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_image)
        file_group.addAction(open_action)
        
        # Save image button
        save_action = QAction(QIcon(icons.SAVE), "Save Image", self)
        save_action.setToolTip("Save current image")
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        file_group.addAction(save_action)
        
        toolbar.addActions(file_group.actions())
        toolbar.addSeparator()
        
        # Edit operations group
        edit_group = QActionGroup(self)
        
        # Reset to original button
        reset_action = QAction(QIcon(":/icons/reset.png"), "Reset to Original", self)
        reset_action.setToolTip("Reset to original image")
        reset_action.triggered.connect(self.reset_to_original)
        edit_group.addAction(reset_action)
        
        # Convert to grayscale button
        grayscale_action = QAction(QIcon(":/icons/grayscale.png"), "Convert to Grayscale", self)
        grayscale_action.setToolTip("Convert image to grayscale")
        grayscale_action.triggered.connect(self.convert_to_grayscale)
        edit_group.addAction(grayscale_action)
        
        toolbar.addActions(edit_group.actions())
        toolbar.addSeparator()
        
        # Enhancement operations group
        enhance_group = QActionGroup(self)
        
        # Equalize histogram button
        equalize_action = QAction(QIcon(":/icons/equalize.png"), "Equalize Histogram", self)
        equalize_action.setToolTip("Enhance contrast using histogram equalization")
        equalize_action.triggered.connect(self.equalize_histogram)
        enhance_group.addAction(equalize_action)
        
        # Normalize image button
        normalize_action = QAction(QIcon(":/icons/normalize.png"), "Normalize Image", self)
        normalize_action.setToolTip("Normalize image intensity values")
        normalize_action.triggered.connect(self.normalize_image)
        enhance_group.addAction(normalize_action)
        
        toolbar.addActions(enhance_group.actions())

    def setup_image_display(self):
        # Left panel for image display - create a container widget
        self.image_panel = QWidget()
        image_layout = QVBoxLayout(self.image_panel)
        image_layout.setContentsMargins(0, 0, 0, 0)
        
        # Image display widget
        self.image_display = ImageDisplayWidget()
        
        # Frame for image area with proper object name for styling
        image_frame = QFrame()
        image_frame.setObjectName("imageFrame")
        # Fix: Set frame shape and shadow separately
        image_frame.setFrameShape(QFrame.Shape.StyledPanel)
        image_frame.setFrameShadow(QFrame.Shadow.Sunken)
        image_frame_layout = QVBoxLayout(image_frame)
        image_frame_layout.addWidget(self.image_display)
        
        # Add image display to layout
        image_layout.addWidget(image_frame, 7)
        
        # Histogram widget below the image
        self.histogram_widget = HistogramWidget()
        
        # Frame for histogram area with proper object name for styling
        hist_frame = QFrame()
        hist_frame.setObjectName("histFrame")
        # Fix: Set frame shape and shadow separately
        hist_frame.setFrameShape(QFrame.Shape.StyledPanel)
        hist_frame.setFrameShadow(QFrame.Shadow.Sunken)
        hist_frame_layout = QVBoxLayout(hist_frame)
        hist_frame_layout.addWidget(self.histogram_widget)
        
        # Add histogram to layout
        image_layout.addWidget(hist_frame, 3)
        
        # Add the image panel to the splitter
        self.main_splitter.addWidget(self.image_panel)

    def setup_sidebar(self):
        # Right sidebar with tabs
        self.sidebar = QTabWidget()
        self.sidebar.setTabPosition(QTabWidget.TabPosition.North)
        self.sidebar.setDocumentMode(True)
        
        # Add the sidebar to the splitter
        self.main_splitter.addWidget(self.sidebar)
        
        # Noise tab
        self.setup_noise_tab()
        
        # Filters tab
        self.setup_filters_tab()
        
        # Edge Detection tab
        self.setup_edge_detection_tab()
        
        # Thresholding tab
        self.setup_threshold_tab()
        
        # Frequency Domain tab
        self.setup_frequency_domain_tab()
        
        
        # Dual Image View tab
        self.setup_dual_image_tab()
        
        # Set tab icons if available
        try:
            self.sidebar.setTabIcon(0, QIcon(":/icons/noise.png"))
            self.sidebar.setTabIcon(1, QIcon(":/icons/filter.png"))
            self.sidebar.setTabIcon(2, QIcon(":/icons/edge.png"))
            self.sidebar.setTabIcon(3, QIcon(":/icons/threshold.png"))
            self.sidebar.setTabIcon(4, QIcon(":/icons/frequency.png"))
            
        except:
            # Skip icons if not available
            pass

    def setup_noise_tab(self):
        noise_widget = QWidget()
        noise_layout = QVBoxLayout(noise_widget)
        
        # Group box for noise type selection
        noise_group = QGroupBox("Add Noise")
        noise_type_layout = QVBoxLayout()
        
        # Noise type selector
        noise_type_label = QLabel("Noise Type:")
        self.noise_type_combo = QComboBox()
        self.noise_type_combo.addItems(["Gaussian", "Salt & Pepper", "Uniform"])
        noise_type_layout.addWidget(noise_type_label)
        noise_type_layout.addWidget(self.noise_type_combo)
        
        # Parameters for Gaussian noise
        self.gaussian_params = QWidget()
        gaussian_layout = QVBoxLayout(self.gaussian_params)
        
        mean_label = QLabel("Mean:")
        self.mean_slider = QSlider(Qt.Orientation.Horizontal)
        self.mean_slider.setRange(0, 100)
        self.mean_slider.setValue(0)
        self.mean_value = QLabel("0")
        self.mean_slider.valueChanged.connect(lambda v: self.mean_value.setText(str(v)))
        
        sigma_label = QLabel("Sigma:")
        self.sigma_slider = QSlider(Qt.Orientation.Horizontal)
        self.sigma_slider.setRange(1, 100)
        self.sigma_slider.setValue(25)
        self.sigma_value = QLabel("25")
        self.sigma_slider.valueChanged.connect(lambda v: self.sigma_value.setText(str(v)))
        
        gaussian_layout.addWidget(mean_label)
        gaussian_layout.addWidget(self.mean_slider)
        gaussian_layout.addWidget(self.mean_value)
        gaussian_layout.addWidget(sigma_label)
        gaussian_layout.addWidget(self.sigma_slider)
        gaussian_layout.addWidget(self.sigma_value)
        
        # Parameters for Salt & Pepper noise
        self.sp_params = QWidget()
        sp_layout = QVBoxLayout(self.sp_params)
        
        salt_label = QLabel("Salt Probability:")
        self.salt_slider = QSlider(Qt.Orientation.Horizontal)
        self.salt_slider.setRange(1, 100)
        self.salt_slider.setValue(10)
        self.salt_value = QLabel("0.01")
        self.salt_slider.valueChanged.connect(lambda v: self.salt_value.setText(f"{v/1000:.3f}"))
        
        pepper_label = QLabel("Pepper Probability:")
        self.pepper_slider = QSlider(Qt.Orientation.Horizontal)
        self.pepper_slider.setRange(1, 100)
        self.pepper_slider.setValue(10)
        self.pepper_value = QLabel("0.01")
        self.pepper_slider.valueChanged.connect(lambda v: self.pepper_value.setText(f"{v/1000:.3f}"))
        
        sp_layout.addWidget(salt_label)
        sp_layout.addWidget(self.salt_slider)
        sp_layout.addWidget(self.salt_value)
        sp_layout.addWidget(pepper_label)
        sp_layout.addWidget(self.pepper_slider)
        sp_layout.addWidget(self.pepper_value)
        
        # Parameters for Uniform noise
        self.uniform_params = QWidget()
        uniform_layout = QVBoxLayout(self.uniform_params)
        
        low_label = QLabel("Low Value:")
        self.low_slider = QSlider(Qt.Orientation.Horizontal)
        self.low_slider.setRange(0, 100)
        self.low_slider.setValue(0)
        self.low_value = QLabel("0")
        self.low_slider.valueChanged.connect(lambda v: self.low_value.setText(str(v)))
        
        high_label = QLabel("High Value:")
        self.high_slider = QSlider(Qt.Orientation.Horizontal)
        self.high_slider.setRange(1, 100)
        self.high_slider.setValue(50)
        self.high_value = QLabel("50")
        self.high_slider.valueChanged.connect(lambda v: self.high_value.setText(str(v)))
        
        uniform_layout.addWidget(low_label)
        uniform_layout.addWidget(self.low_slider)
        uniform_layout.addWidget(self.low_value)
        uniform_layout.addWidget(high_label)
        uniform_layout.addWidget(self.high_slider)
        uniform_layout.addWidget(self.high_value)
        
        # Initial state: show Gaussian parameters
        self.sp_params.hide()
        self.uniform_params.hide()
        
        # Connect the combo box to show/hide parameter widgets
        self.noise_type_combo.currentTextChanged.connect(self.update_noise_params)
        
        # Add all parameter widgets to the layout
        noise_type_layout.addWidget(self.gaussian_params)
        noise_type_layout.addWidget(self.sp_params)
        noise_type_layout.addWidget(self.uniform_params)
        
        # Apply button
        apply_noise_button = QPushButton("Apply Noise")
        apply_noise_button.clicked.connect(self.apply_noise)
        noise_type_layout.addWidget(apply_noise_button)
        
        noise_group.setLayout(noise_type_layout)
        noise_layout.addWidget(noise_group)
        
        # Add stretch to push everything up
        noise_layout.addStretch()
        
        self.sidebar.addTab(noise_widget, "Noise")

    def setup_filters_tab(self):
        filters_widget = QWidget()
        filters_layout = QVBoxLayout(filters_widget)
        
        # Group box for filter selection
        filter_group = QGroupBox("Apply Filter")
        filter_layout = QVBoxLayout()
        
        # Filter type selector
        filter_label = QLabel("Filter Type:")
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["Average", "Gaussian", "Median"])
        filter_layout.addWidget(filter_label)
        filter_layout.addWidget(self.filter_combo)
        
        # Parameters for Average and Median filters
        kernel_label = QLabel("Kernel Size:")
        self.kernel_size = QSpinBox()
        self.kernel_size.setRange(3, 15)
        self.kernel_size.setSingleStep(2)  # To ensure odd numbers
        self.kernel_size.setValue(3)
        filter_layout.addWidget(kernel_label)
        filter_layout.addWidget(self.kernel_size)
        
        # Parameters for Gaussian filter
        sigma_label = QLabel("Sigma:")
        self.filter_sigma = QDoubleSpinBox()
        self.filter_sigma.setRange(0.1, 5.0)
        self.filter_sigma.setSingleStep(0.1)
        self.filter_sigma.setValue(1.0)
        filter_layout.addWidget(sigma_label)
        filter_layout.addWidget(self.filter_sigma)
        
        # Apply button
        apply_filter_button = QPushButton("Apply Filter")
        apply_filter_button.clicked.connect(self.apply_filter)
        filter_layout.addWidget(apply_filter_button)
        
        filter_group.setLayout(filter_layout)
        filters_layout.addWidget(filter_group)
        
        # Add stretch to push everything up
        filters_layout.addStretch()
        
        self.sidebar.addTab(filters_widget, "Filters")

    def setup_edge_detection_tab(self):
        edge_widget = QWidget()
        edge_layout = QVBoxLayout(edge_widget)
        
        # Group box for edge detection
        edge_group = QGroupBox("Edge Detection")
        edge_inner_layout = QVBoxLayout()
        
        # Edge detection type
        edge_label = QLabel("Edge Detector:")
        self.edge_combo = QComboBox()
        self.edge_combo.addItems(["Sobel", "Roberts", "Prewitt", "Canny"])
        edge_inner_layout.addWidget(edge_label)
        edge_inner_layout.addWidget(self.edge_combo)
        
        # Direction options for non-Canny detectors
        self.direction_group = QGroupBox("Direction")
        direction_layout = QVBoxLayout()
        self.direction_combo = QComboBox()
        self.direction_combo.addItems(["Magnitude", "X Direction", "Y Direction"])
        direction_layout.addWidget(self.direction_combo)
        self.direction_group.setLayout(direction_layout)
        edge_inner_layout.addWidget(self.direction_group)
        
        # Parameters for Canny
        self.canny_params = QWidget()
        canny_layout = QVBoxLayout(self.canny_params)
        
        low_thresh_label = QLabel("Low Threshold:")
        self.low_threshold = QSlider(Qt.Orientation.Horizontal)
        self.low_threshold.setRange(0, 255)
        self.low_threshold.setValue(50)
        self.low_threshold_value = QLabel("50")
        self.low_threshold.valueChanged.connect(lambda v: self.low_threshold_value.setText(str(v)))
        
        high_thresh_label = QLabel("High Threshold:")
        self.high_threshold = QSlider(Qt.Orientation.Horizontal)
        self.high_threshold.setRange(0, 255)
        self.high_threshold.setValue(150)
        self.high_threshold_value = QLabel("150")
        self.high_threshold.valueChanged.connect(lambda v: self.high_threshold_value.setText(str(v)))
        
        canny_layout.addWidget(low_thresh_label)
        canny_layout.addWidget(self.low_threshold)
        canny_layout.addWidget(self.low_threshold_value)
        canny_layout.addWidget(high_thresh_label)
        canny_layout.addWidget(self.high_threshold)
        canny_layout.addWidget(self.high_threshold_value)
        
        self.canny_params.hide()  # Initially hidden
        
        # Connect to show/hide parameters based on edge detector type
        self.edge_combo.currentTextChanged.connect(self.update_edge_params)
        
        edge_inner_layout.addWidget(self.canny_params)
        
        # Apply button
        apply_edge_button = QPushButton("Apply Edge Detection")
        apply_edge_button.clicked.connect(self.apply_edge_detection)
        edge_inner_layout.addWidget(apply_edge_button)
        
        edge_group.setLayout(edge_inner_layout)
        edge_layout.addWidget(edge_group)
        
        # Add stretch to push everything up
        edge_layout.addStretch()
        
        self.sidebar.addTab(edge_widget, "Edge Detection")

    def setup_threshold_tab(self):
        threshold_widget = QWidget()
        threshold_layout = QVBoxLayout(threshold_widget)
        
        # Group box for thresholding
        threshold_group = QGroupBox("Thresholding")
        threshold_inner_layout = QVBoxLayout()
        
        # Threshold type (Global vs Local)
        type_label = QLabel("Threshold Type:")
        self.threshold_type = QButtonGroup()
        global_radio = QRadioButton("Global")
        local_radio = QRadioButton("Local")
        global_radio.setChecked(True)
        self.threshold_type.addButton(global_radio, 0)
        self.threshold_type.addButton(local_radio, 1)
        
        threshold_inner_layout.addWidget(type_label)
        threshold_inner_layout.addWidget(global_radio)
        threshold_inner_layout.addWidget(local_radio)
        
        # Parameters for Global thresholding
        self.global_params = QWidget()
        global_layout = QVBoxLayout(self.global_params)
        
        thresh_label = QLabel("Threshold Value:")
        self.thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self.thresh_slider.setRange(0, 255)
        self.thresh_slider.setValue(127)
        self.thresh_value = QLabel("127")
        self.thresh_slider.valueChanged.connect(lambda v: self.thresh_value.setText(str(v)))
        
        global_layout.addWidget(thresh_label)
        global_layout.addWidget(self.thresh_slider)
        global_layout.addWidget(self.thresh_value)
        
        # Parameters for Local thresholding
        self.local_params = QWidget()
        local_layout = QVBoxLayout(self.local_params)
        
        block_label = QLabel("Block Size:")
        self.block_size = QSpinBox()
        self.block_size.setRange(3, 99)
        self.block_size.setSingleStep(2)  # Ensure odd numbers
        self.block_size.setValue(11)
        
        const_label = QLabel("Constant:")
        self.constant = QSpinBox()
        self.constant.setRange(-50, 50)
        self.constant.setValue(2)
        
        local_layout.addWidget(block_label)
        local_layout.addWidget(self.block_size)
        local_layout.addWidget(const_label)
        local_layout.addWidget(self.constant)
        
        self.local_params.hide()  # Initially hidden
        
        # Show/hide parameter widgets based on selected type
        global_radio.toggled.connect(self.update_threshold_params)
        
        threshold_inner_layout.addWidget(self.global_params)
        threshold_inner_layout.addWidget(self.local_params)
        
        # Apply button
        apply_threshold_button = QPushButton("Apply Thresholding")
        apply_threshold_button.clicked.connect(self.apply_thresholding)
        threshold_inner_layout.addWidget(apply_threshold_button)
        
        threshold_group.setLayout(threshold_inner_layout)
        threshold_layout.addWidget(threshold_group)
        
        # Add stretch to push everything up
        threshold_layout.addStretch()
        
        self.sidebar.addTab(threshold_widget, "Thresholding")

    def setup_frequency_domain_tab(self):
        freq_widget = QWidget()
        freq_layout = QVBoxLayout(freq_widget)
        
        # Group box for frequency domain filters
        freq_group = QGroupBox("Frequency Domain Filters")
        freq_inner_layout = QVBoxLayout()
        
        # Filter type (Low Pass vs High Pass)
        type_label = QLabel("Filter Type:")
        self.freq_filter_type = QButtonGroup()
        low_pass_radio = QRadioButton("Low Pass")
        high_pass_radio = QRadioButton("High Pass")
        low_pass_radio.setChecked(True)
        self.freq_filter_type.addButton(low_pass_radio, 0)  # 0 for Low Pass
        self.freq_filter_type.addButton(high_pass_radio, 1)  # 1 for High Pass
        
        freq_inner_layout.addWidget(type_label)
        freq_inner_layout.addWidget(low_pass_radio)
        freq_inner_layout.addWidget(high_pass_radio)
        
        # Filter method (Ideal vs Butterworth)
        method_label = QLabel("Filter Method:")
        self.freq_method_combo = QComboBox()
        self.freq_method_combo.addItems(["Ideal", "Butterworth"])
        freq_inner_layout.addWidget(method_label)
        freq_inner_layout.addWidget(self.freq_method_combo)
        
        # Cutoff frequency
        cutoff_label = QLabel("Cutoff Frequency:")
        self.cutoff_slider = QSlider(Qt.Orientation.Horizontal)
        self.cutoff_slider.setRange(5, 100)
        self.cutoff_slider.setValue(30)
        self.cutoff_value = QLabel("30")
        self.cutoff_slider.valueChanged.connect(lambda v: self.cutoff_value.setText(str(v)))
        
        freq_inner_layout.addWidget(cutoff_label)
        freq_inner_layout.addWidget(self.cutoff_slider)
        freq_inner_layout.addWidget(self.cutoff_value)
        
        # Apply button
        apply_freq_button = QPushButton("Apply Frequency Domain Filter")
        apply_freq_button.clicked.connect(self.apply_frequency_filter)
        freq_inner_layout.addWidget(apply_freq_button)
        
        freq_group.setLayout(freq_inner_layout)
        freq_layout.addWidget(freq_group)
        
        # Add stretch to push everything up
        freq_layout.addStretch()
        
        self.sidebar.addTab(freq_widget, "Frequency Domain")


    def setup_dual_image_tab(self):
        """Create a new tab with a dual image view for comparing two images"""
        # Create dual image view widget
        self.dual_image_view = DualImageView()
        
        # Add instructions at top
        instruction_label = QLabel(""
        )
        instruction_label.setWordWrap(True)
        instruction_label.setStyleSheet("font-style: italic; color: #666; margin: 5px;")
        # Add alpha value display
        alpha_display_layout = QHBoxLayout()
        alpha_display_layout.addWidget(QLabel("Alpha Value:"))
        self.dual_alpha_value = QLabel("0.50")
        self.dual_alpha_value.setStyleSheet("font-weight: bold; color: #0066cc; font-size: 14px;")
        alpha_display_layout.addWidget(self.dual_alpha_value)
        alpha_display_layout.addStretch()
    
        # Create a container widget
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(instruction_label)
        layout.addWidget(self.dual_image_view)
        
        # Connect signals
        self.dual_image_view.image1_loaded.connect(lambda: self.open_image_for_dual_view(1))
        self.dual_image_view.image2_loaded.connect(lambda: self.open_image_for_dual_view(2))
        self.dual_image_view.create_hybrid_clicked.connect(self.create_hybrid_from_dual_view)
        
        # Connect alpha value change to update display
        self.dual_image_view.alpha_value_changed.connect(self.update_dual_alpha_display)
    


        # Add to sidebar
        self.sidebar.addTab(container, "Hybrid Image.")

        
    def update_dual_alpha_display(self, value):
        """Update the alpha value display in the dual image view tab"""
        self.dual_alpha_value.setText(f"{value:.2f}")

    def setup_status_bar(self):
        status = QStatusBar()
        self.setStatusBar(status)
        status.showMessage("Ready")

    def open_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )
        
        if file_name:
            try:
                self.original_image = load_image(file_name)
                if self.original_image is None:
                    self.show_error_message(f"Failed to load image: {file_name}")
                    return
                self.original_image = cv2.resize(self.original_image, (300, 300))
                self.current_image = self.original_image.copy()
                self.update_image_display()
                
                # Update status with filename only (not full path)
                filename_only = os.path.basename(file_name)
                # Update first image status for hybrid tab
                # self.first_image_status.setText(filename_only)
                # self.first_image_status.setStyleSheet("color: #000; font-weight: bold;")
                
                self.show_status_message(f"Loaded image: {filename_only}", 5000)
                
                # Update UI state
                # self.enable_image_operations(True)
                
                # Animate the image display to draw attention
                self.animate_widget(self.image_display)
            except Exception as e:
                self.show_error_message(f"Error loading image: {str(e)}")
        
    def save_image(self):
        if self.current_image is None:
            self.show_status_message("No image to save", 3000)
            return
            
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;All Files (*)"
        )
        
        if file_name:
            try:
                cv2.imwrite(file_name, self.current_image)
                self.show_status_message(f"Image saved as: {file_name}", 5000)
            except Exception as e:
                self.show_error_message(f"Error saving image: {str(e)}")

    def update_image_display(self):
        if self.current_image is None:
            return
        
        
            
        if len(self.current_image.shape) == 3:  # colored image
            # Convert OpenCV BGR to RGB for Qt
            rgb_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb_image.shape
            bytes_per_line = channels * width
            q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        else:
            # Grayscale image
            height, width = self.current_image.shape
            q_image = QImage(self.current_image.data, width, height, width, QImage.Format.Format_Grayscale8)
                
        
        self.image_display.set_image(QPixmap.fromImage(q_image))
        self.update_histogram()

    def update_histogram(self):
        if self.current_image is not None:
            # For grayscale
            if len(self.current_image.shape) == 2:
                self.histogram_widget.set_image_data(self.current_image)
            # For color images
            else:
                # Process each channel
                b, g, r = cv2.split(self.current_image)
                self.histogram_widget.set_image_data(r)  # Just show red channel for now
                # Future improvement: add color histogram display

    def reset_to_original(self):
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.update_image_display()
            self.statusBar().showMessage("Reset to original image")

    def convert_to_grayscale(self):
        if self.current_image is not None:
            self.current_image = convert_to_grayscale(self.current_image)
            self.update_image_display()
            self.statusBar().showMessage("Converted to grayscale")

    def equalize_histogram(self):
        if self.current_image is not None:
            self.current_image = equalize_histogram(self.current_image)
            self.update_image_display()
            self.statusBar().showMessage("Histogram equalized")

    def normalize_image(self):
        if self.current_image is not None:
            self.current_image = normalize_image(self.current_image)
            self.update_image_display()
            self.statusBar().showMessage("Image normalized")

    def update_noise_params(self, noise_type):
        self.gaussian_params.setVisible(noise_type == "Gaussian")
        self.sp_params.setVisible(noise_type == "Salt & Pepper")
        self.uniform_params.setVisible(noise_type == "Uniform")

    def update_edge_params(self, edge_type):
        """Update edge detection parameters based on selected method"""
        self.canny_params.setVisible(edge_type == "Canny")
        self.direction_group.setVisible(edge_type != "Canny")

    def update_threshold_params(self, is_global):
        self.global_params.setVisible(is_global)
        self.local_params.setVisible(not is_global)

    def apply_noise(self):
        if self.current_image is None:
            self.statusBar().showMessage("No image loaded")
            return
        
        noise_type = self.noise_type_combo.currentText()
        
        if noise_type == "Gaussian":
            mean = self.mean_slider.value()
            sigma = self.sigma_slider.value()
            self.current_image = add_gaussian_noise(self.current_image, mean, sigma)
            self.statusBar().showMessage(f"Applied Gaussian noise (mean={mean}, sigma={sigma})")
            
        elif noise_type == "Salt & Pepper":
            salt_prob = self.salt_slider.value() / 1000
            pepper_prob = self.pepper_slider.value() / 1000
            self.current_image = add_salt_and_pepper_noise(self.current_image, salt_prob, pepper_prob)
            self.statusBar().showMessage(f"Applied Salt & Pepper noise (salt={salt_prob}, pepper={pepper_prob})")
            
        elif noise_type == "Uniform":
            low = self.low_slider.value()
            high = self.high_slider.value()
            self.current_image = add_uniform_noise(self.current_image, low, high)
            self.statusBar().showMessage(f"Applied Uniform noise (low={low}, high={high})")

        self.update_image_display()

    def apply_filter(self):
        if self.current_image is None:
            self.statusBar().showMessage("No image loaded")
            return
        
        filter_type = self.filter_combo.currentText()
        kernel_size = self.kernel_size.value()

        if filter_type == "Average":
            self.current_image = apply_low_pass_filter(self.current_image, filter_type, kernel_size)     
            self.statusBar().showMessage(f"Applied Average filter (kernel size={kernel_size})")
        elif filter_type == "Gaussian":
            sigma = self.filter_sigma.value()
            self.current_image = apply_low_pass_filter(self.current_image, filter_type, kernel_size, sigma)
            self.statusBar().showMessage(f"Applied Gaussian filter (kernel size={kernel_size}, sigma={sigma})")
        elif filter_type == "Median":
            self.current_image = apply_low_pass_filter(self.current_image, filter_type, kernel_size)
            self.statusBar().showMessage(f"Applied Median filter (kernel size={kernel_size})")

        self.update_image_display()

    def apply_edge_detection(self):
        if self.current_image is None:
            self.statusBar().showMessage("No image loaded")
            return
        
        edge_type = self.edge_combo.currentText()
        
        # Get direction for non-Canny detectors
        direction = None
        if edge_type != "Canny":
            direction_text = self.direction_combo.currentText()
            if direction_text == "X Direction":
                direction = 'x'
            elif direction_text == "Y Direction":
                direction = 'y'
        
        if edge_type == "Sobel":
            self.current_image = sobel_edge_detection(self.current_image, direction)
            direction_str = f" ({direction_text})" if edge_type != "Canny" else ""
            self.statusBar().showMessage(f"Applied Sobel edge detection{direction_str}")
        elif edge_type == "Roberts":
            self.current_image = roberts_edge_detection(self.current_image, direction)
            direction_str = f" ({direction_text})" if edge_type != "Canny" else ""
            self.statusBar().showMessage(f"Applied Roberts edge detection{direction_str}")
        elif edge_type == "Prewitt":
            self.current_image = prewitt_edge_detection(self.current_image, direction)
            direction_str = f" ({direction_text})" if edge_type != "Canny" else ""
            self.statusBar().showMessage(f"Applied Prewitt edge detection{direction_str}")
        elif edge_type == "Canny":
            low_thresh = self.low_threshold.value()
            high_thresh = self.high_threshold.value()
            self.current_image = canny_edge_detection(self.current_image, low_thresh, high_thresh)
            self.statusBar().showMessage(f"Applied Canny edge detection (low={low_thresh}, high={high_thresh})")

        self.update_image_display()

    def apply_thresholding(self):
        if self.current_image is None:
            self.statusBar().showMessage("No image loaded")
            return
        
        is_global = self.threshold_type.checkedId() == 0

        if is_global:
            threshold_value = self.thresh_slider.value()
            self.current_image = global_threshold(self.current_image, threshold_value)
            self.statusBar().showMessage(f"Applied Global thresholding (threshold={threshold_value})")
        else:
            block_size = self.block_size.value()
            constant = self.constant.value()
            self.current_image = local_threshold(self.current_image, block_size, constant)
            self.statusBar().showMessage(f"Applied Local thresholding (block size={block_size}, constant={constant})")

        self.update_image_display()

    def apply_frequency_filter(self):
        if self.current_image is None:
            self.statusBar().showMessage("No image loaded")
            return
        
        filter_type = self.freq_filter_type.checkedId()
        method = self.freq_method_combo.currentText()
        cutoff = self.cutoff_slider.value()

        if filter_type == 0:  # Low Pass
            self.current_image = apply_freq_lpf(self.current_image, method, cutoff)
            self.statusBar().showMessage(f"Applied Low Pass filter (method={method}, cutoff={cutoff})")
        else:  # High Pass
            self.current_image = apply_freq_hpf(self.current_image, method, cutoff)
            self.statusBar().showMessage(f"Applied High Pass filter (method={method}, cutoff={cutoff})")

        self.update_image_display()

    def open_image_for_dual_view(self, image_number):
        """Open an image for the dual view panel"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, f"Open Image {image_number}", "", "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )
        
        if file_name:
            try:
                image = load_image(file_name)
                if image is None:
                    self.show_error_message(f"Failed to load image: {file_name}")
                    return
                
                # Set the image in the dual view
                if image_number == 1:
                    self.original_image = image
                    self.dual_image_view.set_first_image(image)
                    self.show_status_message(f"Loaded first image: {file_name}", 3000)
                else:
                    self.second_image = image
                    self.dual_image_view.set_second_image(image)
                    self.show_status_message(f"Loaded second image: {file_name}", 3000)
                    
            except Exception as e:
                self.show_error_message(f"Error loading image: {str(e)}")

    def create_hybrid_from_dual_view(self, alpha):
        """Create a hybrid image from the two images in dual view"""
        if self.original_image is None or self.second_image is None:
            self.show_error_message("Both images must be loaded to create a hybrid image")
            return
            
        # Create hybrid image
        hybrid_image = create_hybrid_image(self.original_image, self.second_image, alpha)
        self.current_image = hybrid_image
        self.update_image_display()
        self.show_status_message(f"Created hybrid image with alpha={alpha:.2f}")
        
        # Switch to main display
        self.main_splitter.setSizes([700, 300])  # Refocus on main image panel

    def show_status_message(self, message, timeout=0):
        """Show a message in the status bar with optional timeout"""
        self.statusBar().showMessage(message)
        
        if timeout > 0:
            self.status_timer.start(timeout)
    
    def clear_status(self):
        """Clear the status bar message"""
        self.statusBar().showMessage("Ready")
        self.status_timer.stop()
    
    def show_error_message(self, message):
        """Show an error message dialog"""
        QMessageBox.critical(self, "Error", message)
        self.statusBar().showMessage("Error occurred")
    
    def show_about_dialog(self):
        """Show the about dialog"""
        QMessageBox.about(self, "About Image Processing App",
                         """<h2>Computer Vision - Image Processing App</h2>
                         <p>A comprehensive tool for exploring image processing algorithms.</p>
                         <p>Created as a project for Computer Vision course.</p>
                         <p><b>Team Members:</b><br>
                         [List team member names here]</p>""")
        
    def animate_widget(self, widget):
        """Create a highlight animation effect for a widget"""
        highlight = QFrame(widget)
        highlight.setStyleSheet("background-color: rgba(52, 152, 219, 0.3);")
        highlight.setGeometry(0, 0, widget.width(), widget.height())
        highlight.show()
        
        # Remove the highlight after a delay
        QTimer.singleShot(500, highlight.deleteLater)

    def _restore_settings(self):
        settings = QSettings("CVTeam", "ImageProcessor")
        geometry = settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
            
        state = settings.value("windowState")
        if state:
            self.restoreState(state)

    def _update_recent_files(self):
        settings = QSettings("CVTeam", "ImageProcessor")
        files = settings.value("recentFiles", [])
        self.recent_files_menu.clear()
        
        for file in files:
            action = self.recent_files_menu.addAction(file)
            action.triggered.connect(lambda checked, file=file: self.open_file(file))

    def closeEvent(self, event):
        # Save window state and geometry
        settings = QSettings("CVTeam", "ImageProcessor")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        super().closeEvent(event)

   
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
