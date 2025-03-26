from PyQt6.QtWidgets import (QMainWindow, QFileDialog, QVBoxLayout, QWidget, 
                            QHBoxLayout, QDockWidget, QTabWidget, QToolBar, QLabel, 
                            QComboBox, QSlider, QPushButton, QApplication, QGroupBox,
                            QSpinBox, QDoubleSpinBox, QRadioButton, QButtonGroup,
                            QSplitter, QFrame, QMessageBox, QToolButton, QStatusBar, QSizePolicy, QCheckBox)
from PyQt6.QtGui import QIcon, QPixmap, QImage, QAction, QFont, QKeySequence, QActionGroup
from PyQt6.QtCore import Qt, QSize, QTimer, QPropertyAnimation, QEasingCurve, QSettings, pyqtSignal

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
from ..processing.frequency_domain import gaussian_low_pass_filter
from ..processing.frequency_domain import butterworth_high_pass_filter
from ..processing.hybrid_images import create_hybrid_image
from ..ui.icons import icons
from src.ui.edge_detection_panel import EdgeDetectionPanel
from src.ui.active_contour_panel import ActiveContourPanel

class MainWindow(QMainWindow):
    snake_params_changed = pyqtSignal(float, float, float, float, int)
    canny_params_changed = pyqtSignal(int, int, int)
    hough_lines_params_changed = pyqtSignal(float, float, int)
    hough_circles_params_changed = pyqtSignal(float, int, int, int)
    hough_ellipses_params_changed = pyqtSignal(float, int, int, int, int, int)

    def __init__(self, support_edge_detection=False, support_active_contours=False):
        super().__init__()
        self.setWindowTitle("Computer Vision - Image Processing App")
        self.setGeometry(50, 50, 1200, 800)
        
        # Store dock widget references
        self.edge_dock = None
        self.contour_dock = None
        
        # Try to set application icon
        try:
            self.setWindowIcon(QIcon("resources/app_icon.png"))
        except:
            pass
        
        # Image data
        self.original_image = None
        self.current_image = None
        self.first_image = None
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
        
        # Add edge detection and active contour support if requested
        '''if support_edge_detection:
            self._init_edge_detection()
        
        if support_active_contours:
            self._init_active_contours()'''
        
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

        self.setup_contour()

        self.setup_edge_detection()
        
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
        self.block_size.setSingleStep(2)  # Ensure odd numbers so that there is a clear center pixel
        self.block_size.setValue(11)
        
        
        local_layout.addWidget(block_label)
        local_layout.addWidget(self.block_size)
        
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
        self.low_pass_radio = QRadioButton("Low Pass")
        self.high_pass_radio = QRadioButton("High Pass")
        self.low_pass_radio.setChecked(True)
        self.freq_filter_type.addButton(self.low_pass_radio, 0)  # 0 for Low Pass
        self.freq_filter_type.addButton(self.high_pass_radio, 1)  # 1 for High Pass
        self.freq_filter_type.buttonClicked.connect(self.on_filter_type_changed)
        
        freq_inner_layout.addWidget(type_label)
        freq_inner_layout.addWidget(self.low_pass_radio)
        freq_inner_layout.addWidget(self.high_pass_radio)
        
        # Filter method (Ideal vs Butterworth)
        method_label = QLabel("Filter Method:")
        self.freq_method_combo = QComboBox()
        self.freq_method_combo.addItems(["Gaussian", "Butterworth"])
        self.freq_method_combo.currentIndexChanged.connect(self.on_combobox_changed)
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

    def setup_contour(self):
        contour_widget = QWidget()
        main_layout = QVBoxLayout(contour_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)
        
        # Snake parameters - styled consistent with existing UI
        snake_params_group = QGroupBox("Active Contour Model Parameters")
        snake_params_group.setObjectName("paramGroupBox")
        snake_params_layout = QVBoxLayout()
        snake_params_layout.setSpacing(10)
        
        # Alpha (elasticity) parameter
        alpha_layout = QHBoxLayout()
        alpha_label = QLabel("Alpha (elasticity):")
        alpha_label.setObjectName("paramLabel")
        alpha_layout.addWidget(alpha_label)
        
        self.alpha_param = QDoubleSpinBox()
        self.alpha_param.setRange(0.01, 10.0)
        self.alpha_param.setSingleStep(0.1)
        self.alpha_param.setValue(0.5)
        self.alpha_param.setObjectName("paramSpinBox")
        alpha_layout.addWidget(self.alpha_param)
        snake_params_layout.addLayout(alpha_layout)
        
        # Beta (stiffness) parameter
        beta_layout = QHBoxLayout()
        beta_label = QLabel("Beta (stiffness):")
        beta_label.setObjectName("paramLabel")
        beta_layout.addWidget(beta_label)
        
        self.beta_param = QDoubleSpinBox()
        self.beta_param.setRange(0.01, 10.0)
        self.beta_param.setSingleStep(0.1)
        self.beta_param.setValue(0.5)
        self.beta_param.setObjectName("paramSpinBox")
        beta_layout.addWidget(self.beta_param)
        snake_params_layout.addLayout(beta_layout)
        
        # Gamma (time step) parameter
        gamma_layout = QHBoxLayout()
        gamma_label = QLabel("Gamma (time step):")
        gamma_label.setObjectName("paramLabel")
        gamma_layout.addWidget(gamma_label)
        
        self.gamma_param = QDoubleSpinBox()
        self.gamma_param.setRange(0.01, 2.0)
        self.gamma_param.setSingleStep(0.05)
        self.gamma_param.setValue(0.1)
        self.gamma_param.setObjectName("paramSpinBox")
        gamma_layout.addWidget(self.gamma_param)
        snake_params_layout.addLayout(gamma_layout)
        
        # External force weight
        ext_weight_layout = QHBoxLayout()
        ext_weight_label = QLabel("External force weight:")
        ext_weight_label.setObjectName("paramLabel")
        ext_weight_layout.addWidget(ext_weight_label)
        
        self.ext_weight = QDoubleSpinBox()
        self.ext_weight.setRange(0.1, 10.0)
        self.ext_weight.setSingleStep(0.1)
        self.ext_weight.setValue(2.0)
        self.ext_weight.setObjectName("paramSpinBox")
        ext_weight_layout.addWidget(self.ext_weight)
        snake_params_layout.addLayout(ext_weight_layout)
        
        # Max iterations
        iterations_layout = QHBoxLayout()
        iterations_label = QLabel("Max iterations:")
        iterations_label.setObjectName("paramLabel")
        iterations_layout.addWidget(iterations_label)
        
        self.max_iterations = QSpinBox()
        self.max_iterations.setRange(10, 1000)
        self.max_iterations.setSingleStep(10)
        self.max_iterations.setValue(100)
        self.max_iterations.setObjectName("paramSpinBox")
        iterations_layout.addWidget(self.max_iterations)
        snake_params_layout.addLayout(iterations_layout)
        
        snake_params_group.setLayout(snake_params_layout)
        main_layout.addWidget(snake_params_group)
        
        # Action buttons - styled frame
        buttons_group = QGroupBox("Actions")
        buttons_group.setObjectName("actionGroupBox")
        buttons_layout = QVBoxLayout()
        buttons_layout.setSpacing(8)
        
        # Styled buttons consistent with application
        self.initialize_button = QPushButton("Initialize Contour")
        self.initialize_button.setToolTip("Click on image to place initial contour points")
        self.initialize_button.setObjectName("actionButton")
        buttons_layout.addWidget(self.initialize_button)
        
        self.evolve_button = QPushButton("Evolve Contour")
        self.evolve_button.setObjectName("actionButton")
        buttons_layout.addWidget(self.evolve_button)
        
        self.reset_button = QPushButton("Reset Contour")
        self.reset_button.setObjectName("actionButton")
        buttons_layout.addWidget(self.reset_button)
        
        buttons_group.setLayout(buttons_layout)
        main_layout.addWidget(buttons_group)
        
        # Metrics group - styled frame
        metrics_group = QGroupBox("Contour Metrics")
        metrics_group.setObjectName("metricsGroupBox")
        metrics_layout = QVBoxLayout()
        metrics_layout.setSpacing(8)
        
        self.calculate_metrics_button = QPushButton("Calculate Perimeter and Area")
        self.calculate_metrics_button.setObjectName("actionButton")
        metrics_layout.addWidget(self.calculate_metrics_button)
        
        # Metrics display with styled frame
        metrics_display_frame = QFrame()
        metrics_display_frame.setObjectName("resultsFrame")
        metrics_display_frame.setFrameShape(QFrame.Shape.StyledPanel)
        metrics_display_frame.setFrameShadow(QFrame.Shadow.Sunken)
        metrics_display_layout = QHBoxLayout(metrics_display_frame)
        
        perimeter_label = QLabel("Perimeter:")
        perimeter_label.setObjectName("metricLabel")
        metrics_display_layout.addWidget(perimeter_label)
        
        self.perimeter_label = QLabel("--")
        self.perimeter_label.setObjectName("metricValue")
        self.perimeter_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.perimeter_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        metrics_display_layout.addWidget(self.perimeter_label)
        
        area_label = QLabel("Area:")
        area_label.setObjectName("metricLabel")
        metrics_display_layout.addWidget(area_label)
        
        self.area_label = QLabel("--")
        self.area_label.setObjectName("metricValue")
        self.area_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.area_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        metrics_display_layout.addWidget(self.area_label)
        
        metrics_layout.addWidget(metrics_display_frame)
        
        # Chain code display with checkbox
        chain_code_layout = QHBoxLayout()
        self.show_chain_code_checkbox = QCheckBox("Show Chain Code")
        self.show_chain_code_checkbox.setObjectName("optionCheckbox")
        chain_code_layout.addWidget(self.show_chain_code_checkbox)
        metrics_layout.addLayout(chain_code_layout)
        
        metrics_group.setLayout(metrics_layout)
        main_layout.addWidget(metrics_group)
        
        # Add stretch to push everything to the top
        main_layout.addStretch()
        
        # Connect signals
        self.alpha_param.valueChanged.connect(self._emit_snake_params)
        self.beta_param.valueChanged.connect(self._emit_snake_params)
        self.gamma_param.valueChanged.connect(self._emit_snake_params)
        self.ext_weight.valueChanged.connect(self._emit_snake_params)
        self.max_iterations.valueChanged.connect(self._emit_snake_params)
        
        self.initialize_button.clicked.connect(self._on_initialize_contour)
        self.evolve_button.clicked.connect(self._on_evolve_contour)
        self.reset_button.clicked.connect(self._on_reset_contour)
        self.calculate_metrics_button.clicked.connect(self._on_calculate_metrics)
        self.show_chain_code_checkbox.toggled.connect(self._on_show_chain_code)

        self.sidebar.addTab(contour_widget, "Contour")
    
    def _emit_snake_params(self):
        self.snake_params_changed.emit(
            self.alpha_param.value(),
            self.beta_param.value(),
            self.gamma_param.value(),
            self.ext_weight.value(),
            self.max_iterations.value()
        )
    
    def set_metric_values(self, perimeter, area):
        """Update the perimeter and area display values"""
        self.perimeter_label.setText(f"{perimeter:.2f} px")
        self.area_label.setText(f"{area:.2f} px²")

    def setup_edge_detection(self):
        edge_detection_widget = QWidget()
        main_layout = QVBoxLayout(edge_detection_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create tabbed interface with styling consistent with the existing tabs
        tab_widget = QTabWidget()
        tab_widget.setDocumentMode(True)
        tab_widget.setTabPosition(QTabWidget.TabPosition.North)
        
        canny_tab = QWidget()
        hough_tab = QWidget()
        
        # Style the frames with similar appearance to existing UI components
        canny_tab.setObjectName("cannyTab")
        hough_tab.setObjectName("houghTab")
        
        # Set up Canny edge detection tab
        canny_layout = QVBoxLayout(canny_tab)
        canny_layout.setContentsMargins(8, 8, 8, 8)
        
        # Canny parameters - using styled frame
        canny_group = QGroupBox("Canny Edge Detector Parameters")
        canny_group.setObjectName("paramGroupBox")
        canny_params_layout = QVBoxLayout()
        canny_params_layout.setSpacing(10)
        
        # Gaussian blur
        blur_layout = QHBoxLayout()
        blur_label = QLabel("Gaussian blur:")
        blur_label.setObjectName("paramLabel")
        blur_layout.addWidget(blur_label)
        
        self.blur_kernel_size = QSpinBox()
        self.blur_kernel_size.setRange(3, 15)
        self.blur_kernel_size.setSingleStep(2)  # Ensure odd values
        self.blur_kernel_size.setValue(5)
        self.blur_kernel_size.setObjectName("paramSpinBox")
        blur_layout.addWidget(self.blur_kernel_size)
        canny_params_layout.addLayout(blur_layout)
        
        # Low threshold - use styled slider
        low_threshold_layout = QHBoxLayout()
        low_threshold_label = QLabel("Low threshold:")
        low_threshold_label.setObjectName("paramLabel")
        low_threshold_layout.addWidget(low_threshold_label)
        
        self.low_threshold = QSlider(Qt.Orientation.Horizontal)
        self.low_threshold.setRange(0, 255)
        self.low_threshold.setValue(50)
        self.low_threshold.setObjectName("paramSlider")
        
        self.low_threshold_label = QLabel("50")
        self.low_threshold_label.setObjectName("valueLabel")
        self.low_threshold_label.setMinimumWidth(30)
        self.low_threshold_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        
        low_threshold_layout.addWidget(self.low_threshold)
        low_threshold_layout.addWidget(self.low_threshold_label)
        canny_params_layout.addLayout(low_threshold_layout)
        
        # High threshold - use styled slider
        high_threshold_layout = QHBoxLayout()
        high_threshold_label = QLabel("High threshold:")
        high_threshold_label.setObjectName("paramLabel")
        high_threshold_layout.addWidget(high_threshold_label)
        
        self.high_threshold = QSlider(Qt.Orientation.Horizontal)
        self.high_threshold.setRange(0, 255)
        self.high_threshold.setValue(150)
        self.high_threshold.setObjectName("paramSlider")
        
        self.high_threshold_label = QLabel("150")
        self.high_threshold_label.setObjectName("valueLabel")
        self.high_threshold_label.setMinimumWidth(30)
        self.high_threshold_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        
        high_threshold_layout.addWidget(self.high_threshold)
        high_threshold_layout.addWidget(self.high_threshold_label)
        canny_params_layout.addLayout(high_threshold_layout)
        
        # Apply button with consistent styling
        self.apply_canny_button = QPushButton("Apply Canny Edge Detection")
        self.apply_canny_button.setObjectName("actionButton")
        canny_params_layout.addWidget(self.apply_canny_button)
        
        canny_group.setLayout(canny_params_layout)
        canny_layout.addWidget(canny_group)
        canny_layout.addStretch()
        
        # Set up Hough transform tab
        hough_layout = QVBoxLayout(hough_tab)
        hough_layout.setContentsMargins(8, 8, 8, 8)
        
        # Hough lines parameters - using styled frame
        hough_lines_group = QGroupBox("Hough Line Transform Parameters")
        hough_lines_group.setObjectName("paramGroupBox")
        hough_lines_layout = QVBoxLayout()
        hough_lines_layout.setSpacing(10)
        
        # Rho resolution
        rho_layout = QHBoxLayout()
        rho_label = QLabel("Rho resolution:")
        rho_label.setObjectName("paramLabel")
        rho_layout.addWidget(rho_label)
        
        self.rho_resolution = QDoubleSpinBox() # step size for the perpendicular distance between origin and detected line
        self.rho_resolution.setRange(0.1, 10.0)
        self.rho_resolution.setSingleStep(0.1)
        self.rho_resolution.setValue(1.0)
        self.rho_resolution.setObjectName("paramSpinBox")
        rho_layout.addWidget(self.rho_resolution)
        hough_lines_layout.addLayout(rho_layout)
        
        # Theta resolution
        theta_layout = QHBoxLayout()
        theta_label = QLabel("Theta resolution (Degree):")
        theta_label.setObjectName("paramLabel")
        theta_layout.addWidget(theta_label)
        
        self.theta_resolution = QDoubleSpinBox() # step size for theta in radians
        self.theta_resolution.setRange(0.25, 5)
        self.theta_resolution.setSingleStep(0.25)
        self.theta_resolution.setValue(1)
        self.theta_resolution.setObjectName("paramSpinBox")
        theta_layout.addWidget(self.theta_resolution)
        hough_lines_layout.addLayout(theta_layout)
        
        # Threshold for lines - styled slider
        threshold_lines_layout = QHBoxLayout()
        threshold_lines_label = QLabel("Threshold:")
        threshold_lines_label.setObjectName("paramLabel")
        threshold_lines_layout.addWidget(threshold_lines_label)
        
        self.threshold_lines = QSlider(Qt.Orientation.Horizontal) # minimum number of votes required for value to be considered a line
        self.threshold_lines.setRange(10, 300)
        self.threshold_lines.setValue(100)
        self.threshold_lines.setObjectName("paramSlider")
        
        self.threshold_lines_label = QLabel("100")
        self.threshold_lines_label.setObjectName("valueLabel")
        self.threshold_lines_label.setMinimumWidth(30)
        self.threshold_lines_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        
        threshold_lines_layout.addWidget(self.threshold_lines)
        threshold_lines_layout.addWidget(self.threshold_lines_label)
        hough_lines_layout.addLayout(threshold_lines_layout)
        
        # Detect lines button - styled button
        self.detect_lines_button = QPushButton("Detect Lines")
        self.detect_lines_button.setObjectName("actionButton")
        hough_lines_layout.addWidget(self.detect_lines_button)
        
        hough_lines_group.setLayout(hough_lines_layout)
        hough_layout.addWidget(hough_lines_group)
        
        # Hough circles parameters - styled frame
        hough_circles_group = QGroupBox("Hough Circle Transform Parameters")
        hough_circles_group.setObjectName("paramGroupBox")
        hough_circles_layout = QVBoxLayout()
        hough_circles_layout.setSpacing(10)
        
        # DP accumulator resolution
        dp_layout = QHBoxLayout()
        dp_label = QLabel("Accumulator resolution (Theta step size):")
        dp_label.setObjectName("paramLabel")
        dp_layout.addWidget(dp_label)
        
        self.dp_resolution = QDoubleSpinBox()
        self.dp_resolution.setRange(0.1, 1)
        self.dp_resolution.setSingleStep(0.1)
        self.dp_resolution.setValue(1.0)
        self.dp_resolution.setObjectName("paramSpinBox")
        dp_layout.addWidget(self.dp_resolution)
        hough_circles_layout.addLayout(dp_layout)
        
        '''# Min distance between circles
        min_dist_layout = QHBoxLayout()
        min_dist_label = QLabel("Min distance between centers:")
        min_dist_label.setObjectName("paramLabel")
        min_dist_layout.addWidget(min_dist_label)
        
        self.min_distance = QSpinBox()
        self.min_distance.setRange(1, 200)
        self.min_distance.setValue(20)
        self.min_distance.setObjectName("paramSpinBox")
        min_dist_layout.addWidget(self.min_distance)
        hough_circles_layout.addLayout(min_dist_layout)'''
        
        # Min and max radius
        radius_layout = QHBoxLayout()
        min_radius_label = QLabel("Min radius:")
        min_radius_label.setObjectName("paramLabel")
        radius_layout.addWidget(min_radius_label)
        
        self.min_radius = QSpinBox()
        self.min_radius.setRange(0, 500)
        self.min_radius.setValue(10)
        self.min_radius.setObjectName("paramSpinBox")
        radius_layout.addWidget(self.min_radius)
        
        max_radius_label = QLabel("Max radius:")
        max_radius_label.setObjectName("paramLabel")
        radius_layout.addWidget(max_radius_label)
        
        self.max_radius = QSpinBox()
        self.max_radius.setRange(0, 500)
        self.max_radius.setValue(15)
        self.max_radius.setObjectName("paramSpinBox")
        radius_layout.addWidget(self.max_radius)
        hough_circles_layout.addLayout(radius_layout)
        
        # Threshold for circle detection - styled slider
        threshold_circles_layout = QHBoxLayout()
        threshold_circles_label = QLabel("Threshold:")
        threshold_circles_label.setObjectName("paramLabel")
        threshold_circles_layout.addWidget(threshold_circles_label)
        
        self.threshold_circles = QSlider(Qt.Orientation.Horizontal)
        self.threshold_circles.setRange(10, 300)
        self.threshold_circles.setValue(100)
        self.threshold_circles.setObjectName("paramSlider")
        
        self.threshold_circles_label = QLabel("100")
        self.threshold_circles_label.setObjectName("valueLabel")
        self.threshold_circles_label.setMinimumWidth(30)
        self.threshold_circles_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        
        threshold_circles_layout.addWidget(self.threshold_circles)
        threshold_circles_layout.addWidget(self.threshold_circles_label)
        hough_circles_layout.addLayout(threshold_circles_layout)
        
        # Detection buttons - styled buttons
        button_layout = QHBoxLayout()
        
        self.detect_circles_button = QPushButton("Detect Circles")
        self.detect_circles_button.setObjectName("actionButton")
        button_layout.addWidget(self.detect_circles_button)
        
        hough_circles_layout.addLayout(button_layout)
        
        hough_circles_group.setLayout(hough_circles_layout)
        hough_layout.addWidget(hough_circles_group)

         # Hough circles parameters - styled frame
        hough_ellipse_group = QGroupBox("Hough Ellipse Transform Parameters")
        hough_ellipse_group.setObjectName("paramGroupBox")
        hough_ellipse_layout = QVBoxLayout()
        hough_ellipse_layout.setSpacing(10)
        
        # DP accumulator resolution
        dp_ellipse_layout = QHBoxLayout()
        dp_ellipse_label = QLabel("Angle Step (Degree):")
        dp_ellipse_label.setObjectName("paramLabel")
        dp_ellipse_layout.addWidget(dp_ellipse_label)
        
        self.dp_ellipse_resolution = QDoubleSpinBox()
        self.dp_ellipse_resolution.setRange(1, 360)
        self.dp_ellipse_resolution.setSingleStep(1)
        self.dp_ellipse_resolution.setValue(5.0)
        self.dp_ellipse_resolution.setObjectName("paramSpinBox")
        dp_ellipse_layout.addWidget(self.dp_ellipse_resolution)
        hough_ellipse_layout.addLayout(dp_ellipse_layout)
        
        # Min and max radius
        ellipse_radius_layout = QHBoxLayout()
        min_ellipse_radius_label = QLabel("Major Axis Minimum Value:")
        min_ellipse_radius_label.setObjectName("paramLabel")
        ellipse_radius_layout.addWidget(min_ellipse_radius_label)
        
        self.major_axis_min_val = QSpinBox()
        self.major_axis_min_val.setRange(0, 500)
        self.major_axis_min_val.setValue(10)
        self.major_axis_min_val.setObjectName("paramSpinBox")
        ellipse_radius_layout.addWidget(self.major_axis_min_val)
        
        max_ellipse_radius_label = QLabel("Major Axis Maximum Value:")
        max_ellipse_radius_label.setObjectName("paramLabel")
        ellipse_radius_layout.addWidget(max_ellipse_radius_label)
        
        self.major_axis_max_val = QSpinBox()
        self.major_axis_max_val.setRange(0, 500)
        self.major_axis_max_val.setValue(10)
        self.major_axis_max_val.setObjectName("paramSpinBox")
        ellipse_radius_layout.addWidget(self.major_axis_max_val)
        hough_ellipse_layout.addLayout(ellipse_radius_layout)

        ellipse_minor_radius_layout = QHBoxLayout()
        min_ellipse_minor_radius_label = QLabel("Minor Axis Minimum Value:")
        min_ellipse_minor_radius_label.setObjectName("paramLabel")
        ellipse_minor_radius_layout.addWidget(min_ellipse_minor_radius_label)
        
        self.minor_axis_min_val = QSpinBox()
        self.minor_axis_min_val.setRange(0, 500)
        self.minor_axis_min_val.setValue(10)
        self.minor_axis_min_val.setObjectName("paramSpinBox")
        ellipse_minor_radius_layout.addWidget(self.minor_axis_min_val)
        
        max_ellipse_minor_radius_label = QLabel("Minor Axis Maximum Value:")
        max_ellipse_minor_radius_label.setObjectName("paramLabel")
        ellipse_minor_radius_layout.addWidget(max_ellipse_minor_radius_label)
        
        self.minor_axis_max_val = QSpinBox()
        self.minor_axis_max_val.setRange(0, 500)
        self.minor_axis_max_val.setValue(10)
        self.minor_axis_max_val.setObjectName("paramSpinBox")
        ellipse_minor_radius_layout.addWidget(self.minor_axis_max_val)
        hough_ellipse_layout.addLayout(ellipse_minor_radius_layout)
        
        # Threshold for ellipse detection - styled slider
        threshold_ellipse_layout = QHBoxLayout()
        threshold_ellipse_label = QLabel("Threshold:")
        threshold_ellipse_label.setObjectName("paramLabel")
        threshold_ellipse_layout.addWidget(threshold_ellipse_label)
        
        self.threshold_ellipse = QSlider(Qt.Orientation.Horizontal)
        self.threshold_ellipse.setRange(10, 300)
        self.threshold_ellipse.setValue(100)
        self.threshold_ellipse.setObjectName("paramSlider")
        
        self.threshold_ellipse_label = QLabel("100")
        self.threshold_ellipse_label.setObjectName("valueLabel")
        self.threshold_ellipse_label.setMinimumWidth(30)
        self.threshold_ellipse_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        
        threshold_ellipse_layout.addWidget(self.threshold_ellipse)
        threshold_ellipse_layout.addWidget(self.threshold_ellipse_label)
        hough_ellipse_layout.addLayout(threshold_ellipse_layout)
        
        # Detection buttons - styled buttons
        button_layout = QHBoxLayout()
    
        
        self.detect_ellipses_button = QPushButton("Detect Ellipses")
        self.detect_ellipses_button.setObjectName("actionButton")
        button_layout.addWidget(self.detect_ellipses_button)
        
        hough_ellipse_layout.addLayout(button_layout)
        
        hough_ellipse_group.setLayout(hough_ellipse_layout)
        hough_layout.addWidget(hough_ellipse_group)

        # Add a checkbox to toggle OpenCV usage
        self.use_opencv_checkbox = QCheckBox("Use OpenCV")
        self.use_opencv_checkbox.setObjectName("paramCheckbox")
        self.use_opencv_checkbox.setChecked(False)  # Default to not using OpenCV
        hough_layout.addWidget(self.use_opencv_checkbox)
        hough_layout.addStretch()
        
        # Add tabs to tab widget
        tab_widget.addTab(canny_tab, "Canny Edge Detection")
        tab_widget.addTab(hough_tab, "Hough Transform")
        
        main_layout.addWidget(tab_widget)
        
        # Connect signals
        self.low_threshold.valueChanged.connect(self._update_low_threshold_label)
        self.high_threshold.valueChanged.connect(self._update_high_threshold_label)
        self.threshold_lines.valueChanged.connect(self._update_threshold_lines_label)
        self.threshold_circles.valueChanged.connect(self._update_threshold_circles_label)
        self.threshold_ellipse.valueChanged.connect(self._update_threshold_ellipses_label)
        
        self.blur_kernel_size.valueChanged.connect(self._emit_canny_params)
        self.low_threshold.valueChanged.connect(self._emit_canny_params)
        self.high_threshold.valueChanged.connect(self._emit_canny_params)
        
        self.rho_resolution.valueChanged.connect(self._emit_hough_lines_params)
        self.theta_resolution.valueChanged.connect(self._emit_hough_lines_params)
        self.threshold_lines.valueChanged.connect(self._emit_hough_lines_params)
        
        self.dp_resolution.valueChanged.connect(self._emit_hough_circles_params)
        self.min_radius.valueChanged.connect(self._emit_hough_circles_params)
        self.max_radius.valueChanged.connect(self._emit_hough_circles_params)
        self.threshold_circles.valueChanged.connect(self._emit_hough_circles_params)

        self.dp_ellipse_resolution.valueChanged.connect(self._emit_hough_ellipses_params)
        self.major_axis_min_val.valueChanged.connect(self._emit_hough_ellipses_params)
        self.major_axis_max_val.valueChanged.connect(self._emit_hough_ellipses_params)
        self.minor_axis_min_val.valueChanged.connect(self._emit_hough_ellipses_params)
        self.minor_axis_max_val.valueChanged.connect(self._emit_hough_ellipses_params)
        self.threshold_ellipse.valueChanged.connect(self._emit_hough_ellipses_params)
        
        self.apply_canny_button.clicked.connect(self._on_apply_canny)
        self.detect_lines_button.clicked.connect(self._on_detect_lines)
        self.detect_circles_button.clicked.connect(self._on_detect_circles)
        self.detect_ellipses_button.clicked.connect(self._on_detect_ellipses)

        self.sidebar.addTab(edge_detection_widget, "Edge Detection")
    
    def _update_low_threshold_label(self, value):
        self.low_threshold_label.setText(str(value))
    
    def _update_high_threshold_label(self, value):
        self.high_threshold_label.setText(str(value))
    
    def _update_threshold_lines_label(self, value):
        self.threshold_lines_label.setText(str(value))
    
    def _update_threshold_circles_label(self, value):
        self.threshold_circles_label.setText(str(value))
    
    def _update_threshold_ellipses_label(self, value):
        self.threshold_ellipse_label.setText(str(value))
    
    def _emit_canny_params(self):
        self.canny_params_changed.emit(
            self.blur_kernel_size.value(),
            self.low_threshold.value(),
            self.high_threshold.value()
        )
    
    def _emit_hough_lines_params(self):
        self.hough_lines_params_changed.emit(
            self.rho_resolution.value(),
            self.theta_resolution.value(),
            self.threshold_lines.value()
        )
    
    def _emit_hough_circles_params(self):
        self.hough_circles_params_changed.emit(
            self.dp_resolution.value(),
            self.min_radius.value(),
            self.max_radius.value(),
            self.threshold_circles.value()
        )
            
    def _emit_hough_ellipses_params(self):
            self.hough_ellipses_params_changed.emit(
                self.dp_ellipse_resolution.value(),
                self.major_axis_min_val.value(),
                self.major_axis_max_val.value(),
                self.minor_axis_min_val.value(),
                self.minor_axis_max_val.value(),
                self.threshold_ellipse.value()
            )


    def on_filter_type_changed(self, button):
        selected_button = button.text()
        if selected_button == "Low Pass":
            self.freq_method_combo.setCurrentIndex(0)
        elif selected_button == "High Pass":
            self.freq_method_combo.setCurrentIndex(1)

    def on_combobox_changed(self, index):
        selected_text = self.freq_method_combo.currentText()
        if selected_text == "Gaussian":
            self.low_pass_radio.setChecked(True)
        elif selected_text == "Butterworth":
            self.high_pass_radio.setChecked(True)
    
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
                self.histogram_widget.tab_widget.setTabEnabled(0, False)
                self.histogram_widget.tab_widget.setTabEnabled(1, True)
                self.histogram_widget.tab_widget.setCurrentIndex(1)
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
            self.histogram_widget.set_image_data(self.current_image)

    def reset_to_original(self):
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.histogram_widget.tab_widget.setTabEnabled(0, False)
            self.histogram_widget.tab_widget.setTabEnabled(1, True)
            self.histogram_widget.tab_widget.setCurrentIndex(1)
            self.update_image_display()
            self.statusBar().showMessage("Reset to original image")

    def convert_to_grayscale(self):
        if self.current_image is not None:
            self.current_image = convert_to_grayscale(self.current_image)
            self.histogram_widget.tab_widget.setTabEnabled(1, False)
            self.histogram_widget.tab_widget.setTabEnabled(0, True)
            self.histogram_widget.tab_widget.setCurrentIndex(0)
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
            print("Normalized")
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
            localResult = local_threshold(self.current_image, block_size)
            if isinstance(localResult, str):
                self.statusBar().showMessage(f"{localResult}")
            else:
                self.current_image = localResult
                self.statusBar().showMessage(f"Applied Local thresholding (block size={block_size})")

        self.update_image_display()

    def apply_frequency_filter(self):
        if self.current_image is None:
            self.statusBar().showMessage("No image loaded")
            return
        
        filter_type = self.freq_filter_type.checkedId()
        cutoff = self.cutoff_slider.value()

        if filter_type == 0:  # Low Pass
            self.current_image = gaussian_low_pass_filter(self.current_image, cutoff)
            self.statusBar().showMessage("Applied Gaussian Low Pass filter")
        else:  # High Pass
            self.current_image = butterworth_high_pass_filter(self.current_image, cutoff)
            self.statusBar().showMessage("Applied Butterworth High Pass filter")

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
                image = cv2.resize(image, (300, 300))
                
                # Set the image in the dual view
                if image_number == 1:
                    self.first_image = image
                    self.dual_image_view.set_first_image(image)
                    self.show_status_message(f"Loaded first image: {file_name}", 3000)
                else:
                    self.second_image = image
                    self.dual_image_view.set_second_image(image)
                    self.show_status_message(f"Loaded second image: {file_name}", 3000)
                    
            except Exception as e:
                self.show_error_message(f"Error loading image: {str(e)}")

    def create_hybrid_from_dual_view(self, alpha):
        if self.first_image is None or self.second_image is None:
            self.show_error_message("Both images must be loaded to create a hybrid image")
            return
            
        # Create hybrid image
        self.current_image = create_hybrid_image(self.first_image, self.second_image, alpha)
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
                         [Mostafa Mousa - Rashed Mamdouh - Zyad Amr - Mostafa Ayman]</p>""")
        
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

    '''def _init_edge_detection(self):
        """Initialize edge detection panel and functionality"""
        # Create and set up the edge detection panel
        self.edge_detection_panel = EdgeDetectionPanel()
        
        # Create a dock widget for edge detection
        self.edge_dock = QDockWidget("Edge Detection & Hough Transform", self)
        self.edge_dock.setObjectName("edgeDockWidget")
        self.edge_dock.setWidget(self.edge_detection_panel)
        self.edge_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | 
                                 Qt.DockWidgetArea.RightDockWidgetArea)
        self.edge_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | 
                              QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        
        # Add dock widget to the main window
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.edge_dock)
        
        # Connect signals to slots
        self.edge_detection_panel.apply_canny_clicked.connect(self._on_apply_canny)
        self.edge_detection_panel.detect_lines_clicked.connect(self._on_detect_lines)
        self.edge_detection_panel.detect_circles_clicked.connect(self._on_detect_circles)
        self.edge_detection_panel.detect_ellipses_clicked.connect(self._on_detect_ellipses)
        
        # Add action to View menu if it exists
        if hasattr(self, 'view_menu'):
            self.view_menu.addAction(self.edge_dock.toggleViewAction())
    
    def _init_active_contours(self):
        """Initialize active contours panel and functionality"""
        # Create and set up the active contours panel
        self.active_contour_panel = ActiveContourPanel()
        
        # Create a dock widget for active contours
        self.contour_dock = QDockWidget("Active Contour Model (Snake)", self)
        self.contour_dock.setObjectName("contourDockWidget")
        self.contour_dock.setWidget(self.active_contour_panel)
        self.contour_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | 
                                   Qt.DockWidgetArea.RightDockWidgetArea)
        self.contour_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | 
                                QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        
        # Add dock widget to the main window, stacked below the edge detection panel
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.contour_dock)
        
        # Only tabify if edge_dock was created
        if self.edge_dock:
            self.tabifyDockWidget(self.edge_dock, self.contour_dock)  # Stack them as tabs
        
        # Connect signals to slots
        self.active_contour_panel.initialize_contour_clicked.connect(self._on_initialize_contour)
        self.active_contour_panel.evolve_contour_clicked.connect(self._on_evolve_contour)
        self.active_contour_panel.reset_contour_clicked.connect(self._on_reset_contour)
        self.active_contour_panel.calculate_metrics_clicked.connect(self._on_calculate_metrics)
        self.active_contour_panel.show_chain_code_clicked.connect(self._on_show_chain_code)
        
        # Add action to View menu if it exists
        if hasattr(self, 'view_menu'):
            self.view_menu.addAction(self.contour_dock.toggleViewAction())'''
    
    # Event handlers for edge detection
    def _on_apply_canny(self):
        self.statusBar().showMessage("Applying Canny edge detection...")
        # Add actual implementation here
    
    def _on_detect_lines(self):
        self.statusBar().showMessage("Detecting lines...")
        if self.original_image is None:
            return
        self.current_image = convert_to_grayscale(self.original_image)
        self.current_image = cv2.GaussianBlur(self.current_image, (5, 5), 1.5)
        self.current_image = cv2.Canny(self.current_image, 50, 150)  # Lower and upper threshold
    
        height, width = self.current_image.shape
        max_rho = int(np.sqrt((height ** 2) + (width ** 2)))

        # Define the size of our accumulator matrix
        rhos = np.arange(0, max_rho + 1, self.rho_resolution.value())
        thetas = np.deg2rad(np.arange(0, 180, self.theta_resolution.value()))
        accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.int32)

        if not self.use_opencv_checkbox.isChecked():            
            # Fill the accumulator matrix with votes
            edge_pixels = np.argwhere(self.current_image > 0)  # Get edge pixels efficiently
            rhos_len = len(rhos)
            for y, x in edge_pixels:
                for theta_idx, theta in enumerate(thetas):
                    rho = x * np.cos(theta) + y * np.sin(theta)
                    rho_index = int(np.round(rho / self.rho_resolution.value()))
                    if 0 <= rho_index < rhos_len:
                        accumulator[rho_index, theta_idx] += 1
            
            # Extract lines (r, theta) from the accumulator matrix
            detected_lines = []
            for rho_idx in range(accumulator.shape[0]):  # Loop over rho values
                for theta_idx in range(accumulator.shape[1]):  # Loop over theta values
                    if accumulator[rho_idx, theta_idx] >= self.threshold_lines.value():
                        rho = rhos[rho_idx]
                        theta = thetas[theta_idx]
                        detected_lines.append((rho, theta))
        else:
            # Apply OpenCV's Hough Line Transform
            lines = cv2.HoughLines(self.current_image, rho=rhos[1] - rhos[0], theta=thetas[1] - thetas[0], threshold=self.threshold_lines.value())
            detected_lines = []
            for line in lines:
                detected_lines.append(line[0])
        
        # Superimpose the lines
        output_image = self.original_image.copy()
        scale = max(width, height)         # Large scaling factor to extend the line across the image
        for rho, theta in detected_lines:
            # Compute two points far apart to draw the line
            x0 = rho * np.cos(theta)
            y0 = rho * np.sin(theta)

            # The normal vector to the line is (cos(θ), sin(θ)), (points directly from the origin to the line)
            # The direction of the actual line (which we want to draw) is perpendicular to this normal vector
            # The perpendicular direction can be found by rotating the normal vector by 90 degrees

            # Point in one direction
            x1 = int(x0 + scale * (-np.sin(theta)))
            y1 = int(y0 + scale * (np.cos(theta)))

            # Point in opposite direction
            x2 = int(x0 - scale * (-np.sin(theta)))
            y2 = int(y0 - scale * (np.cos(theta)))

            # Draw the line on the image (green color, thickness 2)
            cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        self.current_image = output_image

        self.update_image_display()
    
    def _on_detect_circles(self):
        self.statusBar().showMessage("Detecting circles...")
        if self.original_image is None:
            return
        self.current_image = convert_to_grayscale(self.original_image)
        self.current_image = cv2.GaussianBlur(self.current_image, (5, 5), 1.5)
        edges = cv2.Canny(self.current_image, 50, 100)
    
        # Define the size of our accumulator matrix
        height, width = self.current_image.shape
        minDistance = self.min_radius.value()
        accumulator = np.zeros((height, width, self.max_radius.value() - self.min_radius.value() + 1), dtype=np.int32)

        if not self.use_opencv_checkbox.isChecked():            
            # Fill the accumulator matrix with votes
            edge_pixels = np.argwhere(edges > 0)  # Get edge pixels efficiently
            for y, x in edge_pixels:
                for r in range(self.min_radius.value(), self.max_radius.value() + 1):
                    # To vote for potential centers, we consider all possible directions around the edge
                    for theta in range(0, 360, max(1, int(1 / self.dp_resolution.value()))):  # Step based on dp
                        # circle center
                        a = int(x - r * np.cos(np.deg2rad(theta)))
                        b = int(y - r * np.sin(np.deg2rad(theta)))
                        
                        if 0 <= a < height and 0 <= b < width:
                            accumulator[a, b, r - self.min_radius.value()] += 1 # r is minus min_radius as index starts from 0
            
            # Extract detected circles
            detected_circles = []            
            for a in range(height):
                for b in range(width):
                    for r_index, votes in enumerate(accumulator[a, b]):
                        if votes > self.threshold_circles.value():
                            r = r_index + self.min_radius.value()
                            detected_circles.append((a, b, r))
            
            # Min Distance Check (If circles are closer than the minimum distance allowed, we merge them together to be 1 circle)
            filtered_circles = []

            for new_circle in detected_circles:
                x1, y1, r1 = new_circle
                merged = False

                for i, (x2, y2, r2) in enumerate(filtered_circles):
                    distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    
                    if distance < minDistance:
                        # Merge by averaging centers & radii
                        merged_x = (x1 + x2) // 2
                        merged_y = (y1 + y2) // 2
                        merged_r = (r1 + r2) // 2

                        filtered_circles[i] = (merged_x, merged_y, merged_r)
                        merged = True
                        break  # Stop checking more circles once merged

                if not merged:
                    filtered_circles.append(new_circle)

            detected_circles = filtered_circles
        else:
            # Apply OpenCV's Hough Circle Transform
            circles = cv2.HoughCircles(self.current_image, cv2.HOUGH_GRADIENT, dp=self.dp_resolution.value(), minDist=minDistance, param1=100,
                                       param2=self.threshold_circles.value(), minRadius=self.min_radius.value(), maxRadius=self.max_radius.value())
            detected_circles = []
            if circles is not None:
                circles = np.uint16(np.around(circles))  # Round values for drawing
                for circle in circles[0, :]:
                    a, b, r = circle  # Extract center (a, b) and radius (r)
                    detected_circles.append((a, b, r))
                
        # Superimpose the lines
        output_image = self.original_image.copy()
        for x, y, r in detected_circles:
                cv2.circle(output_image, (x, y), r, (0, 255, 0), 2)  # Draw outer circle
                
        self.current_image = output_image

        self.update_image_display()
    
    def _on_detect_ellipses(self):
        self.statusBar().showMessage("Detecting ellipses...")
        if self.original_image is None:
            return
        self.current_image = convert_to_grayscale(self.original_image)
        self.current_image = cv2.GaussianBlur(self.current_image, (5, 5), 1.5)
        edges = cv2.Canny(self.current_image, 50, 150)

        # Define accumulator dimensions
        height, width = self.current_image.shape
        min_A = self.major_axis_min_val.value()
        max_A = self.major_axis_max_val.value()
        min_B = self.minor_axis_min_val.value()
        max_B = self.minor_axis_max_val.value()
        angle_step = self.dp_ellipse_resolution.value()

        # 5D accumulator (center_x, center_y, major axis A, minor axis B, angle)
        accumulator = np.zeros((height, width, max_A - min_A + 1, max_B - min_B + 1, len(range(0, 360, int(angle_step)))), dtype=np.int32)

        edge_pixels = np.argwhere(edges > 0)  # Get edge pixels efficiently
        for y, x in edge_pixels:
            for A in range(min_A, max_A + 1):
                for B in range(min_B, max_B + 1):
                    for theta_index, theta in enumerate(range(0, 360, int(angle_step))): # Rotation angle
                        rad = np.deg2rad(theta)
                        a = int(x - A * np.cos(rad))
                        b = int(y - B * np.sin(rad))

                        if 0 <= a < height and 0 <= b < width:
                            accumulator[a, b, A - min_A, B - min_B, theta_index] += 1

        # Extract detected ellipses
        detected_ellipses = []
        for a in range(height):
            for b in range(width):
                for A_index in range(max_A - min_A + 1):
                    for B_index in range(max_B - min_B + 1):
                        for theta_index, theta in enumerate(range(0, 360, int(angle_step))):
                            votes = accumulator[a, b, A_index, B_index, theta_index]
                            if votes > self.threshold_ellipse.value():
                                A = A_index + min_A
                                B = B_index + min_B
                                detected_ellipses.append((a, b, A, B, theta))

        # Draw detected ellipses
        output_image = self.original_image.copy()
        for x, y, A, B, theta in detected_ellipses:
            cv2.ellipse(output_image, (x, y), (A, B), theta, 0, 360, (0, 255, 0), 2)
        
        self.current_image = output_image

        self.update_image_display()
    
    # Event handlers for active contours
    def _on_initialize_contour(self):
        self.statusBar().showMessage("Click on image to place initial contour points...")
        # Add actual implementation here
    
    def _on_evolve_contour(self):
        self.statusBar().showMessage("Evolving contour...")
        # Add actual implementation here
    
    def _on_reset_contour(self):
        self.statusBar().showMessage("Contour reset")
        # Add actual implementation here
    
    def _on_calculate_metrics(self):
        self.statusBar().showMessage("Calculating perimeter and area...")
        # Example values - replace with actual calculation
        perimeter = 142.5
        area = 1256.3
        self.active_contour_panel.set_metric_values(perimeter, area)
    
    def _on_show_chain_code(self, show):
        if show:
            self.statusBar().showMessage("Showing chain code")
        else:
            self.statusBar().showMessage("Hiding chain code")
        # Add actual implementation here

   
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
