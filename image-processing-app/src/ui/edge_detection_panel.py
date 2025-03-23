from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                            QSlider, QComboBox, QGroupBox, QCheckBox, QSpinBox, 
                            QDoubleSpinBox, QTabWidget, QRadioButton, QButtonGroup, QFrame)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor

class EdgeDetectionPanel(QWidget):
    """Panel for edge detection and shape recognition controls"""
    
    # Define signals
    canny_params_changed = pyqtSignal(int, int, int)
    hough_lines_params_changed = pyqtSignal(float, float, int)
    hough_circles_params_changed = pyqtSignal(float, int, int, int, int)
    apply_canny_clicked = pyqtSignal()
    detect_lines_clicked = pyqtSignal()
    detect_circles_clicked = pyqtSignal()
    detect_ellipses_clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("edgeDetectionPanel")
        self.initUI()
    
    def initUI(self):
        main_layout = QVBoxLayout(self)
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
        
        self.rho_resolution = QDoubleSpinBox()
        self.rho_resolution.setRange(0.1, 10.0)
        self.rho_resolution.setSingleStep(0.1)
        self.rho_resolution.setValue(1.0)
        self.rho_resolution.setObjectName("paramSpinBox")
        rho_layout.addWidget(self.rho_resolution)
        hough_lines_layout.addLayout(rho_layout)
        
        # Theta resolution
        theta_layout = QHBoxLayout()
        theta_label = QLabel("Theta resolution (radians):")
        theta_label.setObjectName("paramLabel")
        theta_layout.addWidget(theta_label)
        
        self.theta_resolution = QDoubleSpinBox()
        self.theta_resolution.setRange(0.001, 0.1)
        self.theta_resolution.setSingleStep(0.001)
        self.theta_resolution.setValue(0.01)
        self.theta_resolution.setObjectName("paramSpinBox")
        theta_layout.addWidget(self.theta_resolution)
        hough_lines_layout.addLayout(theta_layout)
        
        # Threshold for lines - styled slider
        threshold_lines_layout = QHBoxLayout()
        threshold_lines_label = QLabel("Threshold:")
        threshold_lines_label.setObjectName("paramLabel")
        threshold_lines_layout.addWidget(threshold_lines_label)
        
        self.threshold_lines = QSlider(Qt.Orientation.Horizontal)
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
        dp_label = QLabel("Accumulator resolution:")
        dp_label.setObjectName("paramLabel")
        dp_layout.addWidget(dp_label)
        
        self.dp_resolution = QDoubleSpinBox()
        self.dp_resolution.setRange(0.5, 5.0)
        self.dp_resolution.setSingleStep(0.1)
        self.dp_resolution.setValue(1.0)
        self.dp_resolution.setObjectName("paramSpinBox")
        dp_layout.addWidget(self.dp_resolution)
        hough_circles_layout.addLayout(dp_layout)
        
        # Min distance between circles
        min_dist_layout = QHBoxLayout()
        min_dist_label = QLabel("Min distance between centers:")
        min_dist_label.setObjectName("paramLabel")
        min_dist_layout.addWidget(min_dist_label)
        
        self.min_distance = QSpinBox()
        self.min_distance.setRange(1, 200)
        self.min_distance.setValue(20)
        self.min_distance.setObjectName("paramSpinBox")
        min_dist_layout.addWidget(self.min_distance)
        hough_circles_layout.addLayout(min_dist_layout)
        
        # Min and max radius
        radius_layout = QHBoxLayout()
        min_radius_label = QLabel("Min radius:")
        min_radius_label.setObjectName("paramLabel")
        radius_layout.addWidget(min_radius_label)
        
        self.min_radius = QSpinBox()
        self.min_radius.setRange(0, 200)
        self.min_radius.setValue(10)
        self.min_radius.setObjectName("paramSpinBox")
        radius_layout.addWidget(self.min_radius)
        
        max_radius_label = QLabel("Max radius:")
        max_radius_label.setObjectName("paramLabel")
        radius_layout.addWidget(max_radius_label)
        
        self.max_radius = QSpinBox()
        self.max_radius.setRange(0, 500)
        self.max_radius.setValue(100)
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
        self.threshold_circles.setValue(80)
        self.threshold_circles.setObjectName("paramSlider")
        
        self.threshold_circles_label = QLabel("80")
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
        
        self.detect_ellipses_button = QPushButton("Detect Ellipses")
        self.detect_ellipses_button.setObjectName("actionButton")
        button_layout.addWidget(self.detect_ellipses_button)
        
        hough_circles_layout.addLayout(button_layout)
        
        hough_circles_group.setLayout(hough_circles_layout)
        hough_layout.addWidget(hough_circles_group)
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
        
        self.blur_kernel_size.valueChanged.connect(self._emit_canny_params)
        self.low_threshold.valueChanged.connect(self._emit_canny_params)
        self.high_threshold.valueChanged.connect(self._emit_canny_params)
        
        self.rho_resolution.valueChanged.connect(self._emit_hough_lines_params)
        self.theta_resolution.valueChanged.connect(self._emit_hough_lines_params)
        self.threshold_lines.valueChanged.connect(self._emit_hough_lines_params)
        
        self.dp_resolution.valueChanged.connect(self._emit_hough_circles_params)
        self.min_distance.valueChanged.connect(self._emit_hough_circles_params)
        self.min_radius.valueChanged.connect(self._emit_hough_circles_params)
        self.max_radius.valueChanged.connect(self._emit_hough_circles_params)
        self.threshold_circles.valueChanged.connect(self._emit_hough_circles_params)
        
        self.apply_canny_button.clicked.connect(self.apply_canny_clicked)
        self.detect_lines_button.clicked.connect(self.detect_lines_clicked)
        self.detect_circles_button.clicked.connect(self.detect_circles_clicked)
        self.detect_ellipses_button.clicked.connect(self.detect_ellipses_clicked)
    
    def _update_low_threshold_label(self, value):
        self.low_threshold_label.setText(str(value))
    
    def _update_high_threshold_label(self, value):
        self.high_threshold_label.setText(str(value))
    
    def _update_threshold_lines_label(self, value):
        self.threshold_lines_label.setText(str(value))
    
    def _update_threshold_circles_label(self, value):
        self.threshold_circles_label.setText(str(value))
    
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
            self.min_distance.value(),
            self.min_radius.value(),
            self.max_radius.value(),
            self.threshold_circles.value()
        )
