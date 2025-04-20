from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                            QSlider, QGroupBox, QSpinBox, QDoubleSpinBox, QCheckBox,
                            QFrame, QSizePolicy)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

class ActiveContourPanel(QWidget):
    """Panel for active contour model (snake) controls"""
    
    # Define signals
    snake_params_changed = pyqtSignal(float, float, float, float, int)
    initialize_contour_clicked = pyqtSignal()
    evolve_contour_clicked = pyqtSignal()
    reset_contour_clicked = pyqtSignal()
    calculate_metrics_clicked = pyqtSignal()
    show_chain_code_clicked = pyqtSignal(bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("activeContourPanel")
        self.initUI()
    
    def initUI(self):
        main_layout = QVBoxLayout(self)
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
        
        self.initialize_button.clicked.connect(self.initialize_contour_clicked)
        self.evolve_button.clicked.connect(self.evolve_contour_clicked)
        self.reset_button.clicked.connect(self.reset_contour_clicked)
        self.calculate_metrics_button.clicked.connect(self.calculate_metrics_clicked)
        self.show_chain_code_checkbox.toggled.connect(self.show_chain_code_clicked)
    
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
