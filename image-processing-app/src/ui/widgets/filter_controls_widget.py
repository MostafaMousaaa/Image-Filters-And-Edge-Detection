from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
                           QSlider, QPushButton, QSpinBox, QDoubleSpinBox, QGroupBox)
from PyQt6.QtCore import Qt, pyqtSignal

class FilterControlsWidget(QWidget):
    """Widget for controlling image filter parameters"""
    
    # Define signals
    filter_changed = pyqtSignal(str, object)  # Signal emitted when a filter is applied (filter_type, parameters)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        
        # Create filter type selector group
        filter_group = QGroupBox("Filter Type")
        filter_layout = QVBoxLayout()
        
        # Filter type combo box
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["Average", "Gaussian", "Median", "Bilateral", "Unsharp Mask"])
        self.filter_combo.currentTextChanged.connect(self.on_filter_type_changed)
        filter_layout.addWidget(self.filter_combo)
        
        filter_group.setLayout(filter_layout)
        main_layout.addWidget(filter_group)
        
        # Create parameters group
        params_group = QGroupBox("Parameters")
        self.params_layout = QVBoxLayout(params_group)
        
        # Kernel size control
        kernel_layout = QHBoxLayout()
        kernel_label = QLabel("Kernel Size:")
        self.kernel_size = QSpinBox()
        self.kernel_size.setRange(3, 15)
        self.kernel_size.setSingleStep(2)  # To ensure odd numbers
        self.kernel_size.setValue(3)
        kernel_layout.addWidget(kernel_label)
        kernel_layout.addWidget(self.kernel_size)
        self.params_layout.addLayout(kernel_layout)
        
        # Sigma control for Gaussian
        sigma_layout = QHBoxLayout()
        sigma_label = QLabel("Sigma:")
        self.sigma = QDoubleSpinBox()
        self.sigma.setRange(0.1, 5.0)
        self.sigma.setSingleStep(0.1)
        self.sigma.setValue(1.0)
        sigma_layout.addWidget(sigma_label)
        sigma_layout.addWidget(self.sigma)
        self.params_layout.addLayout(sigma_layout)
        
        # Color sigma for Bilateral
        color_sigma_layout = QHBoxLayout()
        color_sigma_label = QLabel("Color Sigma:")
        self.color_sigma = QDoubleSpinBox()
        self.color_sigma.setRange(1.0, 150.0)
        self.color_sigma.setSingleStep(5.0)
        self.color_sigma.setValue(75.0)
        color_sigma_layout.addWidget(color_sigma_label)
        color_sigma_layout.addWidget(self.color_sigma)
        self.params_layout.addLayout(color_sigma_layout)
        
        # Unsharp mask amount
        amount_layout = QHBoxLayout()
        amount_label = QLabel("Amount:")
        self.amount = QDoubleSpinBox()
        self.amount.setRange(0.1, 5.0)
        self.amount.setSingleStep(0.1)
        self.amount.setValue(1.5)
        amount_layout.addWidget(amount_label)
        amount_layout.addWidget(self.amount)
        self.params_layout.addLayout(amount_layout)
        
        # Add parameters group to main layout
        main_layout.addWidget(params_group)
        
        # Apply button
        self.apply_button = QPushButton("Apply Filter")
        self.apply_button.clicked.connect(self.apply_filter)
        main_layout.addWidget(self.apply_button)
        
        # Stretch to push everything to the top
        main_layout.addStretch()
        
        # Show/hide appropriate controls for initial filter type
        self.on_filter_type_changed(self.filter_combo.currentText())
    
    def on_filter_type_changed(self, filter_type):
        """Show/hide appropriate controls based on the selected filter type"""
        # Show kernel size for all filters
        self.kernel_size.setVisible(True)
        self.params_layout.itemAt(0).layout().itemAt(0).widget().setVisible(True)  # kernel label
        
        # Hide all specialized controls first
        self.sigma.setVisible(False)
        self.params_layout.itemAt(1).layout().itemAt(0).widget().setVisible(False)  # sigma label
        
        self.color_sigma.setVisible(False)
        self.params_layout.itemAt(2).layout().itemAt(0).widget().setVisible(False)  # color sigma label
        
        self.amount.setVisible(False)
        self.params_layout.itemAt(3).layout().itemAt(0).widget().setVisible(False)  # amount label
        
        # Show specialized controls based on filter type
        if filter_type == "Gaussian":
            self.sigma.setVisible(True)
            self.params_layout.itemAt(1).layout().itemAt(0).widget().setVisible(True)
        
        elif filter_type == "Bilateral":
            self.sigma.setVisible(True)
            self.params_layout.itemAt(1).layout().itemAt(0).widget().setVisible(True)
            self.color_sigma.setVisible(True)
            self.params_layout.itemAt(2).layout().itemAt(0).widget().setVisible(True)
        
        elif filter_type == "Unsharp Mask":
            self.sigma.setVisible(True)
            self.params_layout.itemAt(1).layout().itemAt(0).widget().setVisible(True)
            self.amount.setVisible(True)
            self.params_layout.itemAt(3).layout().itemAt(0).widget().setVisible(True)
    
    def apply_filter(self):
        """Apply the selected filter with the current parameters"""
        filter_type = self.filter_combo.currentText()
        parameters = {
            'kernel_size': self.kernel_size.value()
        }
        
        if filter_type == "Gaussian":
            parameters['sigma'] = self.sigma.value()
        
        elif filter_type == "Bilateral":
            parameters['sigma'] = self.sigma.value()
            parameters['color_sigma'] = self.color_sigma.value()
        
        elif filter_type == "Unsharp Mask":
            parameters['sigma'] = self.sigma.value()
            parameters['amount'] = self.amount.value()
        
        # Emit signal with filter type and parameters
        self.filter_changed.emit(filter_type, parameters)