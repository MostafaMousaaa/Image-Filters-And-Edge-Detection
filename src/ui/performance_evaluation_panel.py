from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                            QGroupBox, QComboBox, QGridLayout, QSizePolicy, 
                            QTableWidget, QTableWidgetItem, QTabWidget, QFileDialog)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

class PerformanceEvaluationPanel(QWidget):
    """Panel for performance evaluation and ROC curve plotting"""
    
    # Define signals
    dataset_loaded_signal = pyqtSignal(str)
    evaluate_performance_clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("performanceEvaluationPanel")
        self.initUI()
        
        # Store test results
        self.true_positives = []
        self.true_negatives = []
        self.false_positives = []
        self.false_negatives = []
        self.thresholds = []
        self.predictions = []
        self.ground_truth = []
    
    def initUI(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)
        
        # Create tabs for ROC curve and confusion matrix
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        
        # ROC Curve Tab
        self.roc_tab = QWidget()
        self.setup_roc_tab()
        self.tabs.addTab(self.roc_tab, "ROC Curve")
        
        # Metrics Tab 
        self.metrics_tab = QWidget()
        self.setup_metrics_tab()
        self.tabs.addTab(self.metrics_tab, "Performance Metrics")
        
        # Dataset Selection and Evaluation Group
        evaluation_group = QGroupBox("Performance Evaluation")
        evaluation_group.setObjectName("paramGroupBox")
        evaluation_layout = QVBoxLayout()
        
        # Dataset selector
        dataset_layout = QHBoxLayout()
        dataset_label = QLabel("Test Dataset:")
        dataset_label.setObjectName("paramLabel")
        dataset_layout.addWidget(dataset_label)
        
        self.dataset_selector = QComboBox()
        self.dataset_selector.addItems(["-- Select Dataset --", "Face Recognition", "Object Detection", "Custom Dataset"])
        dataset_layout.addWidget(self.dataset_selector)
        
        self.load_dataset_btn = QPushButton("Load Dataset")
        self.load_dataset_btn.setObjectName("actionButton")
        self.load_dataset_btn.clicked.connect(self.load_dataset)
        dataset_layout.addWidget(self.load_dataset_btn)
        
        evaluation_layout.addLayout(dataset_layout)
        
        # Threshold layout
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("Threshold Type:")
        threshold_label.setObjectName("paramLabel")
        threshold_layout.addWidget(threshold_label)
        
        self.threshold_selector = QComboBox()
        self.threshold_selector.addItems(["Similarity Score", "Distance Metric", "Reconstruction Error"])
        threshold_layout.addWidget(self.threshold_selector)
        
        evaluation_layout.addLayout(threshold_layout)
        
        # Status indicator
        self.status_label = QLabel("No data loaded for evaluation")
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        evaluation_layout.addWidget(self.status_label)
        
        # Evaluate button
        self.evaluate_btn = QPushButton("Evaluate Performance")
        self.evaluate_btn.setObjectName("actionButton")
        self.evaluate_btn.clicked.connect(self.evaluate_performance_clicked)
        evaluation_layout.addWidget(self.evaluate_btn)
        
        evaluation_group.setLayout(evaluation_layout)
        
        # Add components to main layout
        main_layout.addWidget(evaluation_group)
        main_layout.addWidget(self.tabs)
    
    def setup_roc_tab(self):
        layout = QVBoxLayout(self.roc_tab)
        
        # ROC Curve plot
        self.roc_figure = Figure(figsize=(5, 4), dpi=100)
        self.roc_canvas = FigureCanvas(self.roc_figure)
        self.roc_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Info box for AUC
        auc_box = QGroupBox("Area Under Curve (AUC)")
        auc_box.setObjectName("resultsFrame")
        auc_layout = QHBoxLayout()
        
        auc_label = QLabel("AUC:")
        auc_label.setObjectName("metricLabel")
        auc_layout.addWidget(auc_label)
        
        self.auc_value = QLabel("--")
        self.auc_value.setObjectName("metricValue")
        self.auc_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.auc_value.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        auc_layout.addWidget(self.auc_value)
        
        auc_box.setLayout(auc_layout)
        
        # Controls for saving plot
        controls_layout = QHBoxLayout()
        self.save_plot_btn = QPushButton("Save ROC Plot")
        self.save_plot_btn.setObjectName("actionButton")
        self.save_plot_btn.clicked.connect(self.save_roc_plot)
        controls_layout.addWidget(self.save_plot_btn)
        controls_layout.addStretch()
        
        # Add to layout
        layout.addWidget(self.roc_canvas)
        layout.addWidget(auc_box)
        layout.addLayout(controls_layout)
        
        # Initially draw an empty plot
        self.init_roc_plot()
    
    def setup_metrics_tab(self):
        layout = QVBoxLayout(self.metrics_tab)
        
        # Metrics table
        metrics_group = QGroupBox("Classification Metrics")
        metrics_group.setObjectName("paramGroupBox")
        metrics_layout = QGridLayout()
        
        # Create metrics grid
        metrics = [
            ("Accuracy", "--"),
            ("Precision", "--"),
            ("Recall", "--"),
            ("F1 Score", "--"),
            ("Specificity", "--"),
            ("False Positive Rate", "--")
        ]
        
        self.metric_labels = {}
        
        for row, (metric_name, value) in enumerate(metrics):
            name_label = QLabel(f"{metric_name}:")
            name_label.setObjectName("metricLabel")
            metrics_layout.addWidget(name_label, row, 0)
            
            value_label = QLabel(value)
            value_label.setObjectName("metricValue")
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            metrics_layout.addWidget(value_label, row, 1)
            
            self.metric_labels[metric_name] = value_label
        
        metrics_group.setLayout(metrics_layout)
        
        # Confusion matrix
        confusion_group = QGroupBox("Confusion Matrix")
        confusion_group.setObjectName("paramGroupBox")
        confusion_layout = QVBoxLayout()
        
        # Create confusion matrix table
        self.confusion_table = QTableWidget(2, 2)
        self.confusion_table.setMinimumHeight(120)
        self.confusion_table.setHorizontalHeaderLabels(["Predicted Negative", "Predicted Positive"])
        self.confusion_table.setVerticalHeaderLabels(["Actual Negative", "Actual Positive"])
        
        # Initialize with empty values
        for i in range(2):
            for j in range(2):
                item = QTableWidgetItem("--")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.confusion_table.setItem(i, j, item)
        
        self.confusion_table.resizeColumnsToContents()
        self.confusion_table.resizeRowsToContents()
        
        confusion_layout.addWidget(self.confusion_table)
        confusion_group.setLayout(confusion_layout)
        
        # Add to layout
        layout.addWidget(metrics_group)
        layout.addWidget(confusion_group)
        layout.addStretch()
    
    def init_roc_plot(self):
        """Initialize an empty ROC plot"""
        self.roc_figure.clear()
        ax = self.roc_figure.add_subplot(111)
        ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.grid(True, alpha=0.3)
        self.roc_figure.tight_layout()
        self.roc_canvas.draw()
    
    def plot_roc_curve(self, fpr, tpr, auc):
        """Plot the ROC curve with the given data"""
        self.roc_figure.clear()
        ax = self.roc_figure.add_subplot(111)
        
        # Plot the ROC curve
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'Model (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5)')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC)')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        
        self.roc_figure.tight_layout()
        self.roc_canvas.draw()
        
        # Update AUC value
        self.auc_value.setText(f"{auc:.3f}")
    
    def update_metrics(self, metrics_dict):
        """Update the metrics display with calculated values"""
        for metric_name, value in metrics_dict.items():
            if metric_name in self.metric_labels:
                self.metric_labels[metric_name].setText(f"{value:.3f}")
    
    def update_confusion_matrix(self, tn, fp, fn, tp):
        """Update the confusion matrix table with values"""
        # True Negative (top left)
        self.confusion_table.item(0, 0).setText(str(tn))
        
        # False Positive (top right)
        self.confusion_table.item(0, 1).setText(str(fp))
        
        # False Negative (bottom left)
        self.confusion_table.item(1, 0).setText(str(fn))
        
        # True Positive (bottom right)
        self.confusion_table.item(1, 1).setText(str(tp))
        
        # Resize table to fit content
        self.confusion_table.resizeColumnsToContents()
        self.confusion_table.resizeRowsToContents()
    
    def save_roc_plot(self):
        """Save the ROC plot to a file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save ROC Plot", "", "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)"
        )
        
        if file_path:
            self.roc_figure.savefig(file_path, dpi=300, bbox_inches='tight')
            self.status_label.setText(f"Plot saved to {file_path}")
    
    def load_dataset(self):
        """Open dialog to select dataset folder for evaluation"""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Test Dataset Folder")
        
        if folder_path:
            self.status_label.setText(f"Dataset loaded from {folder_path}")
            self.dataset_loaded_signal.emit(folder_path)
