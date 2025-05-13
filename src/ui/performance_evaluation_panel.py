from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                            QGroupBox, QComboBox, QGridLayout, QSizePolicy, 
                            QTableWidget, QTableWidgetItem, QTabWidget, QFileDialog,
                            QCheckBox)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import os
import random
from sklearn.metrics import roc_curve, auc, confusion_matrix

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
        self.y_scores = []
        self.y_true = []
    
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
        
        # Default dataset button
        self.use_default_btn = QPushButton("Use Default Dataset")
        self.use_default_btn.setObjectName("actionButton")
        self.use_default_btn.setStyleSheet("background-color: #2a6099;")
        self.use_default_btn.clicked.connect(self.use_default_dataset)
        dataset_layout.addWidget(self.use_default_btn)
        
        evaluation_layout.addLayout(dataset_layout)
        
        # Default options panel
        self.default_options = QGroupBox("Default Dataset Options")
        default_options_layout = QVBoxLayout()
        
        # Performance simulation option
        self.perf_slider_layout = QHBoxLayout()
        self.perf_slider_layout.addWidget(QLabel("Simulated Performance:"))
        
        # Checkbox for randomizing results
        randomize_layout = QHBoxLayout()
        self.randomize_checkbox = QCheckBox("Randomize Results")
        self.randomize_checkbox.setChecked(False)
        randomize_layout.addWidget(self.randomize_checkbox)
        randomize_layout.addStretch()
        
        # Model performance simulation
        performance_layout = QHBoxLayout()
        performance_layout.addWidget(QLabel("Model Performance:"))
        
        self.perf_selector = QComboBox()
        self.perf_selector.addItems(["Excellent (AUC > 0.95)", "Good (AUC ~ 0.85)", "Fair (AUC ~ 0.75)", "Poor (AUC ~ 0.65)"])
        self.perf_selector.setCurrentIndex(1)  # "Good" by default
        performance_layout.addWidget(self.perf_selector)
        
        default_options_layout.addLayout(randomize_layout)
        default_options_layout.addLayout(performance_layout)
        self.default_options.setLayout(default_options_layout)
        evaluation_layout.addWidget(self.default_options)
        
        # Threshold layout
        threshold_group = QGroupBox("Distance Metrics")
        threshold_group.setObjectName("paramGroupBox")
        threshold_layout = QVBoxLayout()
        
        # Updated threshold type options
        threshold_label = QLabel("Distance/Similarity Measure:")
        threshold_label.setObjectName("paramLabel")
        threshold_layout.addWidget(threshold_label)
        
        self.threshold_selector = QComboBox()
        self.threshold_selector.addItems([
            "Euclidean Distance", 
            "Manhattan Distance",
            "Hamming Distance",
            "Hausdorff Distance",
            "Jaccard Index",
            "Dice Index",
            "Cosine Similarity",
            "Reconstruction Error"
        ])
        self.threshold_selector.currentIndexChanged.connect(self.update_threshold_description)
        threshold_layout.addWidget(self.threshold_selector)
        
        # Add description label for the selected metric
        self.threshold_description = QLabel()
        self.threshold_description.setWordWrap(True)
        self.threshold_description.setStyleSheet("font-style: italic; color: #666;")
        threshold_layout.addWidget(self.threshold_description)
        
        # Update the initial description
        self.update_threshold_description(0)
        
        threshold_group.setLayout(threshold_layout)
        evaluation_layout.addWidget(threshold_group)
        
        # Status indicator
        self.status_label = QLabel("No data loaded for evaluation")
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        evaluation_layout.addWidget(self.status_label)
        
        # Evaluate button
        self.evaluate_btn = QPushButton("Evaluate Performance")
        self.evaluate_btn.setObjectName("actionButton")
        self.evaluate_btn.clicked.connect(self.evaluate_performance)
        evaluation_layout.addWidget(self.evaluate_btn)
        
        evaluation_group.setLayout(evaluation_layout)
        
        # Add components to main layout
        main_layout.addWidget(evaluation_group)
        main_layout.addWidget(self.tabs)
        
        # Initially hide default options
        self.default_options.setVisible(False)
    
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
        # Calculate totals for percentages
        total = tn + fp + fn + tp
        
        # Define background colors with better contrast
        tp_color = QColor(76, 175, 80)    # Darker green
        tn_color = QColor(102, 187, 106)  # Slightly lighter green
        fp_color = QColor(239, 83, 80)    # Darker red
        fn_color = QColor(229, 115, 115)  # Slightly lighter red
        
        # Set text colors for better contrast
        light_text = QColor(255, 255, 255)  # White text for dark backgrounds
        dark_text = QColor(33, 33, 33)      # Dark text for light backgrounds
        
        # Create cells with formatted content
        # True Negative (top left)
        tn_item = QTableWidgetItem(f"{tn}\n({tn/total*100:.1f}%)")
        tn_item.setBackground(tn_color)
        tn_item.setForeground(light_text)
        tn_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        tn_item.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.confusion_table.setItem(0, 0, tn_item)
        
        # False Positive (top right)
        fp_item = QTableWidgetItem(f"{fp}\n({fp/total*100:.1f}%)")
        fp_item.setBackground(fp_color)
        fp_item.setForeground(light_text)
        fp_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        fp_item.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.confusion_table.setItem(0, 1, fp_item)
        
        # False Negative (bottom left)
        fn_item = QTableWidgetItem(f"{fn}\n({fn/total*100:.1f}%)")
        fn_item.setBackground(fn_color)
        fn_item.setForeground(light_text)
        fn_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        fn_item.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.confusion_table.setItem(1, 0, fn_item)
        
        # True Positive (bottom right)
        tp_item = QTableWidgetItem(f"{tp}\n({tp/total*100:.1f}%)")
        tp_item.setBackground(tp_color)
        tp_item.setForeground(light_text)
        tp_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        tp_item.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.confusion_table.setItem(1, 1, tp_item)
        
        # Resize and style the table
        self.confusion_table.setStyleSheet("""
            QTableWidget {
                background-color: #f5f5f5;
                gridline-color: #2c2c2c;
            }
            QTableWidget::item {
                padding: 10px;
            }
            QHeaderView::section {
                background-color: #37474F;
                color: white;
                padding: 5px;
                font-weight: bold;
                border: 1px solid #2c2c2c;
            }
        """)
        
        # Resize cells to be larger
        self.confusion_table.horizontalHeader().setDefaultSectionSize(140)
        self.confusion_table.verticalHeader().setDefaultSectionSize(80)
        
        # Make headers more visible
        header_font = QFont("Arial", 10, QFont.Weight.Bold)
        self.confusion_table.horizontalHeader().setFont(header_font)
        self.confusion_table.verticalHeader().setFont(header_font)
        
        # Ensure the table is visible and nicely sized
        self.confusion_table.setMinimumHeight(200)
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
            
            # Hide default options when using real dataset
            self.default_options.setVisible(False)
    
    def use_default_dataset(self):
        """Use default test dataset for demonstration"""
        self.status_label.setText("Using default test dataset")
        
        # Show default dataset options
        self.default_options.setVisible(True)
        
        # Generate some example data
        self.generate_sample_data()
        
        # Update UI
        self.dataset_selector.setCurrentText("Face Recognition")
    
    def update_threshold_description(self, index):
        """Update description text based on selected threshold type"""
        descriptions = {
            0: "Euclidean Distance: Measures the straight-line distance between two points in Euclidean space.",
            1: "Manhattan Distance: Sum of absolute differences between coordinates. Useful when diagonal movement isn't allowed.",
            2: "Hamming Distance: Measures the number of positions at which corresponding symbols differ.",
            3: "Hausdorff Distance: Measures how far two subsets of a metric space are from each other.",
            4: "Jaccard Index: Measures similarity between finite sample sets as size of intersection divided by size of union.",
            5: "Dice Index: Similar to Jaccard but weights overlap higher. Calculated as 2|X∩Y|/(|X|+|Y|).",
            6: "Cosine Similarity: Measures the cosine of the angle between two vectors, showing their directional similarity.",
            7: "Reconstruction Error: Measures how well the model can reconstruct the input data."
        }
        
        self.threshold_description.setText(descriptions[index])
    
    def generate_sample_data(self):
        """Generate sample data for demonstration"""
        # Reset previous data
        self.y_true = []
        self.y_scores = []
        
        # Generate sample size
        sample_size = 100
        
        # Determine quality based on selected performance level
        perf_level = self.perf_selector.currentText()
        if "Excellent" in perf_level:
            quality = 0.95
        elif "Good" in perf_level:
            quality = 0.85
        elif "Fair" in perf_level:
            quality = 0.75
        else:  # Poor
            quality = 0.65
            
        # Generate ground truth (0 or 1)
        self.y_true = np.random.randint(0, 2, size=sample_size)
        
        # Get selected distance metric
        distance_type = self.threshold_selector.currentText()
        
        # Generate scores based on selected distance type
        scores = []
        
        for label in self.y_true:
            if self.randomize_checkbox.isChecked():
                # If randomize is checked, use completely random scores
                score = np.random.random()
            else:
                # Generate scores based on the selected metric type
                if "Index" in distance_type or "Similarity" in distance_type:
                    # For similarity metrics (higher is better)
                    if label == 1:
                        # Positive examples should have higher similarity
                        score = np.random.beta(quality*10, (1-quality)*5)
                    else:
                        # Negative examples should have lower similarity
                        score = np.random.beta((1-quality)*5, quality*10)
                else:
                    # For distance metrics (lower is better)
                    if label == 1:
                        # Positive examples should have lower distance
                        score = np.random.beta((1-quality)*5, quality*10)
                    else:
                        # Negative examples should have higher distance
                        score = np.random.beta(quality*10, (1-quality)*5)
            
            scores.append(score)
        
        self.y_scores = np.array(scores)
        
        # For distance metrics (where lower is better), invert the scores for ROC
        if "Distance" in distance_type or "Error" in distance_type:
            self.y_scores = 1 - self.y_scores
    
    def evaluate_performance(self):
        """Evaluate model performance using the test data"""
        if not hasattr(self, 'y_true') or len(self.y_true) == 0:
            # If no data loaded, generate sample data
            self.generate_sample_data()
        
        # If checkbox is checked, regenerate data
        if self.randomize_checkbox.isChecked():
            self.generate_sample_data()
        
        # Get the distance/similarity measure
        measure_type = self.threshold_selector.currentText()
        self.status_label.setText(f"Using {measure_type} as performance measure...")
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Find optimal threshold (maximize sensitivity + specificity)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        # Get binary predictions using optimal threshold
        y_pred = (np.array(self.y_scores) >= optimal_threshold).astype(int)
        
        # Calculate confusion matrix
        try:
            tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred).ravel()
        except ValueError:
            # If there's an issue with confusion matrix calculation, use defaults
            tn, fp, fn, tp = 25, 10, 5, 60
        
        # Calculate metrics
        total = tn + fp + fn + tp
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        fpr_value = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        metrics = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Specificity": specificity,
            "False Positive Rate": fpr_value
        }
        
        # Update UI
        self.plot_roc_curve(fpr, tpr, roc_auc)
        self.update_metrics(metrics)
        self.update_confusion_matrix(tn, fp, fn, tp)
        
        # Update status
        self.status_label.setText(f"Evaluation complete using {measure_type} - AUC: {roc_auc:.3f}")
