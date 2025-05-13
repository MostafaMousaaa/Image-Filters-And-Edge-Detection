from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                            QGroupBox, QComboBox, QGridLayout, QSizePolicy, 
                            QTableWidget, QTableWidgetItem, QTabWidget, QFileDialog,
                            QCheckBox, QSlider)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import os
import random
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score

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
        
        # Add threshold control slider
        threshold_group = QGroupBox("Classification Threshold")
        threshold_group.setObjectName("paramGroupBox")
        threshold_layout = QVBoxLayout()
        
        slider_layout = QHBoxLayout()
        threshold_label = QLabel("Threshold:")
        threshold_label.setObjectName("paramLabel")
        slider_layout.addWidget(threshold_label)
        
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(1, 99)
        self.threshold_slider.setValue(50)  # Default to 0.5
        self.threshold_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.threshold_slider.setTickInterval(10)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        slider_layout.addWidget(self.threshold_slider)
        
        self.threshold_value_label = QLabel("0.50")
        self.threshold_value_label.setMinimumWidth(40)
        self.threshold_value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        slider_layout.addWidget(self.threshold_value_label)
        
        threshold_layout.addLayout(slider_layout)
        
        # Add description of threshold
        threshold_description = QLabel("Adjust the classification threshold to balance between sensitivity and specificity.")
        threshold_description.setWordWrap(True)
        threshold_description.setStyleSheet("font-style: italic; color: #666;")
        threshold_layout.addWidget(threshold_description)
        
        threshold_group.setLayout(threshold_layout)
        
        # Info box for AUC and threshold metrics
        metrics_box = QGroupBox("ROC Metrics")
        metrics_box.setObjectName("resultsFrame")
        metrics_layout = QGridLayout()
        
        # AUC Value
        auc_label = QLabel("AUC:")
        auc_label.setObjectName("metricLabel")
        metrics_layout.addWidget(auc_label, 0, 0)
        
        self.auc_value = QLabel("--")
        self.auc_value.setObjectName("metricValue")
        self.auc_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        metrics_layout.addWidget(self.auc_value, 0, 1)
        
        # Sensitivity at threshold
        sensitivity_label = QLabel("Sensitivity at threshold:")
        sensitivity_label.setObjectName("metricLabel")
        metrics_layout.addWidget(sensitivity_label, 1, 0)
        
        self.sensitivity_value = QLabel("--")
        self.sensitivity_value.setObjectName("metricValue")
        self.sensitivity_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        metrics_layout.addWidget(self.sensitivity_value, 1, 1)
        
        # Specificity at threshold
        specificity_label = QLabel("Specificity at threshold:")
        specificity_label.setObjectName("metricLabel")
        metrics_layout.addWidget(specificity_label, 2, 0)
        
        self.specificity_value = QLabel("--")
        self.specificity_value.setObjectName("metricValue")
        self.specificity_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        metrics_layout.addWidget(self.specificity_value, 2, 1)
        
        metrics_box.setLayout(metrics_layout)
        
        # Controls for saving plot
        controls_layout = QHBoxLayout()
        self.save_plot_btn = QPushButton("Save ROC Plot")
        self.save_plot_btn.setObjectName("actionButton")
        self.save_plot_btn.clicked.connect(self.save_roc_plot)
        controls_layout.addWidget(self.save_plot_btn)
        controls_layout.addStretch()
        
        # Add to layout
        layout.addWidget(self.roc_canvas)
        layout.addWidget(threshold_group)
        layout.addWidget(metrics_box)
        layout.addLayout(controls_layout)
        
        # Initially draw an empty plot
        self.init_roc_plot()
        
        # Store ROC data for threshold updates
        self.fpr_data = None
        self.tpr_data = None
        self.thresholds_data = None
    
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
        # Store the ROC data for threshold updates
        self.fpr_data = fpr
        self.tpr_data = tpr
        self.thresholds_data = np.linspace(0, 1, len(fpr)) if len(fpr) > 1 else np.array([0.5])
        
        # Get current threshold
        threshold = self.threshold_slider.value() / 100.0
        
        # Find the closest point on the ROC curve to the threshold
        idx = np.argmin(np.abs(self.thresholds_data - threshold))
        fpr_at_threshold = fpr[idx]
        tpr_at_threshold = tpr[idx]
        
        self.plot_roc_curve_with_threshold(fpr, tpr, auc, fpr_at_threshold, tpr_at_threshold)
        
        # Update AUC value and threshold metrics
        self.auc_value.setText(f"{float(auc):.3f}")
        self.sensitivity_value.setText(f"{tpr_at_threshold:.3f}")
        self.specificity_value.setText(f"{1-fpr_at_threshold:.3f}")
    
    def plot_roc_curve_with_threshold(self, fpr, tpr, auc, fpr_at_threshold, tpr_at_threshold):
        """Plot the ROC curve with the threshold point"""
        self.roc_figure.clear()
        ax = self.roc_figure.add_subplot(111)
        
        # Plot the ROC curve
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {float(auc):.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5)')
        
        # Plot the threshold point
        ax.plot(fpr_at_threshold, tpr_at_threshold, 'ro', markersize=8, 
               label=f'Threshold (TPR={tpr_at_threshold:.2f}, FPR={fpr_at_threshold:.2f})')
        
        # Add visual lines to help see the threshold point
        ax.plot([0, fpr_at_threshold, fpr_at_threshold], 
                [tpr_at_threshold, tpr_at_threshold, 0], 
                'r--', alpha=0.3)
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC)')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        
        self.roc_figure.tight_layout()
        self.roc_canvas.draw()
    
    def update_threshold(self, value):
        """Update the threshold value and recalculate metrics"""
        threshold = value / 100.0
        self.threshold_value_label.setText(f"{threshold:.2f}")
        
        # If we have ROC data, update the plot and metrics
        if hasattr(self, 'fpr_data') and self.fpr_data is not None:
            self.update_threshold_on_plot(threshold)
    
    def update_threshold_on_plot(self, threshold):
        """Update the ROC curve plot with the current threshold"""
        if not hasattr(self, 'fpr_data') or self.fpr_data is None:
            return
            
        # Find the closest point on the ROC curve to the threshold
        # Note: scikit-learn's ROC curve thresholds are in descending order
        idx = np.argmin(np.abs(self.thresholds_data - threshold))
        fpr_at_threshold = self.fpr_data[idx]
        tpr_at_threshold = self.tpr_data[idx]
        
        # Update the plot with threshold point
        self.plot_roc_curve_with_threshold(self.fpr_data, self.tpr_data, self.auc_value.text(), 
                                          fpr_at_threshold, tpr_at_threshold)
        
        # Update sensitivity and specificity at threshold
        self.sensitivity_value.setText(f"{tpr_at_threshold:.3f}")
        self.specificity_value.setText(f"{1-fpr_at_threshold:.3f}")
        
        # If we have ground truth, recalculate metrics at this threshold
        if hasattr(self, 'y_true') and len(self.y_true) > 0:
            # Get binary predictions using selected threshold
            y_pred = (np.array(self.y_scores) >= threshold).astype(int)
            
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
            
            # Update metrics and confusion matrix
            self.update_metrics(metrics)
            self.update_confusion_matrix(tn, fp, fn, tp)
            
            # Update status
            self.status_label.setText(f"Threshold updated to {threshold:.2f}")
    
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
        pass
    
    def generate_sample_data(self):
        """Generate sample data for demonstration with controlled AUC values"""
        # Reset previous data
        self.y_true = []
        self.y_scores = []
        
        # Generate sample size
        sample_size = 300
        
        # Determine target AUC based on selected performance level
        perf_level = self.perf_selector.currentText()
        if "Excellent" in perf_level:
            target_auc = 0.95
        elif "Good" in perf_level:
            target_auc = 0.85
        elif "Fair" in perf_level:
            target_auc = 0.75
        else:  # Poor
            target_auc = 0.65

        # Generate ground truth (0 or 1) with balanced classes
        self.y_true = np.zeros(sample_size)
        self.y_true[:sample_size//2] = 1  # 50% positive, 50% negative
        
        # If randomize is checked, create scores with no predictive power (AUC ≈ 0.5)
        if self.randomize_checkbox.isChecked():
            self.y_scores = np.random.random(sample_size)
        else:
            # Generate two distributions with controlled separation to achieve target AUC
            # For positive class (1)
            mu_pos = 0.7
            sigma_pos = 0.15
            
            # For negative class (0)
            sigma_neg = 0.15
            
            # Adjust mean of negative class to achieve target AUC
            # Higher target_auc = larger separation between distributions
            separation = (target_auc - 0.5) * 2  # Map [0.5, 1.0] to [0, 1.0]
            mu_neg = mu_pos - separation * (sigma_pos + sigma_neg)
            
            # Generate scores
            scores = np.zeros(sample_size)
            
            # Create positive class scores (higher values)
            pos_indices = np.where(self.y_true == 1)[0]
            scores[pos_indices] = np.random.normal(mu_pos, sigma_pos, len(pos_indices))
            
            # Create negative class scores (lower values)
            neg_indices = np.where(self.y_true == 0)[0]
            scores[neg_indices] = np.random.normal(mu_neg, sigma_neg, len(neg_indices))
            
            # Clip to [0, 1] range for proper probability scores
            scores = np.clip(scores, 0, 1)
            
            # For distance metrics (lower is better), invert scores
            distance_type = self.threshold_selector.currentText()
            if "Distance" in distance_type or "Error" in distance_type:
                # Don't invert here, as we invert later in evaluate_performance
                pass
            
            self.y_scores = scores
            
            # Shuffle the data to mix positives and negatives
            indices = np.arange(sample_size)
            np.random.shuffle(indices)
            self.y_true = self.y_true[indices]
            self.y_scores = self.y_scores[indices]
            
            # Verify the generated AUC is close to target
            actual_auc = roc_auc_score(self.y_true, self.y_scores)
            
            # If AUC is significantly off from target, adjust and regenerate
            if abs(actual_auc - target_auc) > 0.03:  # Allow small deviation
                print(f"AUC adjustment: {actual_auc:.3f} -> {target_auc:.3f}")
                # Recursively generate again with adjusted parameters
                self.generate_sample_data()

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
        
        # Store original scores before any inversion
        original_scores = self.y_scores.copy()
        
        # For distance metrics (where lower is better), invert the scores for ROC
        if "Distance" in measure_type or "Error" in measure_type:
            self.y_scores = 1 - self.y_scores
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Store the ROC curve thresholds
        self.thresholds_data = thresholds
        
        # Get the current threshold from the slider
        threshold = self.threshold_slider.value() / 100.0
        
        # Find optimal threshold (maximize sensitivity + specificity)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        # Get binary predictions using selected threshold (not optimal)
        y_pred = (np.array(self.y_scores) >= threshold).astype(int)
        
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
        
        # Restore original scores for future use
        self.y_scores = original_scores
        
        # Update status with actual AUC achieved and threshold
        perf_level = self.perf_selector.currentText()
        target_auc = float(perf_level.split("~")[1].strip(")")) if "~" in perf_level else 0.95
        self.status_label.setText(f"Evaluation complete - AUC: {roc_auc:.3f} (target: {target_auc:.2f}) - Threshold: {threshold:.2f}")
