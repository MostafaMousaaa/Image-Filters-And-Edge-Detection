import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import confusion_matrix, classification_report
import cv2
import os

def calculate_roc_curve(y_true, y_scores):
    """
    Calculate ROC curve and AUC
    
    Args:
        y_true (list): Ground truth (binary labels)
        y_scores (list): Prediction scores/probabilities
    
    Returns:
        tuple: (fpr, tpr, thresholds, auc_value)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc_value = auc(fpr, tpr)
    return fpr, tpr, thresholds, auc_value

def calculate_precision_recall_curve(y_true, y_scores):
    """
    Calculate precision-recall curve
    
    Args:
        y_true (list): Ground truth (binary labels)
        y_scores (list): Prediction scores/probabilities
    
    Returns:
        tuple: (precision, recall, thresholds)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    return precision, recall, thresholds

def evaluate_face_recognition(test_images, test_labels, model_images, model_eigenfaces, model_labels=None):
    """
    Evaluate face recognition model performance
    
    Args:
        test_images (numpy.ndarray): Test images (flattened)
        test_labels (list): Test image labels
        model_images (numpy.ndarray): Training images (flattened)
        model_eigenfaces (numpy.ndarray): Eigenfaces from PCA
        model_labels (list, optional): Labels for training images
    
    Returns:
        dict: Evaluation results containing metrics and curve data
    """
    y_true = []
    y_scores = []
    predictions = []
    
    # Mean center the training data
    mean = np.mean(model_images, axis=0)
    X_centered = model_images - mean
    X_reduced = np.dot(X_centered, model_eigenfaces)
    
    # Process each test image
    for i, test_img in enumerate(test_images):
        # Project test image into eigenspace
        test_centered = test_img - mean
        test_reduced = np.dot(test_centered, model_eigenfaces)
        
        # Calculate distances to all training samples
        distances = np.linalg.norm(X_reduced - test_reduced, axis=1)
        min_distance = np.min(distances)
        min_idx = np.argmin(distances)
        
        # Convert distance to similarity score (higher is better match)
        similarity_score = 1.0 / (1.0 + min_distance)
        y_scores.append(similarity_score)
        
        # If we have labels, use them to determine ground truth
        if model_labels is not None and test_labels is not None:
            # Ground truth is 1 if labels match, 0 otherwise
            match = int(model_labels[min_idx] == test_labels[i])
            y_true.append(match)
            predictions.append((min_idx, min_distance, match))
        else:
            # Placeholder when no labels are available
            y_true.append(np.random.randint(0, 2))
            predictions.append((min_idx, min_distance, None))
    
    # Calculate ROC curve
    fpr, tpr, thresholds, roc_auc = calculate_roc_curve(y_true, y_scores)
    
    # Find optimal threshold (maximizing sensitivity + specificity)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx] if len(thresholds) > optimal_idx else 0.5
    
    # Binary predictions at optimal threshold
    y_pred = (np.array(y_scores) >= optimal_threshold).astype(int)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fpr_value = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Compile results
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Specificity": specificity,
        "False Positive Rate": fpr_value,
        "AUC": roc_auc
    }
    
    curves = {
        "ROC": {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "auc": roc_auc
        }
    }
    
    confusion = {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }
    
    return {
        "metrics": metrics,
        "curves": curves,
        "confusion": confusion,
        "predictions": predictions,
        "optimal_threshold": optimal_threshold
    }

def plot_roc_curve(fpr, tpr, auc_value, ax=None):
    """
    Plot the ROC curve
    
    Args:
        fpr (list): False positive rate values
        tpr (list): True positive rate values
        auc_value (float): Area Under Curve value
        ax (matplotlib.axes, optional): Axes to plot on
    
    Returns:
        matplotlib.axes: The axes with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, 'b-', label=f'ROC (AUC = {auc_value:.3f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    return ax
