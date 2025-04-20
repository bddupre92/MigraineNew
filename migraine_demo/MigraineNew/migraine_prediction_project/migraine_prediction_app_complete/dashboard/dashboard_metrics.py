"""
Dashboard-specific implementation of performance metrics for the Streamlit interface.
This is a simplified version of the MigrainePerformanceMetrics class that works with
direct y_true and y_pred inputs rather than requiring a model and test data.
"""

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score

class MigrainePerformanceMetrics:
    """
    Dashboard-specific performance metrics for migraine prediction visualization.
    
    This class is designed to work with pre-computed predictions rather than
    requiring a model and test data.
    """
    
    def __init__(self, y_true, y_pred):
        """
        Initialize with true labels and predicted probabilities.
        
        Args:
            y_true (array): True binary labels
            y_pred (array): Predicted probabilities
        """
        # Ensure inputs are numpy arrays
        self.y_true = np.array(y_true).flatten()
        self.y_pred = np.array(y_pred).flatten()
        
        # Initialize metrics attributes
        self.accuracy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.f1_score = 0.0
        self.specificity = 0.0
        self.auc = 0.0
    
    def calculate_all_metrics(self, threshold=0.5):
        """
        Calculate all performance metrics at once and store as attributes.
        
        Args:
            threshold (float): Probability threshold for binary classification
        """
        # Calculate basic metrics
        self.accuracy, self.precision, self.recall, self.f1_score, self.specificity = self.calculate_metrics(threshold)
        
        # Calculate AUC
        self.auc = self.roc_auc()
    
    def calculate_metrics(self, threshold=0.5):
        """
        Calculate performance metrics at the specified threshold.
        
        Args:
            threshold (float): Probability threshold for binary classification
            
        Returns:
            tuple: (accuracy, precision, recall, f1, specificity)
        """
        # Convert probabilities to binary predictions
        y_pred_binary = (self.y_pred >= threshold).astype(int)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred_binary, labels=[0, 1]).ravel()
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return accuracy, precision, recall, f1, specificity
    
    def roc_auc(self):
        """
        Calculate ROC AUC score.
        
        Returns:
            float: ROC AUC score
        """
        # Handle edge cases
        if len(np.unique(self.y_true)) < 2:
            return 0.5  # Default for single-class data
        
        fpr, tpr, _ = roc_curve(self.y_true, self.y_pred)
        return auc(fpr, tpr)
    
    def precision_recall_curve_data(self):
        """
        Get precision-recall curve data.
        
        Returns:
            tuple: (precision, recall, thresholds)
        """
        precision, recall, thresholds = precision_recall_curve(self.y_true, self.y_pred)
        return precision, recall, thresholds
    
    def get_optimal_threshold(self, criterion='f1'):
        """
        Find the optimal threshold based on the specified criterion.
        
        Args:
            criterion (str): Criterion to optimize ('f1', 'balanced', 'sensitivity')
            
        Returns:
            float: Optimal threshold
        """
        # Get ROC curve data
        fpr, tpr, thresholds_roc = roc_curve(self.y_true, self.y_pred)
        
        if criterion == 'f1':
            # Find threshold that maximizes F1 score
            f1_scores = []
            for threshold in thresholds_roc:
                y_pred_binary = (self.y_pred >= threshold).astype(int)
                f1 = f1_score(self.y_true, y_pred_binary)
                f1_scores.append(f1)
            
            # Get threshold with highest F1 score
            best_idx = np.argmax(f1_scores)
            return thresholds_roc[best_idx]
        
        elif criterion == 'balanced':
            # Find threshold that gives the best balance between sensitivity and specificity
            # (closest point to top-left corner in ROC space)
            distances = np.sqrt((1 - tpr) ** 2 + fpr ** 2)
            best_idx = np.argmin(distances)
            return thresholds_roc[best_idx]
        
        elif criterion == 'sensitivity':
            # Find threshold that gives at least 90% sensitivity
            valid_indices = np.where(tpr >= 0.9)[0]
            if len(valid_indices) > 0:
                # Among thresholds with sufficient sensitivity, choose the one with highest specificity
                best_idx = valid_indices[np.argmin(fpr[valid_indices])]
                return thresholds_roc[best_idx]
            else:
                # If no threshold gives 90% sensitivity, return the one with highest sensitivity
                best_idx = np.argmax(tpr)
                return thresholds_roc[best_idx]
        
        else:
            # Default to 0.5 threshold
            return 0.5
