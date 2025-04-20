"""
Performance Metrics for Migraine Prediction App

This module implements specialized performance metrics for the migraine prediction model,
focusing on high-risk day sensitivity, inference latency, and overall model performance.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import metrics, callbacks
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import time

class MigrainePerformanceMetrics:
    """
    Specialized performance metrics for migraine prediction.
    
    Attributes:
        output_dir (str): Directory to save metric results
        high_risk_threshold (float): Threshold for defining high-risk days
        target_sensitivity (float): Target sensitivity for high-risk days
        target_auc (float): Target AUC for model performance
        target_f1 (float): Target F1 score for model performance
        target_latency_ms (float): Target inference latency in milliseconds
    """
    
    def __init__(self, output_dir='./output', config=None):
        """
        Initialize the MigrainePerformanceMetrics.
        
        Args:
            output_dir (str): Directory to save metric results
            config (dict): Configuration parameters for metrics
        """
        self.output_dir = output_dir
        
        # Default configuration
        self.config = {
            'high_risk_threshold': 0.7,  # Probability threshold for high-risk days
            'target_sensitivity': 0.95,   # Target sensitivity for high-risk days
            'target_auc': 0.80,           # Target AUC
            'target_f1': 0.75,            # Target F1 score
            'target_latency_ms': 200      # Target inference latency in milliseconds
        }
        
        # Update with provided configuration
        if config:
            self.config.update(config)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'metrics'), exist_ok=True)
    
    def calculate_metrics(self, model, X_test_list, y_test):
        """
        Calculate comprehensive performance metrics for the model.
        
        Args:
            model: Trained migraine prediction model
            X_test_list (list): List of test data for each expert
            y_test (array): Test target variable
            
        Returns:
            dict: Dictionary of performance metrics
        """
        print("Calculating performance metrics...")
        
        # Get predictions
        start_time = time.time()
        y_pred_prob = model.predict(X_test_list)
        end_time = time.time()
        
        # Calculate inference time
        total_samples = len(y_test)
        inference_time_ms = (end_time - start_time) * 1000 / total_samples
        
        # Convert probabilities to binary predictions using default threshold (0.5)
        # y_pred = (y_pred_prob > 0.5).astype(int)
        y_pred_bool = y_pred_prob > 0.5
        y_pred = tf.cast(y_pred_bool, dtype=tf.int32)
        
        # Calculate standard metrics
        standard_metrics = self._calculate_standard_metrics(y_test, y_pred, y_pred_prob)
        
        # Calculate high-risk day metrics
        high_risk_metrics = self._calculate_high_risk_metrics(y_test, y_pred_prob)
        
        # Calculate threshold-optimized metrics
        threshold_metrics = self._calculate_threshold_optimized_metrics(y_test, y_pred_prob)
        
        # Calculate performance targets
        performance_targets = self._calculate_performance_targets(
            standard_metrics, high_risk_metrics, inference_time_ms
        )
        
        # Combine all metrics
        all_metrics = {
            **standard_metrics,
            **high_risk_metrics,
            **threshold_metrics,
            'inference_time_ms': inference_time_ms,
            **performance_targets
        }
        
        # Save metrics and test predictions
        self._save_metrics(all_metrics, y_test, y_pred_prob, X_test_list)
        
        return all_metrics
    
    def _calculate_standard_metrics(self, y_true, y_pred, y_pred_prob):
        """
        Calculate standard classification metrics.
        
        Args:
            y_true (array): True labels
            y_pred (array): Predicted labels
            y_pred_prob (array): Predicted probabilities
            
        Returns:
            dict: Dictionary of standard metrics
        """
        # Calculate confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = conf_matrix.ravel()
        
        # Calculate basic metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        # Calculate precision-recall curve and AUC
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_prob)
        pr_auc = auc(recall_curve, precision_curve)
        
        # Calculate specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'specificity': specificity,
            'confusion_matrix': conf_matrix,
            'fpr': fpr,
            'tpr': tpr
        }
    
    def _calculate_high_risk_metrics(self, y_true, y_pred_prob):
        """
        Calculate metrics for high-risk days.
        
        Args:
            y_true (array): True labels
            y_pred_prob (array): Predicted probabilities
            
        Returns:
            dict: Dictionary of high-risk metrics
        """
        # Identify high-risk days (predictions above threshold)
        high_risk_threshold = self.config['high_risk_threshold']
        # high_risk_pred = (y_pred_prob >= high_risk_threshold).astype(int)
        high_risk_pred_bool = y_pred_prob >= high_risk_threshold
        high_risk_pred = tf.cast(high_risk_pred_bool, dtype=tf.int32)
        
        # Calculate metrics for high-risk days
        high_risk_conf_matrix = confusion_matrix(y_true, high_risk_pred)
        
        if high_risk_conf_matrix.shape == (2, 2):
            tn_hr, fp_hr, fn_hr, tp_hr = high_risk_conf_matrix.ravel()
            
            # Calculate high-risk sensitivity (recall)
            high_risk_sensitivity = tp_hr / (tp_hr + fn_hr) if (tp_hr + fn_hr) > 0 else 0
            
            # Calculate high-risk precision
            high_risk_precision = tp_hr / (tp_hr + fp_hr) if (tp_hr + fp_hr) > 0 else 0
            
            # Calculate high-risk F1 score
            high_risk_f1 = 2 * high_risk_precision * high_risk_sensitivity / (high_risk_precision + high_risk_sensitivity) if (high_risk_precision + high_risk_sensitivity) > 0 else 0
            
            # Calculate percentage of days classified as high-risk
            high_risk_percentage = np.mean(high_risk_pred) * 100
        else:
            # Handle case where confusion matrix doesn't have expected shape
            high_risk_sensitivity = 0
            high_risk_precision = 0
            high_risk_f1 = 0
            high_risk_percentage = 0
        
        return {
            'high_risk_threshold': high_risk_threshold,
            'high_risk_sensitivity': high_risk_sensitivity,
            'high_risk_precision': high_risk_precision,
            'high_risk_f1': high_risk_f1,
            'high_risk_percentage': high_risk_percentage,
            'high_risk_confusion_matrix': high_risk_conf_matrix
        }
    
    def _calculate_threshold_optimized_metrics(self, y_true, y_pred_prob):
        """
        Calculate metrics with optimized threshold for maximum F1 score.
        
        Args:
            y_true (array): True labels
            y_pred_prob (array): Predicted probabilities
            
        Returns:
            dict: Dictionary of threshold-optimized metrics
        """
        # Find optimal threshold for F1 score
        precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, y_pred_prob)
        
        # Calculate F1 score for each threshold
        f1_scores = np.zeros_like(thresholds)
        for i, threshold in enumerate(thresholds):
            # y_pred_threshold = (y_pred_prob >= threshold).astype(int)
            y_pred_bool = y_pred_prob >= threshold
            y_pred = tf.cast(y_pred_bool, dtype=tf.int32)
            # Ensure y_pred is evaluated to numpy for confusion_matrix
            y_pred_np = y_pred.numpy()
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_np).ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_scores[i] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Find threshold with maximum F1 score
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        
        # Calculate metrics with optimal threshold
        # y_pred_optimal = (y_pred_prob >= best_threshold).astype(int)
        y_pred_optimal_bool = y_pred_prob >= best_threshold
        y_pred_optimal = tf.cast(y_pred_optimal_bool, dtype=tf.int32)
        # Ensure y_pred_optimal is evaluated to numpy for confusion_matrix
        y_pred_optimal_np = y_pred_optimal.numpy()
        optimal_conf_matrix = confusion_matrix(y_true, y_pred_optimal_np)
        tn_opt, fp_opt, fn_opt, tp_opt = optimal_conf_matrix.ravel()
        
        optimal_precision = tp_opt / (tp_opt + fp_opt) if (tp_opt + fp_opt) > 0 else 0
        optimal_recall = tp_opt / (tp_opt + fn_opt) if (tp_opt + fn_opt) > 0 else 0
        optimal_specificity = tn_opt / (tn_opt + fp_opt) if (tn_opt + fp_opt) > 0 else 0
        
        return {
            'optimal_threshold': best_threshold,
            'optimal_f1': best_f1,
            'optimal_precision': optimal_precision,
            'optimal_recall': optimal_recall,
            'optimal_specificity': optimal_specificity,
            'optimal_confusion_matrix': optimal_conf_matrix
        }
    
    def _calculate_performance_targets(self, standard_metrics, high_risk_metrics, inference_time_ms):
        """
        Calculate whether performance targets are met.
        
        Args:
            standard_metrics (dict): Standard metrics
            high_risk_metrics (dict): High-risk metrics
            inference_time_ms (float): Inference time in milliseconds
            
        Returns:
            dict: Dictionary of performance target results
        """
        # Check if targets are met
        auc_target_met = standard_metrics['roc_auc'] >= self.config['target_auc']
        f1_target_met = standard_metrics['f1_score'] >= self.config['target_f1']
        sensitivity_target_met = high_risk_metrics['high_risk_sensitivity'] >= self.config['target_sensitivity']
        latency_target_met = inference_time_ms <= self.config['target_latency_ms']
        
        # Calculate overall performance score (percentage of targets met)
        targets_met = sum([auc_target_met, f1_target_met, sensitivity_target_met, latency_target_met])
        performance_score = targets_met / 4 * 100
        
        # Check if overall performance target is met (>95%)
        overall_target_met = performance_score > 95
        
        return {
            'auc_target_met': auc_target_met,
            'f1_target_met': f1_target_met,
            'sensitivity_target_met': sensitivity_target_met,
            'latency_target_met': latency_target_met,
            'performance_score': performance_score,
            'overall_target_met': overall_target_met
        }
    
    def _save_metrics(self, metrics, y_true, y_pred_prob, X_test_list=None):
        """
        Save metrics and generate plots.
        
        Args:
            metrics (dict): Dictionary of metrics
            y_true (array): True labels
            y_pred_prob (array): Predicted probabilities
            X_test_list (list, optional): List of test data for each expert
        """
        # Create metrics directory
        metrics_dir = os.path.join(self.output_dir, 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame({k: [v] for k, v in metrics.items() 
                                  if not isinstance(v, (np.ndarray, list, tuple))})
        metrics_df.to_csv(os.path.join(metrics_dir, 'performance_metrics.csv'), index=False)
        
        # Save test predictions for dashboard visualization
        if X_test_list is not None:
            try:
                # Convert to numpy arrays if they're not already
                y_true_np = y_true.numpy() if hasattr(y_true, 'numpy') else np.array(y_true)
                y_pred_prob_np = y_pred_prob.numpy() if hasattr(y_pred_prob, 'numpy') else np.array(y_pred_prob)
                
                # Save test predictions, actual values, and test data
                np.savez(
                    os.path.join(self.output_dir, 'test_predictions.npz'),
                    y_true=y_true_np,
                    y_pred=y_pred_prob_np,
                    # Save a simplified version of X_test_list for visualization
                    # This might need adjustment based on the exact structure of X_test_list
                    X_test_sleep=X_test_list[0] if len(X_test_list) > 0 else np.array([]),
                    X_test_weather=X_test_list[1] if len(X_test_list) > 1 else np.array([]),
                    X_test_stress_diet=X_test_list[2] if len(X_test_list) > 2 else np.array([])
                )
                print(f"Test predictions saved to {os.path.join(self.output_dir, 'test_predictions.npz')}")
            except Exception as e:
                print(f"Error saving test predictions: {e}")
        
        # Generate and save plots
        self._generate_roc_curve_plot(metrics, metrics_dir)
        self._generate_precision_recall_curve(y_true, y_pred_prob, metrics_dir)
        self._generate_confusion_matrix_plot(metrics, metrics_dir)
        self._generate_threshold_analysis_plot(y_true, y_pred_prob, metrics_dir)
        self._generate_performance_summary_plot(metrics, metrics_dir)
    
    def _generate_roc_curve_plot(self, metrics, output_dir):
        """Generate and save ROC curve plot."""
        plt.figure(figsize=(8, 6))
        plt.plot(metrics['fpr'], metrics['tpr'], color='darkorange', lw=2, 
                 label=f'ROC curve (area = {metrics["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        # Add target AUC line
        plt.axhline(y=self.config['target_auc'], color='red', linestyle=':', 
                    label=f'Target AUC = {self.config["target_auc"]}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_precision_recall_curve(self, y_true, y_pred_prob, output_dir):
        """Generate and save precision-recall curve plot."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, 
                 label=f'PR curve (area = {pr_auc:.3f})')
        
        # Add target F1 as contour
        f1_scores = np.linspace(0.1, 0.9, 9)
        for f1_score in f1_scores:
            x = np.linspace(0.01, 1)
            y = f1_score * x / (2 * x - f1_score)
            mask = (y <= 1) & (y >= 0)
            plt.plot(x[mask], y[mask], color='gray', alpha=0.3)
            plt.annotate(f'F1={f1_score:.1f}', 
                         xy=(x[mask][-1], y[mask][-1]), 
                         color='gray', alpha=0.5)
        
        # Highlight target F1
        target_f1 = self.config['target_f1']
        x = np.linspace(0.01, 1)
        y = target_f1 * x / (2 * x - target_f1)
        mask = (y <= 1) & (y >= 0)
        plt.plot(x[mask], y[mask], color='red', linestyle=':', 
                 label=f'Target F1 = {target_f1}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_confusion_matrix_plot(self, metrics, output_dir):
        """Generate and save confusion matrix plots."""
        # Standard confusion matrix
        plt.figure(figsize=(10, 8))
        
        plt.subplot(1, 2, 1)
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Migraine', 'Migraine'],
                    yticklabels=['No Migraine', 'Migraine'])
        plt.title('Standard Threshold (0.5)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # High-risk confusion matrix
        plt.subplot(1, 2, 2)
        sns.heatmap(metrics['high_risk_confusion_matrix'], annot=True, fmt='d', cmap='Oranges',
                    xticklabels=['No Migraine', 'Migraine'],
                    yticklabels=['No Migraine', 'Migraine'])
        plt.title(f'High-Risk Threshold ({self.config["high_risk_threshold"]})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_threshold_analysis_plot(self, y_true, y_pred_prob, output_dir):
        """Generate and save threshold analysis plot."""
        # Calculate metrics at different thresholds
        thresholds = np.linspace(0.1, 0.9, 9)
        sensitivities = []
        specificities = []
        precisions = []
        f1_scores = []
        
        for threshold in thresholds:
            # y_pred = (y_pred_prob >= threshold).astype(int)
            y_pred_bool = y_pred_prob >= threshold
            y_pred = tf.cast(y_pred_bool, dtype=tf.int32)
            # Ensure y_pred is evaluated to numpy for confusion_matrix
            y_pred_np = y_pred.numpy()
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_np).ravel()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
            
            sensitivities.append(sensitivity)
            specificities.append(specificity)
            precisions.append(precision)
            f1_scores.append(f1)
        
        # Plot metrics vs threshold
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, sensitivities, 'o-', label='Sensitivity (Recall)')
        plt.plot(thresholds, specificities, 's-', label='Specificity')
        plt.plot(thresholds, precisions, '^-', label='Precision')
        plt.plot(thresholds, f1_scores, 'D-', label='F1 Score')
        
        # Add target lines
        plt.axhline(y=self.config['target_sensitivity'], color='red', linestyle=':', 
                    label=f'Target Sensitivity = {self.config["target_sensitivity"]}')
        plt.axhline(y=self.config['target_f1'], color='green', linestyle=':', 
                    label=f'Target F1 = {self.config["target_f1"]}')
        
        # Add high-risk threshold
        plt.axvline(x=self.config['high_risk_threshold'], color='purple', linestyle='--', 
                    label=f'High-Risk Threshold = {self.config["high_risk_threshold"]}')
        
        plt.xlabel('Threshold')
        plt.ylabel('Metric Value')
        plt.title('Metrics vs. Threshold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'threshold_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_performance_summary_plot(self, metrics, output_dir):
        """Generate and save performance summary plot."""
        # Create a radar chart of key metrics
        categories = ['AUC', 'F1 Score', 'High-Risk\nSensitivity', 'Latency\n(Normalized)']
        
        # Normalize latency (lower is better)
        latency_normalized = 1 - min(metrics['inference_time_ms'] / self.config['target_latency_ms'], 1)
        
        # Values achieved
        values = [
            metrics['roc_auc'],
            metrics['f1_score'],
            metrics['high_risk_sensitivity'],
            latency_normalized
        ]
        
        # Target values
        targets = [
            self.config['target_auc'],
            self.config['target_f1'],
            self.config['target_sensitivity'],
            1.0  # Target for normalized latency is 1.0 (perfect)
        ]
        
        # Create radar chart
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        values += values[:1]  # Close the loop
        targets += targets[:1]  # Close the loop
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        # Plot targets
        ax.plot(angles, targets, 'r--', linewidth=2, label='Target')
        ax.fill(angles, targets, 'r', alpha=0.1)
        
        # Plot achieved values
        ax.plot(angles, values, 'b-', linewidth=2, label='Achieved')
        ax.fill(angles, values, 'b', alpha=0.2)
        
        # Set category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        
        # Set y-axis limits
        ax.set_ylim(0, 1)
        
        # Add performance score
        plt.figtext(0.5, 0.05, f'Overall Performance Score: {metrics["performance_score"]:.1f}%', 
                   ha='center', fontsize=12, 
                   bbox=dict(facecolor='green' if metrics["overall_target_met"] else 'red', 
                             alpha=0.2, boxstyle='round'))
        
        plt.legend(loc='upper right')
        plt.title('Performance Metrics Summary')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()


class PerformanceMetricsCallback(callbacks.Callback):
    """
    Callback to monitor and log performance metrics during training.
    
    Attributes:
        metrics_tracker: MigrainePerformanceMetrics instance
        validation_data: Validation data for metrics calculation
        log_dir: Directory to save logs
    """
    
    def __init__(self, metrics_tracker, validation_data, log_dir='./logs'):
        """
        Initialize the PerformanceMetricsCallback.
        
        Args:
            metrics_tracker: MigrainePerformanceMetrics instance
            validation_data: Validation data for metrics calculation (X_val_list, y_val)
            log_dir: Directory to save logs
        """
        super(PerformanceMetricsCallback, self).__init__()
        self.metrics_tracker = metrics_tracker
        self.validation_data = validation_data
        self.log_dir = log_dir
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize metrics history
        self.metrics_history = {
            'epoch': [],
            'high_risk_sensitivity': [],
            'roc_auc': [],
            'f1_score': [],
            'inference_time_ms': [],
            'performance_score': []
        }
    
    def on_epoch_end(self, epoch, logs=None):
        """
        Calculate and log metrics at the end of each epoch.
        
        Args:
            epoch: Current epoch
            logs: Training logs
        """
        # Calculate metrics on validation data
        X_val_list, y_val = self.validation_data
        metrics = self.metrics_tracker.calculate_metrics(self.model, X_val_list, y_val)
        
        # Log key metrics
        self.metrics_history['epoch'].append(epoch)
        self.metrics_history['high_risk_sensitivity'].append(metrics['high_risk_sensitivity'])
        self.metrics_history['roc_auc'].append(metrics['roc_auc'])
        self.metrics_history['f1_score'].append(metrics['f1_score'])
        self.metrics_history['inference_time_ms'].append(metrics['inference_time_ms'])
        self.metrics_history['performance_score'].append(metrics['performance_score'])
        
        # Print key metrics
        print(f"\nEpoch {epoch} - Performance Metrics:")
        print(f"  High-Risk Sensitivity: {metrics['high_risk_sensitivity']:.4f}")
        print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  Inference Time: {metrics['inference_time_ms']:.2f} ms")
        print(f"  Performance Score: {metrics['performance_score']:.1f}%")
        
        # Save metrics history
        pd.DataFrame(self.metrics_history).to_csv(
            os.path.join(self.log_dir, 'metrics_history.csv'), index=False
        )
        
        # Generate and save metrics history plot
        self._plot_metrics_history()
    
    def _plot_metrics_history(self):
        """Generate and save metrics history plot."""
        plt.figure(figsize=(12, 8))
        
        # Plot metrics
        plt.subplot(2, 1, 1)
        plt.plot(self.metrics_history['epoch'], self.metrics_history['high_risk_sensitivity'], 
                 'o-', label='High-Risk Sensitivity')
        plt.plot(self.metrics_history['epoch'], self.metrics_history['roc_auc'], 
                 's-', label='ROC AUC')
        plt.plot(self.metrics_history['epoch'], self.metrics_history['f1_score'], 
                 '^-', label='F1 Score')
        
        # Add target lines
        plt.axhline(y=self.metrics_tracker.config['target_sensitivity'], color='red', linestyle=':', 
                    label=f'Target Sensitivity = {self.metrics_tracker.config["target_sensitivity"]}')
        plt.axhline(y=self.metrics_tracker.config['target_auc'], color='green', linestyle=':', 
                    label=f'Target AUC = {self.metrics_tracker.config["target_auc"]}')
        plt.axhline(y=self.metrics_tracker.config['target_f1'], color='blue', linestyle=':', 
                    label=f'Target F1 = {self.metrics_tracker.config["target_f1"]}')
        
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.title('Metrics History')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        # Plot performance score
        plt.subplot(2, 1, 2)
        plt.plot(self.metrics_history['epoch'], self.metrics_history['performance_score'], 
                 'D-', color='purple', label='Performance Score')
        plt.axhline(y=95, color='red', linestyle='--', 
                    label='Target Performance (95%)')
        
        plt.xlabel('Epoch')
        plt.ylabel('Performance Score (%)')
        plt.title('Overall Performance Score History')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'metrics_history.png'), dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    # Example usage would be demonstrated in the main training script
    pass
