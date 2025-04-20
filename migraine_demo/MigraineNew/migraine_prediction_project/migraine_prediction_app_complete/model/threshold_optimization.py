import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc, f1_score, precision_score, recall_score
import tensorflow as tf
import os

class ThresholdOptimizer:
    """
    A class for optimizing classification thresholds to improve precision and recall
    for migraine prediction models.
    """
    
    def __init__(self, output_dir='output'):
        """
        Initialize the ThresholdOptimizer.
        
        Args:
            output_dir (str): Directory to save results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def find_optimal_threshold(self, y_true, y_pred, method='f1'):
        """
        Find the optimal threshold based on the specified method.
        
        Args:
            y_true (np.ndarray): True binary labels
            y_pred (np.ndarray): Predicted probabilities
            method (str): Method to use for finding optimal threshold
                          Options: 'f1', 'precision_recall_balance', 'cost_based'
                          
        Returns:
            float: Optimal threshold
        """
        if method == 'f1':
            return self._find_optimal_f1_threshold(y_true, y_pred)
        elif method == 'precision_recall_balance':
            return self._find_precision_recall_balance(y_true, y_pred)
        elif method == 'cost_based':
            # Default cost ratio: false negative is 3x more costly than false positive
            return self._find_cost_based_threshold(y_true, y_pred, cost_ratio=3.0)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _find_optimal_f1_threshold(self, y_true, y_pred):
        """
        Find threshold that maximizes F1 score.
        
        Args:
            y_true (np.ndarray): True binary labels
            y_pred (np.ndarray): Predicted probabilities
            
        Returns:
            float: Optimal threshold
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        
        # Add a threshold of 1.0 to match the length of precision and recall
        thresholds = np.append(thresholds, 1.0)
        
        # Calculate F1 score for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # Find threshold that maximizes F1 score
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        print(f"Optimal F1 threshold: {optimal_threshold:.4f}")
        print(f"Optimal F1 score: {f1_scores[optimal_idx]:.4f}")
        print(f"Precision at optimal threshold: {precision[optimal_idx]:.4f}")
        print(f"Recall at optimal threshold: {recall[optimal_idx]:.4f}")
        
        return optimal_threshold
    
    def _find_precision_recall_balance(self, y_true, y_pred):
        """
        Find threshold where precision equals recall.
        
        Args:
            y_true (np.ndarray): True binary labels
            y_pred (np.ndarray): Predicted probabilities
            
        Returns:
            float: Optimal threshold
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        
        # Add a threshold of 1.0 to match the length of precision and recall
        thresholds = np.append(thresholds, 1.0)
        
        # Find the point where precision and recall are closest
        diff = np.abs(precision - recall)
        optimal_idx = np.argmin(diff)
        optimal_threshold = thresholds[optimal_idx]
        
        print(f"Optimal balanced threshold: {optimal_threshold:.4f}")
        print(f"Precision at optimal threshold: {precision[optimal_idx]:.4f}")
        print(f"Recall at optimal threshold: {recall[optimal_idx]:.4f}")
        
        return optimal_threshold
    
    def _find_cost_based_threshold(self, y_true, y_pred, cost_ratio=3.0):
        """
        Find threshold based on cost ratio between false negatives and false positives.
        For migraine prediction, false negatives (missing a migraine) are typically more 
        costly than false positives (false alarm).
        
        Args:
            y_true (np.ndarray): True binary labels
            y_pred (np.ndarray): Predicted probabilities
            cost_ratio (float): Ratio of cost of false negative to false positive
            
        Returns:
            float: Optimal threshold
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        
        # Add a threshold of 1.0 to match the length of precision and recall
        thresholds = np.append(thresholds, 1.0)
        
        # Calculate the number of true positives, false positives, and false negatives at each threshold
        n_pos = np.sum(y_true)
        n_neg = len(y_true) - n_pos
        
        tps = recall * n_pos
        fps = (tps / precision) - tps if precision.all() else np.zeros_like(tps)
        fns = n_pos - tps
        
        # Calculate cost at each threshold
        costs = (cost_ratio * fns + fps) / len(y_true)
        
        # Find threshold that minimizes cost
        optimal_idx = np.argmin(costs)
        optimal_threshold = thresholds[optimal_idx]
        
        print(f"Optimal cost-based threshold (cost ratio {cost_ratio}): {optimal_threshold:.4f}")
        print(f"Precision at optimal threshold: {precision[optimal_idx]:.4f}")
        print(f"Recall at optimal threshold: {recall[optimal_idx]:.4f}")
        
        return optimal_threshold
    
    def plot_precision_recall_curve(self, y_true, y_pred, optimal_threshold=None, save_path=None):
        """
        Plot precision-recall curve and mark the optimal threshold.
        
        Args:
            y_true (np.ndarray): True binary labels
            y_pred (np.ndarray): Predicted probabilities
            optimal_threshold (float, optional): Optimal threshold to mark on the plot
            save_path (str, optional): Path to save the plot
            
        Returns:
            tuple: (precision, recall, thresholds)
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        
        # Add a threshold of 1.0 to match the length of precision and recall
        thresholds = np.append(thresholds, 1.0)
        
        # Calculate F1 score for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, 'b-', linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True, alpha=0.3)
        
        # Mark optimal threshold if provided
        if optimal_threshold is not None:
            # Find the index of the closest threshold
            idx = np.argmin(np.abs(thresholds - optimal_threshold))
            plt.plot(recall[idx], precision[idx], 'ro', markersize=8)
            plt.annotate(f'Threshold: {optimal_threshold:.2f}\nPrecision: {precision[idx]:.2f}\nRecall: {recall[idx]:.2f}',
                        xy=(recall[idx], precision[idx]),
                        xytext=(recall[idx] - 0.2, precision[idx] - 0.2),
                        arrowprops=dict(arrowstyle='->', lw=1.5))
        
        # Add F1 score contours
        f1_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for f1_level in f1_levels:
            # Calculate precision for each recall value to maintain constant F1
            r = np.linspace(0.01, 0.99, 100)
            p = (f1_level * r) / (2 * r - f1_level + 1e-10)
            
            # Only plot valid precision values (0 <= p <= 1)
            valid_indices = (p >= 0) & (p <= 1)
            if np.any(valid_indices):
                plt.plot(r[valid_indices], p[valid_indices], 'g--', alpha=0.5)
                # Add label at the midpoint of the line
                mid_idx = np.where(valid_indices)[0][len(np.where(valid_indices)[0])//2]
                plt.annotate(f'F1={f1_level}', 
                            xy=(r[mid_idx], p[mid_idx]),
                            xytext=(r[mid_idx], p[mid_idx] + 0.05),
                            ha='center',
                            alpha=0.7)
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-recall curve saved to {save_path}")
        
        plt.close()
        
        return precision, recall, thresholds
    
    def plot_threshold_metrics(self, y_true, y_pred, save_path=None):
        """
        Plot precision, recall, and F1 score as a function of threshold.
        
        Args:
            y_true (np.ndarray): True binary labels
            y_pred (np.ndarray): Predicted probabilities
            save_path (str, optional): Path to save the plot
            
        Returns:
            tuple: (thresholds, precision, recall, f1_scores)
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        
        # Add a threshold of 1.0 to match the length of precision and recall
        thresholds = np.append(thresholds, 1.0)
        
        # Calculate F1 score for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(thresholds, precision, 'b-', label='Precision', linewidth=2)
        plt.plot(thresholds, recall, 'r-', label='Recall', linewidth=2)
        plt.plot(thresholds, f1_scores, 'g-', label='F1 Score', linewidth=2)
        
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Precision, Recall, and F1 Score vs. Threshold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # Find optimal thresholds for different metrics
        optimal_f1_idx = np.argmax(f1_scores)
        optimal_f1_threshold = thresholds[optimal_f1_idx]
        
        # Find threshold where precision equals recall
        diff = np.abs(precision - recall)
        optimal_balance_idx = np.argmin(diff)
        optimal_balance_threshold = thresholds[optimal_balance_idx]
        
        # Mark optimal thresholds on the plot
        plt.axvline(x=optimal_f1_threshold, color='g', linestyle='--', alpha=0.7)
        plt.annotate(f'Optimal F1: {optimal_f1_threshold:.2f}',
                    xy=(optimal_f1_threshold, f1_scores[optimal_f1_idx]),
                    xytext=(optimal_f1_threshold + 0.1, f1_scores[optimal_f1_idx]),
                    arrowprops=dict(arrowstyle='->', lw=1.5))
        
        plt.axvline(x=optimal_balance_threshold, color='purple', linestyle='--', alpha=0.7)
        plt.annotate(f'Balanced: {optimal_balance_threshold:.2f}',
                    xy=(optimal_balance_threshold, precision[optimal_balance_idx]),
                    xytext=(optimal_balance_threshold + 0.1, precision[optimal_balance_idx] - 0.1),
                    arrowprops=dict(arrowstyle='->', lw=1.5))
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Threshold metrics plot saved to {save_path}")
        
        plt.close()
        
        return thresholds, precision, recall, f1_scores
    
    def evaluate_with_optimal_threshold(self, y_true, y_pred, optimal_threshold):
        """
        Evaluate model performance using the optimal threshold.
        
        Args:
            y_true (np.ndarray): True binary labels
            y_pred (np.ndarray): Predicted probabilities
            optimal_threshold (float): Optimal threshold
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        # Apply threshold to get binary predictions
        y_pred_binary = (y_pred >= optimal_threshold).astype(int)
        
        # Calculate metrics
        precision = precision_score(y_true, y_pred_binary)
        recall = recall_score(y_true, y_pred_binary)
        f1 = f1_score(y_true, y_pred_binary)
        
        # Calculate ROC AUC
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        # Print metrics
        print("\nModel Performance with Optimal Threshold:")
        print(f"Threshold: {optimal_threshold:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        
        return {
            'threshold': optimal_threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }
    
    def compare_thresholds(self, y_true, y_pred, thresholds_dict, save_path=None):
        """
        Compare model performance at different thresholds.
        
        Args:
            y_true (np.ndarray): True binary labels
            y_pred (np.ndarray): Predicted probabilities
            thresholds_dict (dict): Dictionary of thresholds to compare
                                    {name: threshold}
            save_path (str, optional): Path to save the comparison plot
            
        Returns:
            dict: Dictionary of evaluation metrics for each threshold
        """
        results = {}
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        
        # Plot precision-recall curve
        plt.plot(recall, precision, 'b-', linewidth=2, label='Precision-Recall Curve')
        
        # Evaluate each threshold
        for name, threshold in thresholds_dict.items():
            # Apply threshold to get binary predictions
            y_pred_binary = (y_pred >= threshold).astype(int)
            
            # Calculate metrics
            precision_val = precision_score(y_true, y_pred_binary)
            recall_val = recall_score(y_true, y_pred_binary)
            f1_val = f1_score(y_true, y_pred_binary)
            
            # Store results
            results[name] = {
                'threshold': threshold,
                'precision': precision_val,
                'recall': recall_val,
                'f1': f1_val
            }
            
            # Mark threshold on the plot
            plt.plot(recall_val, precision_val, 'o', markersize=8, label=f'{name} (t={threshold:.2f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Comparison of Different Thresholds')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Threshold comparison plot saved to {save_path}")
        
        plt.close()
        
        # Print comparison table
        print("\nThreshold Comparison:")
        print(f"{'Threshold':<20} {'Value':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10}")
        print("-" * 60)
        for name, metrics in results.items():
            print(f"{name:<20} {metrics['threshold']:<10.4f} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1']:<10.4f}")
        
        return results
    
    def optimize_and_evaluate(self, y_true, y_pred, methods=None, save_dir=None):
        """
        Optimize thresholds using multiple methods and evaluate performance.
        
        Args:
            y_true (np.ndarray): True binary labels
            y_pred (np.ndarray): Predicted probabilities
            methods (list, optional): List of methods to use for threshold optimization
                                     Default: ['f1', 'precision_recall_balance', 'cost_based']
            save_dir (str, optional): Directory to save plots
            
        Returns:
            dict: Dictionary of results for each method
        """
        if methods is None:
            methods = ['f1', 'precision_recall_balance', 'cost_based']
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        results = {}
        thresholds_dict = {}
        
        # Find optimal threshold for each method
        for method in methods:
            print(f"\nOptimizing threshold using {method} method:")
            optimal_threshold = self.find_optimal_threshold(y_true, y_pred, method=method)
            
            # Evaluate with optimal threshold
            metrics = self.evaluate_with_optimal_threshold(y_true, y_pred, optimal_threshold)
            
            # Store results
            results[method] = metrics
            thresholds_dict[method] = optimal_threshold
        
        # Add default threshold (0.5) for comparison
        default_threshold = 0.5
        metrics_default = self.evaluate_with_optimal_threshold(y_true, y_pred, default_threshold)
        results['default'] = metrics_default
        thresholds_dict['default'] = default_threshold
        
        # Add low threshold (0.3) for comparison (as suggested in the user feedback)
        low_threshold = 0.3
        metrics_low = self.evaluate_with_optimal_threshold(y_true, y_pred, low_threshold)
        results['low_threshold'] = metrics_low
        thresholds_dict['low_threshold'] = low_threshold
        
        # Generate plots
        if save_dir:
            # Precision-recall curve
            self.plot_precision_recall_curve(y_true, y_pred, 
                                            optimal_threshold=results['f1']['threshold'],
                                            save_path=os.path.join(save_dir, 'precision_recall_curve.png'))
            
            # Threshold metrics
            self.plot_threshold_metrics(y_true, y_pred,
                                      save_path=os.path.join(save_dir, 'threshold_metrics.png'))
            
            # Compare thresholds
            self.compare_thresholds(y_true, y_pred, thresholds_dict,
                                  save_path=os.path.join(save_dir, 'threshold_comparison.png'))
        
        return results


# Example usage
if __name__ == "__main__":
    # Load test predictions
    try:
        test_data = np.load('output/test_predictions.npz', allow_pickle=True)
        y_true = test_data['y_true']
        y_pred = test_data['y_pred']
        
        # Initialize threshold optimizer
        optimizer = ThresholdOptimizer(output_dir='output/threshold_optimization')
        
        # Optimize and evaluate
        results = optimizer.optimize_and_evaluate(y_true, y_pred, save_dir='output/threshold_optimization')
        
        # Print summary
        print("\nSummary of Threshold Optimization:")
        print(f"{'Method':<25} {'Threshold':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10}")
        print("-" * 65)
        for method, metrics in results.items():
            print(f"{method:<25} {metrics['threshold']:<10.4f} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1']:<10.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
