import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
import matplotlib.pyplot as plt
import os

class ClassBalancer:
    """
    A class for implementing various class balancing techniques to address
    imbalanced data in migraine prediction.
    """
    
    def __init__(self, output_dir='output'):
        """
        Initialize the ClassBalancer.
        
        Args:
            output_dir (str): Directory to save results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def apply_smote(self, X, y, sampling_strategy=0.5, k_neighbors=5, random_state=42):
        """
        Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance classes.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Labels
            sampling_strategy (float or str): Ratio of minority to majority class after resampling
                                             or 'auto' to automatically determine
            k_neighbors (int): Number of nearest neighbors to use
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_resampled, y_resampled)
        """
        print(f"Applying SMOTE with sampling_strategy={sampling_strategy}, k_neighbors={k_neighbors}")
        
        # Reshape X if needed (SMOTE expects 2D array)
        original_shape = X.shape
        if len(original_shape) > 2:
            X_reshaped = X.reshape(original_shape[0], -1)
        else:
            X_reshaped = X
        
        # Apply SMOTE
        smote = SMOTE(sampling_strategy=sampling_strategy, 
                     k_neighbors=k_neighbors, 
                     random_state=random_state)
        
        X_resampled, y_resampled = smote.fit_resample(X_reshaped, y)
        
        # Reshape X back to original shape if needed
        if len(original_shape) > 2:
            new_shape = (X_resampled.shape[0],) + original_shape[1:]
            X_resampled = X_resampled.reshape(new_shape)
        
        # Print class distribution
        unique, counts = np.unique(y_resampled, return_counts=True)
        print("Class distribution after SMOTE:")
        for cls, count in zip(unique, counts):
            print(f"Class {cls}: {count} samples ({count/len(y_resampled)*100:.2f}%)")
        
        return X_resampled, y_resampled
    
    def apply_advanced_smote(self, X, y, method='borderline', sampling_strategy=0.5, random_state=42):
        """
        Apply advanced SMOTE variants for better synthetic sample generation.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Labels
            method (str): SMOTE variant to use ('borderline', 'adasyn', 'smoteenn', 'smotetomek')
            sampling_strategy (float or str): Ratio of minority to majority class after resampling
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_resampled, y_resampled)
        """
        print(f"Applying {method} SMOTE with sampling_strategy={sampling_strategy}")
        
        # Reshape X if needed (SMOTE expects 2D array)
        original_shape = X.shape
        if len(original_shape) > 2:
            X_reshaped = X.reshape(original_shape[0], -1)
        else:
            X_reshaped = X
        
        # Select SMOTE variant
        if method == 'borderline':
            sampler = BorderlineSMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
        elif method == 'adasyn':
            sampler = ADASYN(sampling_strategy=sampling_strategy, random_state=random_state)
        elif method == 'smoteenn':
            sampler = SMOTEENN(sampling_strategy=sampling_strategy, random_state=random_state)
        elif method == 'smotetomek':
            sampler = SMOTETomek(sampling_strategy=sampling_strategy, random_state=random_state)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Apply sampling
        X_resampled, y_resampled = sampler.fit_resample(X_reshaped, y)
        
        # Reshape X back to original shape if needed
        if len(original_shape) > 2:
            new_shape = (X_resampled.shape[0],) + original_shape[1:]
            X_resampled = X_resampled.reshape(new_shape)
        
        # Print class distribution
        unique, counts = np.unique(y_resampled, return_counts=True)
        print(f"Class distribution after {method}:")
        for cls, count in zip(unique, counts):
            print(f"Class {cls}: {count} samples ({count/len(y_resampled)*100:.2f}%)")
        
        return X_resampled, y_resampled
    
    def calculate_class_weights(self, y, balanced=True):
        """
        Calculate class weights for imbalanced data.
        
        Args:
            y (np.ndarray): Labels
            balanced (bool): Whether to use 'balanced' weighting
            
        Returns:
            dict: Class weights dictionary
        """
        if balanced:
            # Automatically calculate balanced class weights
            class_weights = class_weight.compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y),
                y=y
            )
            
            # Convert to dictionary
            class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
        else:
            # Calculate weights based on inverse frequency
            unique, counts = np.unique(y, return_counts=True)
            total = len(y)
            
            # Calculate weights as inverse of frequency
            weights = total / (len(unique) * counts)
            
            # Convert to dictionary
            class_weights_dict = {cls: weight for cls, weight in zip(unique, weights)}
        
        print("Class weights:")
        for cls, weight in class_weights_dict.items():
            print(f"Class {cls}: {weight:.4f}")
        
        return class_weights_dict
    
    def focal_loss(self, gamma=2.0, alpha=0.25):
        """
        Create a focal loss function for imbalanced classification.
        Focal loss focuses more on hard examples and down-weights easy examples.
        
        Args:
            gamma (float): Focusing parameter that controls how much to down-weight easy examples
            alpha (float): Weighting factor for the positive class
            
        Returns:
            function: Focal loss function
        """
        def focal_loss_fixed(y_true, y_pred):
            """
            Focal loss function implementation.
            
            Args:
                y_true (tf.Tensor): True labels
                y_pred (tf.Tensor): Predicted probabilities
                
            Returns:
                tf.Tensor: Focal loss value
            """
            # Clip predictions to prevent NaN losses
            epsilon = K.epsilon()
            y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
            
            # Calculate cross entropy
            cross_entropy = -y_true * K.log(y_pred)
            
            # Calculate focal loss
            loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
            
            # Sum over all samples
            return K.mean(loss, axis=-1)
        
        return focal_loss_fixed
    
    def plot_class_distribution(self, y_original, y_resampled=None, save_path=None):
        """
        Plot class distribution before and after resampling.
        
        Args:
            y_original (np.ndarray): Original labels
            y_resampled (np.ndarray, optional): Resampled labels
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        # Count original class distribution
        unique_original, counts_original = np.unique(y_original, return_counts=True)
        
        # Plot original distribution
        plt.bar(unique_original - 0.2, counts_original, width=0.4, label='Original', color='blue', alpha=0.7)
        
        # Add percentages to original bars
        for i, count in enumerate(counts_original):
            percentage = count / len(y_original) * 100
            plt.text(unique_original[i] - 0.2, count + 5, f"{percentage:.1f}%", ha='center')
        
        # Plot resampled distribution if provided
        if y_resampled is not None:
            unique_resampled, counts_resampled = np.unique(y_resampled, return_counts=True)
            
            plt.bar(unique_resampled + 0.2, counts_resampled, width=0.4, label='Resampled', color='red', alpha=0.7)
            
            # Add percentages to resampled bars
            for i, count in enumerate(counts_resampled):
                percentage = count / len(y_resampled) * 100
                plt.text(unique_resampled[i] + 0.2, count + 5, f"{percentage:.1f}%", ha='center')
        
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Class Distribution')
        plt.xticks(unique_original, [f'Class {int(cls)}' for cls in unique_original])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Class distribution plot saved to {save_path}")
        
        plt.close()
    
    def compare_balancing_techniques(self, X, y, techniques=None, save_dir=None):
        """
        Compare different class balancing techniques.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Labels
            techniques (list, optional): List of techniques to compare
                                        Default: ['smote', 'borderline', 'adasyn', 'class_weights']
            save_dir (str, optional): Directory to save plots
            
        Returns:
            dict: Dictionary of resampled data for each technique
        """
        if techniques is None:
            techniques = ['smote', 'borderline', 'adasyn', 'class_weights']
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        results = {}
        
        # Plot original class distribution
        if save_dir:
            self.plot_class_distribution(y, save_path=os.path.join(save_dir, 'original_distribution.png'))
        
        # Apply each technique
        for technique in techniques:
            print(f"\nApplying {technique} technique:")
            
            if technique == 'smote':
                X_resampled, y_resampled = self.apply_smote(X, y)
                results[technique] = (X_resampled, y_resampled)
                
                # Plot resampled distribution
                if save_dir:
                    self.plot_class_distribution(y, y_resampled, 
                                               save_path=os.path.join(save_dir, f'{technique}_distribution.png'))
            
            elif technique == 'borderline':
                X_resampled, y_resampled = self.apply_advanced_smote(X, y, method='borderline')
                results[technique] = (X_resampled, y_resampled)
                
                # Plot resampled distribution
                if save_dir:
                    self.plot_class_distribution(y, y_resampled, 
                                               save_path=os.path.join(save_dir, f'{technique}_distribution.png'))
            
            elif technique == 'adasyn':
                X_resampled, y_resampled = self.apply_advanced_smote(X, y, method='adasyn')
                results[technique] = (X_resampled, y_resampled)
                
                # Plot resampled distribution
                if save_dir:
                    self.plot_class_distribution(y, y_resampled, 
                                               save_path=os.path.join(save_dir, f'{technique}_distribution.png'))
            
            elif technique == 'class_weights':
                class_weights = self.calculate_class_weights(y)
                results[technique] = class_weights
            
            elif technique == 'focal_loss':
                focal_loss_fn = self.focal_loss(gamma=2.0, alpha=0.25)
                results[technique] = focal_loss_fn
            
            else:
                print(f"Unknown technique: {technique}")
        
        return results


# Example usage
if __name__ == "__main__":
    # Generate sample imbalanced data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Create imbalanced dataset (10% positive, 90% negative)
    X = np.random.randn(n_samples, n_features)
    y = np.zeros(n_samples)
    y[:100] = 1  # 10% positive samples
    
    # Initialize class balancer
    balancer = ClassBalancer(output_dir='output/class_balancing')
    
    # Compare balancing techniques
    results = balancer.compare_balancing_techniques(X, y, save_dir='output/class_balancing')
    
    # Print summary
    print("\nSummary of Class Balancing Techniques:")
    for technique, result in results.items():
        if technique in ['smote', 'borderline', 'adasyn']:
            X_resampled, y_resampled = result
            unique, counts = np.unique(y_resampled, return_counts=True)
            print(f"{technique}: {len(y_resampled)} samples, {counts[1]}/{counts[0]} positive/negative ratio")
        elif technique == 'class_weights':
            print(f"{technique}: {result}")
        elif technique == 'focal_loss':
            print(f"{technique}: Focal loss function created with gamma=2.0, alpha=0.25")
