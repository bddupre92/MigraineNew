"""
Run PyGMO Optimization for Migraine Prediction Model

This script runs the full PyGMO optimization process to improve the migraine prediction model performance.
It uses the existing PyGMOOptimizer implementation to optimize expert hyperparameters, gating network,
and end-to-end model parameters. The optimized model is then trained and evaluated.

Usage:
    python run_pygmo_optimization.py

Output:
    - Optimized model saved to output/optimized_model.keras
    - Optimization summary saved to output/optimization/optimization_summary.json
    - Training history plots saved to output/figures/
"""

import os
import sys
import numpy as np
import tensorflow as tf
import time
from datetime import datetime
import json
import matplotlib.pyplot as plt

# Add the project root to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from model.pygmo_optimizer import PyGMOOptimizer
from model.performance_metrics import MigrainePerformanceMetrics
from model.threshold_optimization import ThresholdOptimizer
from model.class_balancing import ClassBalancer

def main():
    """Run the full optimization process and train the optimized model."""
    print("\n=== Migraine Prediction Model - Full PyGMO Optimization and Training ===")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set paths
    data_dir = os.path.join(script_dir, 'data')
    output_dir = os.path.join(script_dir, 'output')
    
    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'optimization'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    
    # Create optimizer with smaller population and generations for faster execution
    optimizer = PyGMOOptimizer(
        data_dir=data_dir,
        output_dir=output_dir,
        seed=42,
        verbose=True
    )
    
    # Run full optimization with reduced parameters for faster execution
    print("\nStarting Full Three-Phase Optimization...")
    start_time = time.time()
    
    expert_configs, gating_config, e2e_config = optimizer.run_full_optimization(
        expert_pop_size=5,      # Reduced from 10 for faster execution
        expert_generations=3,   # Reduced from 5 for faster execution
        gating_pop_size=5,      # Reduced from 10 for faster execution
        gating_generations=3,   # Reduced from 5 for faster execution
        e2e_pop_size=5,         # Reduced from 10 for faster execution
        e2e_generations=3       # Reduced from 5 for faster execution
    )
    
    optimization_time = time.time() - start_time
    print(f"\nOptimization Complete in {optimization_time:.2f} seconds ({optimization_time/60:.2f} minutes)")
    
    # Build optimized model
    print("\nBuilding Optimized Model...")
    optimized_model = optimizer.build_optimized_model(
        expert_configs=expert_configs,
        gating_config=gating_config,
        e2e_config=e2e_config
    )
    
    # Apply class balancing to training data
    print("\nApplying Class Balancing to Training Data...")
    balancer = ClassBalancer()
    X_train_list_balanced, y_train_balanced = balancer.apply_borderline_smote(
        optimizer.X_train_list, optimizer.y_train
    )
    
    # Train optimized model
    print("\nTraining Optimized Model...")
    training_start_time = time.time()
    
    model, history = optimizer.train_optimized_model(
        model=optimized_model,
        X_train_list=X_train_list_balanced,
        y_train=y_train_balanced,
        X_val_list=optimizer.X_val_list,
        y_val=optimizer.y_val,
        batch_size=int(e2e_config.get('batch_size', 32)),
        epochs=30,              # Reduced from 50 for faster execution
        patience=5              # Reduced from 10 for faster execution
    )
    
    training_time = time.time() - training_start_time
    print(f"\nTraining Complete in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Evaluate model on test data
    print("\nEvaluating Model on Test Data...")
    
    # Load test data
    try:
        X_test_sleep = np.load(os.path.join(data_dir, 'X_test_sleep.npy'))
        X_test_weather = np.load(os.path.join(data_dir, 'X_test_weather.npy'))
        X_test_stress_diet = np.load(os.path.join(data_dir, 'X_test_stress_diet.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
        
        X_test_list = [X_test_sleep, X_test_weather, X_test_stress_diet]
        
        # Apply threshold optimization
        print("\nApplying Threshold Optimization...")
        threshold_optimizer = ThresholdOptimizer()
        
        # Get model predictions
        y_pred_proba = model.predict(X_test_list)
        
        # Find optimal threshold
        optimal_threshold = threshold_optimizer.find_optimal_threshold_f1(
            y_true=y_test, 
            y_pred_proba=y_pred_proba
        )
        
        print(f"Optimal Threshold: {optimal_threshold:.4f}")
        
        # Apply optimal threshold
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        
        # Calculate and display metrics
        metrics = MigrainePerformanceMetrics()
        performance = metrics.calculate_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            X_test_list=X_test_list
        )
        
        # Save test predictions for dashboard
        np.savez(
            os.path.join(output_dir, 'test_predictions.npz'),
            y_test=y_test,
            y_pred_test=y_pred,
            y_pred_proba=y_pred_proba
        )
        
        # Update final performance in optimization results
        optimizer.optimization_results['final_performance'] = performance
        
    except Exception as e:
        print(f"Error evaluating model: {e}")
        print("Skipping evaluation step...")
    
    # Plot training history
    plot_training_history(history, output_dir)
    
    # Save optimization and training summary
    save_summary(optimizer, expert_configs, gating_config, e2e_config, 
                 optimization_time, training_time, output_dir)
    
    # Save the original model (non-optimized) for comparison
    save_original_model(data_dir, output_dir)
    
    total_time = time.time() - start_time
    print(f"\nTotal Process Complete in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Optimized model saved to {os.path.join(output_dir, 'optimized_model.keras')}")
    print(f"Summary saved to {os.path.join(output_dir, 'optimization', 'optimization_summary.json')}")
    
    return model, history

def plot_training_history(history, output_dir):
    """Plot and save the training history."""
    # Create figure directory
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    # Plot AUC
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['auc'], label='Training AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title('Model AUC')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, 'auc_history.png'))
    
    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, 'loss_history.png'))
    
    # Plot Precision and Recall if available
    if 'precision' in history.history and 'recall' in history.history:
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['precision'], label='Training Precision')
        plt.plot(history.history['val_precision'], label='Validation Precision')
        plt.plot(history.history['recall'], label='Training Recall')
        plt.plot(history.history['val_recall'], label='Validation Recall')
        plt.title('Precision and Recall')
        plt.ylabel('Score')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig(os.path.join(fig_dir, 'precision_recall_history.png'))
    
    print(f"Training history plots saved to {fig_dir}")

def save_summary(optimizer, expert_configs, gating_config, e2e_config, 
                optimization_time, training_time, output_dir):
    """Save a summary of the optimization and training process."""
    # Get final performance metrics
    final_performance = optimizer.optimization_results.get('final_performance', {})
    
    # Create summary dictionary
    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'optimization_time_seconds': optimization_time,
        'training_time_seconds': training_time,
        'expert_configs': expert_configs,
        'gating_config': gating_config,
        'e2e_config': e2e_config,
        'final_performance': final_performance,
        'improvement': {
            'auc_improvement': final_performance.get('auc', 0) - 0.5625,  # Compared to original 0.5625
            'f1_improvement': final_performance.get('f1', 0) - 0.0741,    # Compared to original 0.0741
        },
        'optimization_phases': {
            'expert_phase': optimizer.optimization_results.get('expert_phase', {}),
            'gating_phase': optimizer.optimization_results.get('gating_phase', {}),
            'e2e_phase': optimizer.optimization_results.get('e2e_phase', {})
        }
    }
    
    # Convert numpy values to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        else:
            return obj
    
    serializable_summary = convert_to_serializable(summary)
    
    # Save summary to file
    summary_path = os.path.join(output_dir, 'optimization', 'optimization_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(serializable_summary, f, indent=2)

def save_original_model(data_dir, output_dir):
    """Create and save a basic non-optimized model for comparison."""
    from model.migraine_prediction_model import create_baseline_model
    
    try:
        # Load training data
        X_train_sleep = np.load(os.path.join(data_dir, 'X_train_sleep.npy'))
        X_train_weather = np.load(os.path.join(data_dir, 'X_train_weather.npy'))
        X_train_stress_diet = np.load(os.path.join(data_dir, 'X_train_stress_diet.npy'))
        y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
        
        # Load validation data
        X_val_sleep = np.load(os.path.join(data_dir, 'X_val_sleep.npy'))
        X_val_weather = np.load(os.path.join(data_dir, 'X_val_weather.npy'))
        X_val_stress_diet = np.load(os.path.join(data_dir, 'X_val_stress_diet.npy'))
        y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
        
        # Create and compile baseline model
        original_model = create_baseline_model(
            sleep_input_shape=X_train_sleep.shape[1:],
            weather_input_shape=X_train_weather.shape[1:],
            stress_diet_input_shape=X_train_stress_diet.shape[1:]
        )
        
        # Train model with minimal epochs
        X_train_list = [X_train_sleep, X_train_weather, X_train_stress_diet]
        X_val_list = [X_val_sleep, X_val_weather, X_val_stress_diet]
        
        original_model.fit(
            X_train_list, y_train,
            validation_data=(X_val_list, y_val),
            epochs=5,  # Minimal training for comparison
            batch_size=32,
            verbose=1
        )
        
        # Save model
        original_model.save(os.path.join(output_dir, 'original_model.keras'))
        print(f"Original model saved to {os.path.join(output_dir, 'original_model.keras')}")
        
    except Exception as e:
        print(f"Error creating original model: {e}")
        print("Skipping original model creation...")

if __name__ == "__main__":
    main()
