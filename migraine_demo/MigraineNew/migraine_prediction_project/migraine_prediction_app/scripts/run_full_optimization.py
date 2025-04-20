"""
Run Full PyGMO Optimization and Train Optimized Model

This script runs all three phases of PyGMO optimization and trains the final optimized model.
It provides a complete end-to-end solution for optimizing and training the migraine prediction model.
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
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from model.pygmo_optimizer import PyGMOOptimizer

def main():
    """Run the full optimization process and train the optimized model."""
    print("\n=== Migraine Prediction Model - Full PyGMO Optimization and Training ===")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set paths
    data_dir = os.path.join(project_root, 'data')
    output_dir = os.path.join(project_root, 'output')
    
    # Create optimizer
    optimizer = PyGMOOptimizer(
        data_dir=data_dir,
        output_dir=output_dir,
        seed=42,
        verbose=True
    )
    
    # Run full optimization
    print("\nStarting Full Three-Phase Optimization...")
    start_time = time.time()
    
    expert_configs, gating_config, e2e_config = optimizer.run_full_optimization(
        expert_pop_size=10,
        expert_generations=5,
        gating_pop_size=10,
        gating_generations=5,
        e2e_pop_size=10,
        e2e_generations=5
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
    
    # Train optimized model
    print("\nTraining Optimized Model...")
    training_start_time = time.time()
    
    model, history = optimizer.train_optimized_model(
        model=optimized_model,
        batch_size=int(e2e_config.get('batch_size', 32)),
        epochs=50,
        patience=10
    )
    
    training_time = time.time() - training_start_time
    print(f"\nTraining Complete in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Plot training history
    plot_training_history(history, output_dir)
    
    # Save optimization and training summary
    save_summary(optimizer, expert_configs, gating_config, e2e_config, 
                 optimization_time, training_time, output_dir)
    
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
    
    # Plot Precision and Recall
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
            'auc_improvement': final_performance.get('val_auc', 0) - 0.5625,  # Compared to original 0.5625
            'f1_improvement': final_performance.get('val_f1', 0) - 0.0741,    # Compared to original 0.0741
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

if __name__ == "__main__":
    main()
