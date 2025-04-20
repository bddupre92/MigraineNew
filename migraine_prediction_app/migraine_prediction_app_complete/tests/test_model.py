"""
Test Script for Migraine Prediction Model

This script tests the migraine prediction model using the implemented performance metrics.
It generates synthetic data, trains the model, and evaluates its performance.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Import our modules
from data_generator.synthetic_data_generator import SyntheticDataGenerator
from migraine_prediction_model import MigrainePredictionModel
from performance_metrics import MigrainePerformanceMetrics, PerformanceMetricsCallback

def main():
    """
    Main function to test the migraine prediction model.
    """
    print("Starting migraine prediction model testing...")
    
    # Set random seed for reproducibility
    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Create directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    output_dir = os.path.join(base_dir, 'output')
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Generate synthetic data if it doesn't exist
    if not os.path.exists(os.path.join(data_dir, 'combined_data.csv')):
        print("\n=== Generating Synthetic Data ===")
        data_generator = SyntheticDataGenerator(
            num_patients=200,  # Smaller dataset for testing
            days=180,          # 6 months of data
            output_dir=data_dir,
            seed=seed
        )
        data_generator.generate_data()
    
    # Step 2: Create and train the migraine prediction model
    print("\n=== Training Migraine Prediction Model ===")
    
    # Configuration for testing (reduced optimization to speed up testing)
    test_config = {
        'test_size': 0.2,
        'val_size': 0.2,
        'sequence_length': 7,
        'batch_size': 32,
        'epochs': 20,
        'learning_rate': 0.001,
        'early_stopping_patience': 5,
        'load_balance_coef': 0.01,
        'optimize_experts': True,
        'optimize_gating': True,
        'optimize_end_to_end': True,
        'expert_pop_size': 10,
        'expert_generations': 3,
        'gating_pop_size': 10,
        'gating_generations': 3,
        'e2e_pop_size': 10,
        'e2e_generations': 3
    }
    
    # Create the model
    model = MigrainePredictionModel(
        data_dir=data_dir,
        output_dir=output_dir,
        config=test_config,
        seed=seed
    )
    
    # Load data
    X_train_list, y_train, X_val_list, y_val, X_test_list, y_test = model.load_data()
    
    # Create performance metrics tracker
    metrics_tracker = MigrainePerformanceMetrics(
        output_dir=output_dir,
        config={
            'high_risk_threshold': 0.7,
            'target_sensitivity': 0.95,
            'target_auc': 0.80,
            'target_f1': 0.75,
            'target_latency_ms': 200
        }
    )
    
    # Step 3: Optimize experts
    if test_config['optimize_experts']:
        expert_configs = model.optimize_experts(X_train_list, y_train, X_val_list, y_val)
    else:
        # Use default configurations
        expert_configs = [
            {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 128, 'dropout_rate': 0.3, 'output_dim': 64},
            {'hidden_units': 128, 'activation': 'relu', 'dropout_rate': 0.3, 'output_dim': 64},
            {'embedding_dim': 64, 'num_heads': 4, 'transformer_dim': 64, 'dropout_rate': 0.2, 'output_dim': 64}
        ]
    
    # Step 4: Optimize gating
    if test_config['optimize_gating']:
        gating_config = model.optimize_gating(expert_configs, X_train_list, y_train, X_val_list, y_val)
    else:
        # Use default configuration
        gating_config = {
            'gate_hidden_size': 128,
            'gate_top_k': 2,
            'load_balance_coef': 0.01
        }
    
    # Step 5: Optimize end-to-end
    if test_config['optimize_end_to_end']:
        e2e_config = model.optimize_end_to_end(expert_configs, gating_config, X_train_list, y_train, X_val_list, y_val)
    else:
        e2e_config = None
    
    # Step 6: Build model
    model.build_model(expert_configs, gating_config, e2e_config)
    
    # Step 7: Create performance metrics callback
    metrics_callback = PerformanceMetricsCallback(
        metrics_tracker=metrics_tracker,
        validation_data=(X_val_list, y_val),
        log_dir=os.path.join(output_dir, 'logs')
    )
    
    # Step 8: Train model with performance metrics callback
    # Create callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        patience=test_config['early_stopping_patience'],
        restore_best_weights=True,
        mode='max'
    )
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(output_dir, 'best_model.h5'),
        monitor='val_auc',
        save_best_only=True,
        mode='max'
    )
    
    # Train the model
    history = model.model.fit(
        X_train_list, y_train,
        epochs=test_config['epochs'],
        batch_size=test_config['batch_size'],
        validation_data=(X_val_list, y_val),
        callbacks=[early_stopping, model_checkpoint, metrics_callback],
        verbose=1
    )
    
    # Step 9: Evaluate model on test set
    print("\n=== Evaluating Model on Test Set ===")
    test_metrics = metrics_tracker.calculate_metrics(model.model, X_test_list, y_test)
    
    # Step 10: Print final performance results
    print("\n=== Final Performance Results ===")
    print(f"ROC AUC: {test_metrics['roc_auc']:.4f} (Target: {metrics_tracker.config['target_auc']})")
    print(f"F1 Score: {test_metrics['f1_score']:.4f} (Target: {metrics_tracker.config['target_f1']})")
    print(f"High-Risk Sensitivity: {test_metrics['high_risk_sensitivity']:.4f} (Target: {metrics_tracker.config['target_sensitivity']})")
    print(f"Inference Time: {test_metrics['inference_time_ms']:.2f} ms (Target: {metrics_tracker.config['target_latency_ms']} ms)")
    print(f"Overall Performance Score: {test_metrics['performance_score']:.1f}%")
    print(f"Target Met: {'Yes' if test_metrics['overall_target_met'] else 'No'}")
    
    # Step 11: Save model
    model.save_model()
    
    print("\nTesting completed. Results saved to:", output_dir)
    
    return test_metrics

if __name__ == "__main__":
    main()
