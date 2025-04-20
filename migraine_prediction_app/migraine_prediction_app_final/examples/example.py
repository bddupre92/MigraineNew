"""
Example script for using the Migraine Prediction App.

This script demonstrates how to:
1. Generate synthetic data
2. Train an optimized model
3. Make predictions
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add code directory to path
sys.path.append('../code')

# Import modules
from data_generator.synthetic_data_generator import SyntheticDataGenerator
from optimized_model import OptimizedMigrainePredictionModel

def main():
    # Create directories
    os.makedirs('./data', exist_ok=True)
    os.makedirs('./output', exist_ok=True)
    
    # Step 1: Generate synthetic data
    print("Generating synthetic data...")
    data_generator = SyntheticDataGenerator(
        num_patients=50,  # Small number for example
        days=30,          # 1 month of data
        output_dir='./data',
        seed=42
    )
    data_generator.generate_data()
    
    # Step 2: Train optimized model with reduced settings for example
    print("Training optimized model...")
    optimizer = OptimizedMigrainePredictionModel(
        data_dir='./data',
        output_dir='./output',
        seed=42
    )
    
    # Modify config for faster example run
    optimizer.config['epochs'] = 5
    optimizer.config['expert_pop_size'] = 5
    optimizer.config['expert_generations'] = 2
    optimizer.config['gating_pop_size'] = 5
    optimizer.config['gating_generations'] = 2
    
    # Run simplified optimization (just train one model)
    model, history, test_metrics = optimizer.train_optimized_model()
    
    # Step 3: Print results
    print("\nModel Performance:")
    print(f"AUC: {test_metrics['roc_auc']:.4f}")
    print(f"F1 Score: {test_metrics['f1_score']:.4f}")
    print(f"High-Risk Sensitivity: {test_metrics['high_risk_sensitivity']:.4f}")
    print(f"Inference Time: {test_metrics['inference_time_ms']:.2f} ms")
    
    # Step 4: Make a prediction with random data
    print("\nMaking example prediction...")
    # Create example input data
    sleep_data = np.random.randn(1, 7, 6)  # 7-day sequence, 6 features
    weather_data = np.random.randn(1, 4)   # 4 features
    stress_diet_data = np.random.randn(1, 7, 6)  # 7-day sequence, 6 features
    
    # Make prediction
    prediction = model.predict([sleep_data, weather_data, stress_diet_data])
    print(f"Migraine probability: {prediction[0][0]:.4f}")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()
