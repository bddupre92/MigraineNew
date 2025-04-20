"""
Gating Network Hyperparameter Optimization Runner

This script runs the second phase of PyGMO optimization: Gating Hyperparameters.
It optimizes the gating network with fixed experts using Particle Swarm Optimization.
"""

import os
import sys
import numpy as np
import tensorflow as tf
import time
from datetime import datetime
import json

# Add the project root to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from model.pygmo_optimizer import PyGMOOptimizer

def load_expert_configs():
    """Load the optimized expert configurations from phase 1."""
    optimization_results_path = os.path.join(project_root, 'output', 'optimization', 'optimization_results.json')
    
    if not os.path.exists(optimization_results_path):
        print(f"Error: Could not find optimization results at {optimization_results_path}")
        print("Please run optimize_experts.py first.")
        return None
    
    try:
        with open(optimization_results_path, 'r') as f:
            results = json.load(f)
        
        # Extract expert configurations
        sleep_config = results['expert_phase']['sleep']['config']
        weather_config = results['expert_phase']['weather']['config']
        stress_diet_config = results['expert_phase']['stress_diet']['config']
        
        return [sleep_config, weather_config, stress_diet_config]
    
    except Exception as e:
        print(f"Error loading expert configurations: {e}")
        return None

def main():
    """Run the gating network hyperparameter optimization."""
    print("\n=== Migraine Prediction Model - Gating Network Hyperparameter Optimization ===")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load expert configurations from phase 1
    expert_configs = load_expert_configs()
    
    if expert_configs is None:
        print("Using default expert configurations for testing.")
        # Default configurations for testing
        expert_configs = [
            {
                'conv_filters': 64,
                'kernel_size': 5,
                'lstm_units': 128,
                'dropout_rate': 0.3,
                'output_dim': 64
            },
            {
                'hidden_units': 128,
                'activation': 'relu',
                'dropout_rate': 0.2,
                'output_dim': 64
            },
            {
                'embedding_dim': 64,
                'num_heads': 4,
                'transformer_dim': 64,
                'dropout_rate': 0.2,
                'output_dim': 64
            }
        ]
    
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
    
    # Run gating network optimization
    print("\nStarting Gating Network Hyperparameter Optimization...")
    start_time = time.time()
    
    gating_config = optimizer.optimize_gating(
        expert_configs=expert_configs,
        pop_size=10,  # Reduced for faster execution
        generations=5,  # Reduced for faster execution
        algorithm='pso'  # Particle Swarm Optimization
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nGating Network Optimization Complete!")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Optimized gating configuration saved to {os.path.join(output_dir, 'optimization')}")
    
    # Print optimized configuration
    print("\nOptimized Gating Network Configuration:")
    print(gating_config)
    
    return gating_config

if __name__ == "__main__":
    main()
