"""
End-to-End MoE Hyperparameter Optimization Runner

This script runs the third phase of PyGMO optimization: End-to-End MoE Optimization.
It fine-tunes the entire model with the best configurations from previous phases.
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

def load_previous_configs():
    """Load the optimized configurations from phases 1 and 2."""
    optimization_results_path = os.path.join(project_root, 'output', 'optimization', 'optimization_results.json')
    
    if not os.path.exists(optimization_results_path):
        print(f"Error: Could not find optimization results at {optimization_results_path}")
        print("Please run optimize_experts.py and optimize_gating.py first.")
        return None, None
    
    try:
        with open(optimization_results_path, 'r') as f:
            results = json.load(f)
        
        # Check if both phases are complete
        if 'expert_phase' not in results or 'gating_phase' not in results:
            print("Error: Previous optimization phases are incomplete.")
            print("Please run optimize_experts.py and optimize_gating.py first.")
            return None, None
        
        # Extract expert configurations
        sleep_config = results['expert_phase']['sleep']['config']
        weather_config = results['expert_phase']['weather']['config']
        stress_diet_config = results['expert_phase']['stress_diet']['config']
        expert_configs = [sleep_config, weather_config, stress_diet_config]
        
        # Extract gating configuration
        gating_config = results['gating_phase']['config']
        
        return expert_configs, gating_config
    
    except Exception as e:
        print(f"Error loading previous configurations: {e}")
        return None, None

def main():
    """Run the end-to-end MoE hyperparameter optimization."""
    print("\n=== Migraine Prediction Model - End-to-End MoE Optimization ===")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load configurations from previous phases
    expert_configs, gating_config = load_previous_configs()
    
    if expert_configs is None or gating_config is None:
        print("Using default configurations for testing.")
        # Default expert configurations for testing
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
        
        # Default gating configuration for testing
        gating_config = {
            'gate_hidden_size': 128,
            'gate_top_k': 2,
            'load_balance_coef': 0.01
        }
    
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
    
    # Run end-to-end optimization
    print("\nStarting End-to-End MoE Optimization...")
    start_time = time.time()
    
    e2e_config = optimizer.optimize_end_to_end(
        expert_configs=expert_configs,
        gating_config=gating_config,
        pop_size=10,  # Reduced for faster execution
        generations=5,  # Reduced for faster execution
        algorithm='nsga2'  # NSGA-II for multi-objective optimization
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nEnd-to-End MoE Optimization Complete!")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Optimized end-to-end configuration saved to {os.path.join(output_dir, 'optimization')}")
    
    # Print optimized configuration
    print("\nOptimized End-to-End Configuration:")
    print(e2e_config)
    
    return e2e_config

if __name__ == "__main__":
    main()
