"""
Expert Hyperparameter Optimization Runner

This script runs the first phase of PyGMO optimization: Expert Hyperparameters.
It optimizes each expert's architecture independently using Differential Evolution.
"""

import os
import sys
import numpy as np
import tensorflow as tf
import time
from datetime import datetime

# Add the project root to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from model.pygmo_optimizer import PyGMOOptimizer

def main():
    """Run the expert hyperparameter optimization."""
    print("\n=== Migraine Prediction Model - Expert Hyperparameter Optimization ===")
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
    
    # Run expert optimization
    print("\nStarting Expert Hyperparameter Optimization...")
    start_time = time.time()
    
    expert_configs = optimizer.optimize_experts(
        pop_size=10,  # Reduced for faster execution
        generations=5,  # Reduced for faster execution
        algorithm='de'  # Differential Evolution
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nExpert Optimization Complete!")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Optimized expert configurations saved to {os.path.join(output_dir, 'optimization')}")
    
    # Print optimized configurations
    print("\nOptimized Sleep Expert Configuration:")
    print(expert_configs[0])
    
    print("\nOptimized Weather Expert Configuration:")
    print(expert_configs[1])
    
    print("\nOptimized Stress/Diet Expert Configuration:")
    print(expert_configs[2])
    
    return expert_configs

if __name__ == "__main__":
    main()
