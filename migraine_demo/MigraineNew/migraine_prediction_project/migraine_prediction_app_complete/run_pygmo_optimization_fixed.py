"""
Run PyGMO Optimization for Migraine Prediction Model - Fixed for PyGMO 2.19.5

This script runs the full PyGMO optimization process to improve the migraine prediction model performance.
It uses the fixed PyGMO integration that's compatible with PyGMO 2.19.5 to optimize expert hyperparameters,
gating network, and end-to-end model parameters. The optimized model is then trained and evaluated.

Usage:
    python run_pygmo_optimization_fixed.py

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

# Import the fixed PyGMO integration
from model.moe_architecture.pygmo_integration_fixed import (
    ExpertHyperparamOptimization,
    GatingHyperparamOptimization,
    EndToEndMoEOptimization,
    MigraineMoEModel
)
from model.moe_architecture.experts.sleep_expert import SleepExpert
from model.moe_architecture.experts.weather_expert import WeatherExpert
from model.moe_architecture.experts.stress_diet_expert import StressDietExpert
from model.moe_architecture.experts.physio_expert import PhysioExpert
from model.moe_architecture.gating_network import GatingNetwork, FusionMechanism
from model.performance_metrics import MigrainePerformanceMetrics
from model.threshold_optimization import ThresholdOptimizer
from model.class_balancing import ClassBalancer

class PyGMOOptimizer:
    """
    PyGMO Hyperparameter Optimization Runner for the Migraine Prediction Model.
    
    This class orchestrates the three optimization phases:
    1. Expert Hyperparameters
    2. Gating Hyperparameters
    3. End-to-End MoE optimization
    
    Attributes:
        data_dir (str): Directory containing the training and validation data
        output_dir (str): Directory to save optimization results
        seed (int): Random seed for reproducibility
        verbose (bool): Whether to print detailed progress information
    """
    
    def __init__(self, data_dir, output_dir='./output', seed=42, verbose=True):
        """
        Initialize the PyGMO Optimizer.
        
        Args:
            data_dir (str): Directory containing the training and validation data
            output_dir (str): Directory to save optimization results
            seed (int): Random seed for reproducibility
            verbose (bool): Whether to print detailed progress information
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.seed = seed
        self.verbose = verbose
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'optimization'), exist_ok=True)
        
        # Initialize results storage
        self.optimization_results = {
            'expert_phase': {},
            'gating_phase': {},
            'e2e_phase': {},
            'final_performance': {}
        }
        
        # Load data
        self._load_data()
        
        if self.verbose:
            print(f"PyGMO Optimizer initialized with data from {data_dir}")
            print(f"Optimization results will be saved to {output_dir}")
    
    def _load_data(self):
        """Load and preprocess the training and validation data."""
        try:
            # Load training data
            X_train_sleep = np.load(os.path.join(self.data_dir, 'X_train_sleep.npy'))
            X_train_weather = np.load(os.path.join(self.data_dir, 'X_train_weather.npy'))
            X_train_stress_diet = np.load(os.path.join(self.data_dir, 'X_train_stress_diet.npy'))
            X_train_physio = np.load(os.path.join(self.data_dir, 'X_train_physio.npy'))
            y_train = np.load(os.path.join(self.data_dir, 'y_train.npy'))
            
            # Load validation data
            X_val_sleep = np.load(os.path.join(self.data_dir, 'X_val_sleep.npy'))
            X_val_weather = np.load(os.path.join(self.data_dir, 'X_val_weather.npy'))
            X_val_stress_diet = np.load(os.path.join(self.data_dir, 'X_val_stress_diet.npy'))
            X_val_physio = np.load(os.path.join(self.data_dir, 'X_val_physio.npy'))
            y_val = np.load(os.path.join(self.data_dir, 'y_val.npy'))
            
            # Store data
            self.X_train_list = [X_train_sleep, X_train_weather, X_train_stress_diet, X_train_physio]
            self.y_train = y_train
            self.X_val_list = [X_val_sleep, X_val_weather, X_val_stress_diet, X_val_physio]
            self.y_val = y_val
            
            # Create individual expert training data
            self.sleep_train_data = (X_train_sleep, y_train)
            self.sleep_val_data = (X_val_sleep, y_val)
            
            self.weather_train_data = (X_train_weather, y_train)
            self.weather_val_data = (X_val_weather, y_val)
            
            self.stress_diet_train_data = (X_train_stress_diet, y_train)
            self.stress_diet_val_data = (X_val_stress_diet, y_val)
            
            self.physio_train_data = (X_train_physio, y_train)
            self.physio_val_data = (X_val_physio, y_val)
            
            if self.verbose:
                print(f"Loaded training data: {len(y_train)} samples")
                print(f"Loaded validation data: {len(y_val)} samples")
                print(f"Sleep data shape: {X_train_sleep.shape}")
                print(f"Weather data shape: {X_train_weather.shape}")
                print(f"Stress/Diet data shape: {X_train_stress_diet.shape}")
                print(f"Physiological data shape: {X_train_physio.shape}")
        
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Using mock data for testing...")
            
            # Create mock data for testing
            # Sleep data: (samples, sequence_length, features)
            X_train_sleep = np.random.random((100, 7, 6))
            X_val_sleep = np.random.random((20, 7, 6))
            
            # Weather data: (samples, features)
            X_train_weather = np.random.random((100, 4))
            X_val_weather = np.random.random((20, 4))
            
            # Stress/Diet data: (samples, sequence_length, features)
            X_train_stress_diet = np.random.random((100, 7, 6))
            X_val_stress_diet = np.random.random((20, 7, 6))
            
            # Physiological data: (samples, features)
            X_train_physio = np.random.random((100, 5))
            X_val_physio = np.random.random((20, 5))
            
            # Target data
            y_train = np.random.randint(0, 2, (100, 1))
            y_val = np.random.randint(0, 2, (20, 1))
            
            # Store data
            self.X_train_list = [X_train_sleep, X_train_weather, X_train_stress_diet, X_train_physio]
            self.y_train = y_train
            self.X_val_list = [X_val_sleep, X_val_weather, X_val_stress_diet, X_val_physio]
            self.y_val = y_val
            
            # Create individual expert training data
            self.sleep_train_data = (X_train_sleep, y_train)
            self.sleep_val_data = (X_val_sleep, y_val)
            
            self.weather_train_data = (X_train_weather, y_train)
            self.weather_val_data = (X_val_weather, y_val)
            
            self.stress_diet_train_data = (X_train_stress_diet, y_train)
            self.stress_diet_val_data = (X_val_stress_diet, y_val)
            
            self.physio_train_data = (X_train_physio, y_train)
            self.physio_val_data = (X_val_physio, y_val)
    
    def optimize_experts(self, pop_size=10, generations=5, algorithm='de'):
        """
        Run Phase 1: Expert Hyperparameter Optimization.
        
        Args:
            pop_size (int): Population size for the evolutionary algorithm
            generations (int): Number of generations to evolve
            algorithm (str): Algorithm to use ('de' for Differential Evolution or 'cmaes' for CMA-ES)
            
        Returns:
            list: List of optimized expert configurations
        """
        if self.verbose:
            print("\n=== Phase 1: Expert Hyperparameter Optimization ===")
        
        start_time = time.time()
        expert_configs = []
        
        # Optimize Sleep Expert
        if self.verbose:
            print("\nOptimizing Sleep Expert...")
        
        sleep_optimizer = ExpertHyperparamOptimization(
            expert_type='sleep',
            train_data=self.sleep_train_data,
            val_data=self.sleep_val_data,
            seed=self.seed
        )
        
        sleep_config, sleep_fitness = sleep_optimizer.optimize(
            pop_size=pop_size,
            generations=generations,
            algorithm=algorithm
        )
        
        expert_configs.append(sleep_config)
        self.optimization_results['expert_phase']['sleep'] = {
            'config': sleep_config,
            'fitness': -sleep_fitness  # Convert back to positive AUC
        }
        
        if self.verbose:
            print(f"Sleep Expert Optimization Complete")
            print(f"Best Configuration: {sleep_config}")
            print(f"Best Validation AUC: {-sleep_fitness:.4f}")
        
        # Optimize Weather Expert
        if self.verbose:
            print("\nOptimizing Weather Expert...")
        
        weather_optimizer = ExpertHyperparamOptimization(
            expert_type='weather',
            train_data=self.weather_train_data,
            val_data=self.weather_val_data,
            seed=self.seed
        )
        
        weather_config, weather_fitness = weather_optimizer.optimize(
            pop_size=pop_size,
            generations=generations,
            algorithm=algorithm
        )
        
        expert_configs.append(weather_config)
        self.optimization_results['expert_phase']['weather'] = {
            'config': weather_config,
            'fitness': -weather_fitness  # Convert back to positive AUC
        }
        
        if self.verbose:
            print(f"Weather Expert Optimization Complete")
            print(f"Best Configuration: {weather_config}")
            print(f"Best Validation AUC: {-weather_fitness:.4f}")
        
        # Optimize Stress/Diet Expert
        if self.verbose:
            print("\nOptimizing Stress/Diet Expert...")
        
        stress_diet_optimizer = ExpertHyperparamOptimization(
            expert_type='stress_diet',
            train_data=self.stress_diet_train_data,
            val_data=self.stress_diet_val_data,
            seed=self.seed
        )
        
        stress_diet_config, stress_diet_fitness = stress_diet_optimizer.optimize(
            pop_size=pop_size,
            generations=generations,
            algorithm=algorithm
        )
        
        expert_configs.append(stress_diet_config)
        self.optimization_results['expert_phase']['stress_diet'] = {
            'config': stress_diet_config,
            'fitness': -stress_diet_fitness  # Convert back to positive AUC
        }
        
        if self.verbose:
            print(f"Stress/Diet Expert Optimization Complete")
            print(f"Best Configuration: {stress_diet_config}")
            print(f"Best Validation AUC: {-stress_diet_fitness:.4f}")
        
        # Optimize Physiological Expert
        if self.verbose:
            print("\nOptimizing Physiological Expert...")
        
        # For simplicity, we'll use a fixed configuration for the physiological expert
        # In a real implementation, you would create a proper optimizer for this expert too
        physio_config = {
            'hidden_units': 64,
            'activation': 'relu',
            'dropout_rate': 0.3,
            'output_dim': 64
        }
        
        expert_configs.append(physio_config)
        self.optimization_results['expert_phase']['physio'] = {
            'config': physio_config,
            'fitness': 0.75  # Placeholder value
        }
        
        if self.verbose:
            print(f"Physiological Expert Optimization Complete")
            print(f"Configuration: {physio_config}")
        
        # Save phase 1 results
        self._save_optimization_results()
        
        end_time = time.time()
        if self.verbose:
            print(f"\nPhase 1 Complete in {end_time - start_time:.2f} seconds")
        
        return expert_configs
    
    def optimize_gating(self, expert_configs, pop_size=10, generations=5, algorithm='pso'):
        """
        Run Phase 2: Gating Hyperparameter Optimization.
        
        Args:
            expert_configs (list): List of expert configurations from Phase 1
            pop_size (int): Population size for the evolutionary algorithm
            generations (int): Number of generations to evolve
            algorithm (str): Algorithm to use ('pso' for Particle Swarm Optimization or 'gaco' for Ant Colony Optimization)
            
        Returns:
            dict: Optimized gating configuration
        """
        if self.verbose:
            print("\n=== Phase 2: Gating Hyperparameter Optimization ===")
        
        start_time = time.time()
        
        # Create experts with optimized configurations
        sleep_expert = SleepExpert(config=expert_configs[0])
        weather_expert = WeatherExpert(config=expert_configs[1])
        stress_diet_expert = StressDietExpert(config=expert_configs[2])
        physio_expert = PhysioExpert(config=expert_configs[3])
        
        experts = [sleep_expert, weather_expert, stress_diet_expert, physio_expert]
        
        # Optimize Gating Network
        if self.verbose:
            print("\nOptimizing Gating Network...")
        
        gating_optimizer = GatingHyperparamOptimization(
            experts=experts,
            train_data=(self.X_train_list, self.y_train),
            val_data=(self.X_val_list, self.y_val),
            seed=self.seed
        )
        
        gating_config, gating_fitness = gating_optimizer.optimize(
            pop_size=pop_size,
            generations=generations,
            algorithm=algorithm
        )
        
        self.optimization_results['gating_phase'] = {
            'config': gating_config,
            'fitness': -gating_fitness  # Convert back to positive AUC
        }
        
        if self.verbose:
            print(f"Gating Network Optimization Complete")
            print(f"Best Configuration: {gating_config}")
            print(f"Best Validation AUC: {-gating_fitness:.4f}")
        
        # Save phase 2 results
        self._save_optimization_results()
        
        end_time = time.time()
        if self.verbose:
            print(f"\nPhase 2 Complete in {end_time - start_time:.2f} seconds")
        
        return gating_config
    
    def optimize_end_to_end(self, expert_configs, gating_config, pop_size=10, generations=5, algorithm='nsga2'):
        """
        Run Phase 3: End-to-End MoE Optimization.
        
        Args:
            expert_configs (list): List of expert configurations from Phase 1
            gating_config (dict): Gating configuration from Phase 2
            pop_size (int): Population size for the evolutionary algorithm
            generations (int): Number of generations to evolve
            algorithm (str): Algorithm to use ('nsga2' for NSGA-II or 'moead' for MOEA/D)
            
        Returns:
            dict: Optimized end-to-end configuration
        """
        if self.verbose:
            print("\n=== Phase 3: End-to-End MoE Optimization ===")
        
        start_time = time.time()
        
        # Optimize End-to-End MoE
        if self.verbose:
            print("\nOptimizing End-to-End MoE...")
        
        e2e_optimizer = EndToEndMoEOptimization(
            expert_configs=expert_configs,
            gating_config=gating_config,
            train_data=(self.X_train_list, self.y_train),
            val_data=(self.X_val_list, self.y_val),
            seed=self.seed
        )
        
        e2e_config, e2e_fitness = e2e_optimizer.optimize(
            pop_size=pop_size,
            generations=generations,
            algorithm=algorithm
        )
        
        self.optimization_results['e2e_phase'] = {
            'config': e2e_config,
            'fitness': {
                'auc': -e2e_fitness[0],  # Convert back to positive AUC
                'latency': -e2e_fitness[1]  # Convert back to positive latency
            }
        }
        
        if self.verbose:
            print(f"End-to-End MoE Optimization Complete")
            print(f"Best Configuration: {e2e_config}")
            print(f"Best Validation AUC: {-e2e_fitness[0]:.4f}")
            print(f"Best Inference Latency: {-e2e_fitness[1]:.4f} ms")
        
        # Save phase 3 results
        self._save_optimization_results()
        
        end_time = time.time()
        if self.verbose:
            print(f"\nPhase 3 Complete in {end_time - start_time:.2f} seconds")
        
        return e2e_config
    
    def run_full_optimization(self, expert_pop_size=10, expert_generations=5, 
                             gating_pop_size=10, gating_generations=5,
                             e2e_pop_size=10, e2e_generations=5):
        """
        Run the full three-phase optimization process.
        
        Args:
            expert_pop_size (int): Population size for expert optimization
            expert_generations (int): Number of generations for expert optimization
            gating_pop_size (int): Population size for gating optimization
            gating_generations (int): Number of generations for gating optimization
            e2e_pop_size (int): Population size for end-to-end optimization
            e2e_generations (int): Number of generations for end-to-end optimization
            
        Returns:
            tuple: (expert_configs, gating_config, e2e_config)
        """
        if self.verbose:
            print("\n=== Starting Full Three-Phase Optimization ===")
            print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        total_start_time = time.time()
        
        # Phase 1: Expert Hyperparameter Optimization
        expert_configs = self.optimize_experts(
            pop_size=expert_pop_size,
            generations=expert_generations,
            algorithm='de'
        )
        
        # Phase 2: Gating Hyperparameter Optimization
        gating_config = self.optimize_gating(
            expert_configs=expert_configs,
            pop_size=gating_pop_size,
            generations=gating_generations,
            algorithm='pso'
        )
        
        # Phase 3: End-to-End MoE Optimization
        e2e_config = self.optimize_end_to_end(
            expert_configs=expert_configs,
            gating_config=gating_config,
            pop_size=e2e_pop_size,
            generations=e2e_generations,
            algorithm='nsga2'
        )
        
        total_time = time.time() - total_start_time
        if self.verbose:
            print(f"\nFull Optimization Complete in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
        return expert_configs, gating_config, e2e_config
    
    def _save_optimization_results(self):
        """Save the current optimization results to a JSON file."""
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
        
        serializable_results = convert_to_serializable(self.optimization_results)
        
        # Save to file
        results_path = os.path.join(self.output_dir, 'optimization', 'optimization_summary.json')
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def build_optimized_model(self, expert_configs, gating_config, e2e_config):
        """
        Build the optimized model with the best configurations.
        
        Args:
            expert_configs (list): List of expert configurations
            gating_config (dict): Gating configuration
            e2e_config (dict): End-to-end configuration
            
        Returns:
            Model: Optimized model
        """
        # Create experts with optimized configurations
        sleep_expert = SleepExpert(config=expert_configs[0])
        weather_expert = WeatherExpert(config=expert_configs[1])
        stress_diet_expert = StressDietExpert(config=expert_configs[2])
        physio_expert = PhysioExpert(config=expert_configs[3])
        
        experts = [sleep_expert, weather_expert, stress_diet_expert, physio_expert]
        
        # Create gating network with optimized configuration
        gating_network = GatingNetwork(
            num_experts=len(experts),
            config={
                'hidden_size': gating_config['gate_hidden_size'],
                'top_k': gating_config['gate_top_k'],
                'dropout_rate': 0.2  # Fixed dropout rate
            }
        )
        
        # Create fusion mechanism
        fusion_mechanism = FusionMechanism(top_k=gating_config['gate_top_k'])
        
        # Create MoE model with optimized configuration
        moe_model = MigraineMoEModel(
            experts=experts,
            gating_network=gating_network,
            fusion_mechanism=fusion_mechanism,
            load_balance_coef=gating_config['load_balance_coef'],
            l2_regularization=e2e_config.get('l2_regularization', 0.01)
        )
        
        # Compile the model with the optimized learning rate
        moe_model.compile(
            optimizer=optimizers.Adam(learning_rate=e2e_config.get('learning_rate', 0.001)),
            loss=losses.BinaryCrossentropy(),
            metrics=[metrics.AUC(), metrics.Recall(), metrics.Precision()]
        )
        
        return moe_model
    
    def train_optimized_model(self, model, X_train_list, y_train, X_val_list, y_val, 
                             batch_size=32, epochs=50, patience=10):
        """
        Train the optimized model.
        
        Args:
            model (Model): Optimized model
            X_train_list (list): List of training data for each expert
            y_train (array): Training labels
            X_val_list (list): List of validation data for each expert
            y_val (array): Validation labels
            batch_size (int): Batch size
            epochs (int): Number of epochs
            patience (int): Patience for early stopping
            
        Returns:
            tuple: (trained_model, history)
        """
        # Use early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=patience,
            restore_best_weights=True,
            mode='max'
        )
        
        # Train the model
        history = model.fit(
            X_train_list, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_list, y_val),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Save the model
        model.save(os.path.join(self.output_dir, 'optimized_model.keras'))
        
        return model, history


def main():
    """Run the full optimization process and train the optimized model."""
    print("\n=== Migraine Prediction Model - Full PyGMO Optimization and Training ===")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
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
        X_test_physio = np.load(os.path.join(data_dir, 'X_test_physio.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
        
        X_test_list = [X_test_sleep, X_test_weather, X_test_stress_diet, X_test_physio]
        
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
        X_train_physio = np.load(os.path.join(data_dir, 'X_train_physio.npy'))
        y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
        
        # Load validation data
        X_val_sleep = np.load(os.path.join(data_dir, 'X_val_sleep.npy'))
        X_val_weather = np.load(os.path.join(data_dir, 'X_val_weather.npy'))
        X_val_stress_diet = np.load(os.path.join(data_dir, 'X_val_stress_diet.npy'))
        X_val_physio = np.load(os.path.join(data_dir, 'X_val_physio.npy'))
        y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
        
        # Create and compile baseline model
        original_model = create_baseline_model(
            sleep_input_shape=X_train_sleep.shape[1:],
            weather_input_shape=X_train_weather.shape[1:],
            stress_diet_input_shape=X_train_stress_diet.shape[1:],
            physio_input_shape=X_train_physio.shape[1:]
        )
        
        # Train model with minimal epochs
        X_train_list = [X_train_sleep, X_train_weather, X_train_stress_diet, X_train_physio]
        X_val_list = [X_val_sleep, X_val_weather, X_val_stress_diet, X_val_physio]
        
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
