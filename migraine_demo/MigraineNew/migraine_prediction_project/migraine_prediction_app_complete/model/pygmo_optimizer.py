"""
PyGMO Hyperparameter Optimization Runner

This module implements the runner for PyGMO hyperparameter optimization of the MoE architecture.
It orchestrates the three optimization phases and provides a unified interface for model optimization.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import pygmo as pg
import time
import json
from datetime import datetime

from model.moe_architecture.pygmo_integration import (
    ExpertHyperparamOptimization,
    GatingHyperparamOptimization,
    EndToEndMoEOptimization
)
from model.moe_architecture.experts.sleep_expert import SleepExpert
from model.moe_architecture.experts.weather_expert import WeatherExpert
from model.moe_architecture.experts.stress_diet_expert import StressDietExpert
from model.moe_architecture.gating_network import GatingNetwork, FusionMechanism
from model.input_preprocessing import preprocess_expert_inputs

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
            y_train = np.load(os.path.join(self.data_dir, 'y_train.npy'))
            
            # Load validation data
            X_val_sleep = np.load(os.path.join(self.data_dir, 'X_val_sleep.npy'))
            X_val_weather = np.load(os.path.join(self.data_dir, 'X_val_weather.npy'))
            X_val_stress_diet = np.load(os.path.join(self.data_dir, 'X_val_stress_diet.npy'))
            y_val = np.load(os.path.join(self.data_dir, 'y_val.npy'))
            
            # Store data
            self.X_train_list = [X_train_sleep, X_train_weather, X_train_stress_diet]
            self.y_train = y_train
            self.X_val_list = [X_val_sleep, X_val_weather, X_val_stress_diet]
            self.y_val = y_val
            
            # Create individual expert training data
            self.sleep_train_data = (X_train_sleep, y_train)
            self.sleep_val_data = (X_val_sleep, y_val)
            
            self.weather_train_data = (X_train_weather, y_train)
            self.weather_val_data = (X_val_weather, y_val)
            
            self.stress_diet_train_data = (X_train_stress_diet, y_train)
            self.stress_diet_val_data = (X_val_stress_diet, y_val)
            
            if self.verbose:
                print(f"Loaded training data: {len(y_train)} samples")
                print(f"Loaded validation data: {len(y_val)} samples")
                print(f"Sleep data shape: {X_train_sleep.shape}")
                print(f"Weather data shape: {X_train_weather.shape}")
                print(f"Stress/Diet data shape: {X_train_stress_diet.shape}")
        
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
            
            # Target data
            y_train = np.random.randint(0, 2, (100, 1))
            y_val = np.random.randint(0, 2, (20, 1))
            
            # Store data
            self.X_train_list = [X_train_sleep, X_train_weather, X_train_stress_diet]
            self.y_train = y_train
            self.X_val_list = [X_val_sleep, X_val_weather, X_val_stress_diet]
            self.y_val = y_val
            
            # Create individual expert training data
            self.sleep_train_data = (X_train_sleep, y_train)
            self.sleep_val_data = (X_val_sleep, y_val)
            
            self.weather_train_data = (X_train_weather, y_train)
            self.weather_val_data = (X_val_weather, y_val)
            
            self.stress_diet_train_data = (X_train_stress_diet, y_train)
            self.stress_diet_val_data = (X_val_stress_diet, y_val)
    
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
        
        experts = [sleep_expert, weather_expert, stress_diet_expert]
        
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
        
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        
        if self.verbose:
            print("\n=== Full Optimization Complete ===")
            print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
            print(f"Final results saved to {os.path.join(self.output_dir, 'optimization')}")
        
        return expert_configs, gating_config, e2e_config
    
    def build_optimized_model(self, expert_configs, gating_config, e2e_config):
        """
        Build the optimized model with the best configurations from all phases.
        
        Args:
            expert_configs (list): List of expert configurations from Phase 1
            gating_config (dict): Gating configuration from Phase 2
            e2e_config (dict): End-to-end configuration from Phase 3
            
        Returns:
            Model: Optimized MoE model
        """
        from model.moe_architecture.pygmo_integration import MigraineMoEModel
        
        # Create experts with optimized configurations
        sleep_expert = SleepExpert(config=expert_configs[0])
        weather_expert = WeatherExpert(config=expert_configs[1])
        stress_diet_expert = StressDietExpert(config=expert_configs[2])
        
        experts = [sleep_expert, weather_expert, stress_diet_expert]
        
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
        
        # Create MoE model
        moe_model = MigraineMoEModel(
            experts=experts,
            gating_network=gating_network,
            fusion_mechanism=fusion_mechanism,
            load_balance_coef=gating_config['load_balance_coef']
        )
        
        # Compile the model with optimized learning rate and regularization
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=e2e_config['learning_rate'],
            clipnorm=1.0
        )
        
        moe_model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.Precision(name='precision')
            ]
        )
        
        return moe_model
    
    def train_optimized_model(self, model, batch_size=32, epochs=50, patience=10):
        """
        Train the optimized model with the best configurations.
        
        Args:
            model: Optimized MoE model
            batch_size (int): Batch size for training
            epochs (int): Maximum number of epochs to train
            patience (int): Patience for early stopping
            
        Returns:
            tuple: (trained_model, history)
        """
        if self.verbose:
            print("\n=== Training Optimized Model ===")
        
        # Create callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=patience,
            restore_best_weights=True,
            mode='max'
        )
        
        # Train the model
        history = model.fit(
            self.X_train_list, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_val_list, self.y_val),
            callbacks=[early_stopping],
            verbose=1 if self.verbose else 0
        )
        
        # Evaluate on validation set
        val_loss, val_auc, val_recall, val_precision = model.evaluate(
            self.X_val_list, self.y_val,
            verbose=0
        )
        
        # Calculate F1 score
        val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0
        
        # Store final performance
        self.optimization_results['final_performance'] = {
            'val_loss': float(val_loss),
            'val_auc': float(val_auc),
            'val_recall': float(val_recall),
            'val_precision': float(val_precision),
            'val_f1': float(val_f1)
        }
        
        if self.verbose:
            print(f"\nFinal Validation Performance:")
            print(f"AUC: {val_auc:.4f}")
            print(f"Recall: {val_recall:.4f}")
            print(f"Precision: {val_precision:.4f}")
            print(f"F1 Score: {val_f1:.4f}")
        
        # Save final results
        self._save_optimization_results()
        
        # Save the model
        model_path = os.path.join(self.output_dir, 'optimized_model.keras')
        model.save(model_path)
        
        if self.verbose:
            print(f"Optimized model saved to {model_path}")
        
        return model, history
    
    def _save_optimization_results(self):
        """Save the optimization results to a JSON file."""
        results_path = os.path.join(self.output_dir, 'optimization', 'optimization_results.json')
        
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
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
