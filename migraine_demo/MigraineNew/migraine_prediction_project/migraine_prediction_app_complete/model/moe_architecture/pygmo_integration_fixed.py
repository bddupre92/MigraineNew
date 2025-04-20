"""
PyGMO Integration for Migraine Prediction App - Fixed for PyGMO 2.19.5

This module integrates PyGMO for hyperparameter optimization of the MoE architecture.
It implements the three optimization phases: Expert Hyperparameters, Gating Hyperparameters,
and End-to-End MoE optimization.

This version is compatible with PyGMO 2.19.5 by using proper problem classes instead of fitness_wrapper.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import pygmo as pg
from tensorflow.keras import layers, Model, optimizers, losses, metrics
import time
from sklearn.metrics import roc_auc_score

# Use relative imports for modules within the same package (moe_architecture)
from .experts.sleep_expert import SleepExpert
from .experts.weather_expert import WeatherExpert
from .experts.stress_diet_expert import StressDietExpert
from .gating_network import GatingNetwork, FusionMechanism

# Define proper problem classes for PyGMO 2.19.5
class ExpertOptimizationProblem:
    """
    PyGMO-compatible problem class for expert hyperparameter optimization.
    """
    
    def __init__(self, expert_type, train_data, val_data, seed=None):
        """
        Initialize the Expert Optimization Problem.
        
        Args:
            expert_type (str): Type of expert to optimize ('sleep', 'weather', or 'stress_diet')
            train_data (tuple): Training data for the expert (X_train, y_train)
            val_data (tuple): Validation data for the expert (X_val, y_val)
            seed (int): Random seed for reproducibility
        """
        self.expert_type = expert_type
        self.train_data = train_data
        self.val_data = val_data
        self.seed = seed
        
        # Set up hyperparameter search space based on expert type
        if expert_type == 'sleep':
            self.param_bounds = self._get_sleep_expert_bounds()
            self.integer_dims = 3  # conv_filters, kernel_size, lstm_units are integers
        elif expert_type == 'weather':
            self.param_bounds = self._get_weather_expert_bounds()
            self.integer_dims = 1  # hidden_units is integer
        elif expert_type == 'stress_diet':
            self.param_bounds = self._get_stress_diet_expert_bounds()
            self.integer_dims = 3  # embedding_dim, num_heads, transformer_dim are integers
        else:
            raise ValueError(f"Unknown expert type: {expert_type}")
    
    def _get_sleep_expert_bounds(self):
        """Get hyperparameter bounds for Sleep Expert."""
        # Define bounds for each hyperparameter as two lists: [lower_bounds], [upper_bounds]
        # Ensure integer bounds are integers
        lb = [16, 3, 32, 0.1]  # First 3 are integers
        ub = [128, 7, 256, 0.5]
        return (lb, ub)
    
    def _get_weather_expert_bounds(self):
        """Get hyperparameter bounds for Weather Expert."""
        # Define bounds for each hyperparameter as two lists: [lower_bounds], [upper_bounds]
        # Ensure integer bounds are integers
        lb = [32, 0.0, 0.1]  # First 1 is integer
        ub = [256, 1.0, 0.5]
        return (lb, ub)
    
    def _get_stress_diet_expert_bounds(self):
        """Get hyperparameter bounds for Stress/Diet Expert."""
        # Define bounds for each hyperparameter as two lists: [lower_bounds], [upper_bounds]
        # Ensure integer bounds are integers
        lb = [32, 2, 32, 0.1]  # First 3 are integers
        ub = [128, 8, 128, 0.4]
        return (lb, ub)
    
    def get_bounds(self):
        """Return the bounds of the problem."""
        return self.param_bounds
    
    def get_nobj(self):
        """Return the number of objectives."""
        return 1  # Single objective optimization
    
    def get_nix(self):
        """Return the number of integer dimensions."""
        return self.integer_dims
    
    def get_nec(self):
        """Return the number of equality constraints."""
        return 0
    
    def get_nic(self):
        """Return the number of inequality constraints."""
        return 0
    
    def fitness(self, x):
        """
        Fitness function for PyGMO optimization.
        
        Args:
            x (list): List of hyperparameter values
            
        Returns:
            float: Negative validation accuracy (to be minimized)
        """
        # Convert continuous parameters to discrete where needed
        params = self._process_params(x)
        
        # Create expert model with the given hyperparameters
        expert = self._create_expert(params)
        
        # Compile the model
        expert.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss=losses.BinaryCrossentropy(),
            metrics=[metrics.AUC(), metrics.Recall(), metrics.Precision()]
        )
        
        # Train the model
        X_train, y_train = self.train_data
        X_val, y_val = self.val_data
        
        # Use early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=5,
            restore_best_weights=True,
            mode='max'
        )
        
        # Train with a small number of epochs for optimization speed
        history = expert.fit(
            X_train, y_train,
            epochs=10,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Get the best validation AUC
        best_val_auc = max(history.history['val_auc'])
        
        # Return negative AUC (since PyGMO minimizes)
        return [-best_val_auc]
    
    def _process_params(self, x):
        """
        Process hyperparameters from continuous to discrete values where needed.
        
        Args:
            x (list): List of continuous hyperparameter values
            
        Returns:
            dict: Dictionary of processed hyperparameters
        """
        if self.expert_type == 'sleep':
            return {
                'conv_filters': int(x[0]),
                'kernel_size': int(x[1]),
                'lstm_units': int(x[2]),
                'dropout_rate': x[3],
                'output_dim': 64  # Fixed output dimension
            }
        elif self.expert_type == 'weather':
            return {
                'hidden_units': int(x[0]),
                'activation': 'relu' if x[1] < 0.5 else 'elu',
                'dropout_rate': x[2],
                'output_dim': 64  # Fixed output dimension
            }
        elif self.expert_type == 'stress_diet':
            return {
                'embedding_dim': int(x[0]),
                'num_heads': int(x[1]),
                'transformer_dim': int(x[2]),
                'dropout_rate': x[3],
                'output_dim': 64  # Fixed output dimension
            }
    
    def _create_expert(self, params):
        """
        Create expert model with the given hyperparameters.
        
        Args:
            params (dict): Dictionary of hyperparameters
            
        Returns:
            Model: Expert model
        """
        if self.expert_type == 'sleep':
            return SleepExpert(config=params)
        elif self.expert_type == 'weather':
            return WeatherExpert(config=params)
        elif self.expert_type == 'stress_diet':
            return StressDietExpert(config=params)


class GatingOptimizationProblem:
    """
    PyGMO-compatible problem class for gating network hyperparameter optimization.
    """
    
    def __init__(self, experts, train_data, val_data, seed=None):
        """
        Initialize the Gating Optimization Problem.
        
        Args:
            experts (list): List of pre-trained expert models
            train_data (tuple): Training data for all modalities (X_train_list, y_train)
            val_data (tuple): Validation data for all modalities (X_val_list, y_val)
            seed (int): Random seed for reproducibility
        """
        self.experts = experts
        self.train_data = train_data
        self.val_data = val_data
        self.seed = seed
        self.num_experts = len(experts)
        
        # Set up hyperparameter search space
        self.param_bounds = self._get_gating_bounds()
        self.integer_dims = 2  # gate_hidden_size and gate_top_k are integers
    
    def _get_gating_bounds(self):
        """Get hyperparameter bounds for Gating Network."""
        # Define bounds for each hyperparameter as two lists: [lower_bounds], [upper_bounds]
        # Ensure integer bounds are integers
        lb = [32, 1, 0.001]  # First 2 are integers
        ub = [256, 3, 0.1]
        return (lb, ub)
    
    def get_bounds(self):
        """Return the bounds of the problem."""
        return self.param_bounds
    
    def get_nobj(self):
        """Return the number of objectives."""
        return 1  # Single objective optimization
    
    def get_nix(self):
        """Return the number of integer dimensions."""
        return self.integer_dims
    
    def get_nec(self):
        """Return the number of equality constraints."""
        return 0
    
    def get_nic(self):
        """Return the number of inequality constraints."""
        return 0
    
    def fitness(self, x):
        """
        Fitness function for PyGMO optimization.
        
        Args:
            x (list): List of hyperparameter values
            
        Returns:
            float: Negative validation accuracy (to be minimized)
        """
        # Convert continuous parameters to discrete where needed
        params = self._process_params(x)
        
        # Create gating network with the given hyperparameters
        gating_network = GatingNetwork(
            num_experts=self.num_experts,
            config={
                'hidden_size': params['gate_hidden_size'],
                'top_k': params['gate_top_k'],
                'dropout_rate': 0.2  # Fixed dropout rate
            }
        )
        
        # Create fusion mechanism
        fusion_mechanism = FusionMechanism(top_k=params['gate_top_k'])
        
        # Create MoE model
        moe_model = MigraineMoEModel(
            experts=self.experts,
            gating_network=gating_network,
            fusion_mechanism=fusion_mechanism,
            load_balance_coef=params['load_balance_coef']
        )
        
        # Compile the model
        moe_model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss=losses.BinaryCrossentropy(),
            metrics=[metrics.AUC(), metrics.Recall(), metrics.Precision()]
        )
        
        # Train the model
        X_train_list, y_train = self.train_data
        X_val_list, y_val = self.val_data
        
        # Use early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=5,
            restore_best_weights=True,
            mode='max'
        )
        
        # Train with a small number of epochs for optimization speed
        history = moe_model.fit(
            X_train_list, y_train,
            epochs=10,
            batch_size=32,
            validation_data=(X_val_list, y_val),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Get the best validation AUC
        best_val_auc = max(history.history['val_auc'])
        
        # Return negative AUC (since PyGMO minimizes)
        return [-best_val_auc]
    
    def _process_params(self, x):
        """
        Process hyperparameters from continuous to discrete values where needed.
        
        Args:
            x (list): List of continuous hyperparameter values
            
        Returns:
            dict: Dictionary of processed hyperparameters
        """
        return {
            'gate_hidden_size': int(x[0]),
            'gate_top_k': int(x[1]),
            'load_balance_coef': x[2]
        }


class EndToEndOptimizationProblem:
    """
    PyGMO-compatible problem class for end-to-end MoE optimization.
    """
    
    def __init__(self, expert_configs, gating_config, train_data, val_data, seed=None):
        """
        Initialize the End-to-End Optimization Problem.
        
        Args:
            expert_configs (list): List of expert configurations
            gating_config (dict): Gating network configuration
            train_data (tuple): Training data for all modalities (X_train_list, y_train)
            val_data (tuple): Validation data for all modalities (X_val_list, y_val)
            seed (int): Random seed for reproducibility
        """
        self.expert_configs = expert_configs
        self.gating_config = gating_config
        self.train_data = train_data
        self.val_data = val_data
        self.seed = seed
        
        # Set up hyperparameter search space
        self.param_bounds = self._get_e2e_bounds()
        self.integer_dims = 1  # batch_size is integer
    
    def _get_e2e_bounds(self):
        """Get hyperparameter bounds for End-to-End optimization."""
        # Define bounds for each hyperparameter as two lists: [lower_bounds], [upper_bounds]
        # Ensure integer bounds are integers
        lb = [0.0001, 16, 0.0]  # Second parameter is integer
        ub = [0.01, 128, 0.5]
        return (lb, ub)
    
    def get_bounds(self):
        """Return the bounds of the problem."""
        return self.param_bounds
    
    def get_nobj(self):
        """Return the number of objectives."""
        return 2  # Multi-objective optimization (AUC and latency)
    
    def get_nix(self):
        """Return the number of integer dimensions."""
        return self.integer_dims
    
    def get_nec(self):
        """Return the number of equality constraints."""
        return 0
    
    def get_nic(self):
        """Return the number of inequality constraints."""
        return 0
    
    def fitness(self, x):
        """
        Fitness function for PyGMO optimization.
        
        Args:
            x (list): List of hyperparameter values
            
        Returns:
            list: Multi-objective fitness values (negative metrics to be minimized)
        """
        # Convert continuous parameters to discrete where needed
        params = self._process_params(x)
        
        # Create experts with optimized configurations
        sleep_expert = SleepExpert(config=self.expert_configs[0])
        weather_expert = WeatherExpert(config=self.expert_configs[1])
        stress_diet_expert = StressDietExpert(config=self.expert_configs[2])
        
        experts = [sleep_expert, weather_expert, stress_diet_expert]
        
        # Create gating network with optimized configuration
        gating_network = GatingNetwork(
            num_experts=len(experts),
            config={
                'hidden_size': self.gating_config['gate_hidden_size'],
                'top_k': self.gating_config['gate_top_k'],
                'dropout_rate': 0.2  # Fixed dropout rate
            }
        )
        
        # Create fusion mechanism
        fusion_mechanism = FusionMechanism(top_k=self.gating_config['gate_top_k'])
        
        # Create MoE model with L2 regularization
        moe_model = MigraineMoEModel(
            experts=experts,
            gating_network=gating_network,
            fusion_mechanism=fusion_mechanism,
            load_balance_coef=self.gating_config['load_balance_coef'],
            l2_regularization=params['l2_regularization']
        )
        
        # Compile the model with the optimized learning rate
        moe_model.compile(
            optimizer=optimizers.Adam(learning_rate=params['learning_rate']),
            loss=losses.BinaryCrossentropy(),
            metrics=[metrics.AUC(), metrics.Recall(), metrics.Precision()]
        )
        
        # Train the model
        X_train_list, y_train = self.train_data
        X_val_list, y_val = self.val_data
        
        # Use early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=5,
            restore_best_weights=True,
            mode='max'
        )
        
        # Train with a small number of epochs for optimization speed
        start_time = time.time()
        history = moe_model.fit(
            X_train_list, y_train,
            epochs=10,
            batch_size=params['batch_size'],
            validation_data=(X_val_list, y_val),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Get the best validation AUC
        best_val_auc = max(history.history['val_auc'])
        
        # Measure inference latency
        inference_start = time.time()
        _ = moe_model.predict(X_val_list[:10])  # Predict on a small batch
        inference_time = (time.time() - inference_start) * 100  # Scale for better optimization
        
        # Return negative AUC and latency (since PyGMO minimizes)
        return [-best_val_auc, inference_time]
    
    def _process_params(self, x):
        """
        Process hyperparameters from continuous to discrete values where needed.
        
        Args:
            x (list): List of continuous hyperparameter values
            
        Returns:
            dict: Dictionary of processed hyperparameters
        """
        return {
            'learning_rate': x[0],
            'batch_size': int(x[1]),
            'l2_regularization': x[2]
        }


class ExpertHyperparamOptimization:
    """
    Phase 1: Expert Hyperparameter Optimization using PyGMO.
    
    Optimizes each expert's architecture independently using Differential Evolution
    or CMA-ES algorithms.
    
    Attributes:
        expert_type (str): Type of expert to optimize ('sleep', 'weather', or 'stress_diet')
        train_data (tuple): Training data for the expert
        val_data (tuple): Validation data for the expert
        seed (int): Random seed for reproducibility
    """
    
    def __init__(self, expert_type, train_data, val_data, seed=None):
        """
        Initialize the Expert Hyperparameter Optimization.
        
        Args:
            expert_type (str): Type of expert to optimize ('sleep', 'weather', or 'stress_diet')
            train_data (tuple): Training data for the expert (X_train, y_train)
            val_data (tuple): Validation data for the expert (X_val, y_val)
            seed (int): Random seed for reproducibility
        """
        self.expert_type = expert_type
        self.train_data = train_data
        self.val_data = val_data
        self.seed = seed
    
    def optimize(self, pop_size=20, generations=10, algorithm='de'):
        """
        Run the optimization process using PyGMO.
        
        Args:
            pop_size (int): Population size for the evolutionary algorithm
            generations (int): Number of generations to evolve
            algorithm (str): Algorithm to use ('de' for Differential Evolution or 'cmaes' for CMA-ES)
            
        Returns:
            tuple: (best_hyperparameters, best_fitness)
        """
        # Create PyGMO problem using the custom problem class
        prob = ExpertOptimizationProblem(
            expert_type=self.expert_type,
            train_data=self.train_data,
            val_data=self.val_data,
            seed=self.seed
        )
        
        # Create population
        pop = pg.population(prob, size=pop_size, seed=self.seed)
        
        # Select algorithm
        if algorithm == 'de':
            algo = pg.de(gen=generations, seed=self.seed)
        elif algorithm == 'cmaes':
            algo = pg.cmaes(gen=generations, seed=self.seed)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Evolve the population
        pop = algo.evolve(pop)
        
        # Get the best solution
        best_idx = pop.best_idx()
        best_params = pop.get_x()[best_idx]
        best_fitness = pop.get_f()[best_idx]
        
        # Process the best parameters
        best_hyperparameters = prob._process_params(best_params)
        
        return best_hyperparameters, best_fitness[0]


class GatingHyperparamOptimization:
    """
    Phase 2: Gating Hyperparameter Optimization using PyGMO.
    
    Optimizes gating network with fixed experts using Particle Swarm Optimization
    or Ant Colony Optimization algorithms.
    
    Attributes:
        experts (list): List of pre-trained expert models
        train_data (tuple): Training data for all modalities
        val_data (tuple): Validation data for all modalities
        seed (int): Random seed for reproducibility
    """
    
    def __init__(self, experts, train_data, val_data, seed=None):
        """
        Initialize the Gating Hyperparameter Optimization.
        
        Args:
            experts (list): List of pre-trained expert models
            train_data (tuple): Training data for all modalities (X_train_list, y_train)
            val_data (tuple): Validation data for all modalities (X_val_list, y_val)
            seed (int): Random seed for reproducibility
        """
        self.experts = experts
        self.train_data = train_data
        self.val_data = val_data
        self.seed = seed
    
    def optimize(self, pop_size=20, generations=10, algorithm='pso'):
        """
        Run the optimization process using PyGMO.
        
        Args:
            pop_size (int): Population size for the evolutionary algorithm
            generations (int): Number of generations to evolve
            algorithm (str): Algorithm to use ('pso' for Particle Swarm Optimization or 'gaco' for Ant Colony Optimization)
            
        Returns:
            tuple: (best_hyperparameters, best_fitness)
        """
        # Create PyGMO problem using the custom problem class
        prob = GatingOptimizationProblem(
            experts=self.experts,
            train_data=self.train_data,
            val_data=self.val_data,
            seed=self.seed
        )
        
        # Create population
        pop = pg.population(prob, size=pop_size, seed=self.seed)
        
        # Select algorithm
        if algorithm == 'pso':
            algo = pg.pso(gen=generations, seed=self.seed)
        elif algorithm == 'gaco':
            algo = pg.gaco(gen=generations, seed=self.seed)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Evolve the population
        pop = algo.evolve(pop)
        
        # Get the best solution
        best_idx = pop.best_idx()
        best_params = pop.get_x()[best_idx]
        best_fitness = pop.get_f()[best_idx]
        
        # Process the best parameters
        best_hyperparameters = prob._process_params(best_params)
        
        return best_hyperparameters, best_fitness[0]


class EndToEndMoEOptimization:
    """
    Phase 3: End-to-End MoE Optimization using PyGMO.
    
    Fine-tunes the entire model with the best configurations from previous phases.
    
    Attributes:
        expert_configs (list): List of expert configurations
        gating_config (dict): Gating network configuration
        train_data (tuple): Training data for all modalities
        val_data (tuple): Validation data for all modalities
        seed (int): Random seed for reproducibility
    """
    
    def __init__(self, expert_configs, gating_config, train_data, val_data, seed=None):
        """
        Initialize the End-to-End MoE Optimization.
        
        Args:
            expert_configs (list): List of expert configurations
            gating_config (dict): Gating network configuration
            train_data (tuple): Training data for all modalities (X_train_list, y_train)
            val_data (tuple): Validation data for all modalities (X_val_list, y_val)
            seed (int): Random seed for reproducibility
        """
        self.expert_configs = expert_configs
        self.gating_config = gating_config
        self.train_data = train_data
        self.val_data = val_data
        self.seed = seed
    
    def optimize(self, pop_size=20, generations=10, algorithm='nsga2'):
        """
        Run the optimization process using PyGMO.
        
        Args:
            pop_size (int): Population size for the evolutionary algorithm
            generations (int): Number of generations to evolve
            algorithm (str): Algorithm to use ('nsga2' for NSGA-II or 'moead' for MOEA/D)
            
        Returns:
            tuple: (best_hyperparameters, best_fitness)
        """
        # Create PyGMO problem using the custom problem class
        prob = EndToEndOptimizationProblem(
            expert_configs=self.expert_configs,
            gating_config=self.gating_config,
            train_data=self.train_data,
            val_data=self.val_data,
            seed=self.seed
        )
        
        # Create population
        pop = pg.population(prob, size=pop_size, seed=self.seed)
        
        # Select algorithm
        if algorithm == 'nsga2':
            algo = pg.nsga2(gen=generations, seed=self.seed)
        elif algorithm == 'moead':
            algo = pg.moead(gen=generations, seed=self.seed)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Evolve the population
        pop = algo.evolve(pop)
        
        # For multi-objective optimization, select the solution with the best AUC
        # (first objective) as the primary metric
        best_auc = float('inf')
        best_idx = 0
        
        for i, f in enumerate(pop.get_f()):
            if f[0] < best_auc:  # Remember we're minimizing negative AUC
                best_auc = f[0]
                best_idx = i
        
        best_params = pop.get_x()[best_idx]
        best_fitness = pop.get_f()[best_idx]
        
        # Process the best parameters
        best_hyperparameters = prob._process_params(best_params)
        
        return best_hyperparameters, best_fitness


# MigraineMoEModel class definition (needed for the optimization)
class MigraineMoEModel(Model):
    """
    Mixture of Experts model for migraine prediction.
    
    This model combines multiple expert models using a gating network.
    
    Attributes:
        experts (list): List of expert models
        gating_network (GatingNetwork): Gating network for expert selection
        fusion_mechanism (FusionMechanism): Mechanism to fuse expert outputs
        load_balance_coef (float): Coefficient for load balancing loss
        l2_regularization (float): L2 regularization coefficient
    """
    
    def __init__(self, experts, gating_network, fusion_mechanism, 
                 load_balance_coef=0.01, l2_regularization=0.0):
        """
        Initialize the MigraineMoEModel.
        
        Args:
            experts (list): List of expert models
            gating_network (GatingNetwork): Gating network for expert selection
            fusion_mechanism (FusionMechanism): Mechanism to fuse expert outputs
            load_balance_coef (float): Coefficient for load balancing loss
            l2_regularization (float): L2 regularization coefficient
        """
        super(MigraineMoEModel, self).__init__()
        self.experts = experts
        self.gating_network = gating_network
        self.fusion_mechanism = fusion_mechanism
        self.load_balance_coef = load_balance_coef
        self.l2_regularization = l2_regularization
        
        # Final classification layer
        self.classifier = layers.Dense(
            1, activation='sigmoid',
            kernel_regularizer=tf.keras.regularizers.l2(l2_regularization)
        )
    
    def call(self, inputs, training=None):
        """
        Forward pass of the MoE model.
        
        Args:
            inputs (list): List of input tensors for each expert
            training (bool): Whether the model is in training mode
            
        Returns:
            tensor: Model output
        """
        # Get expert outputs
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_output = expert(inputs[i], training=training)
            expert_outputs.append(expert_output)
        
        # Stack expert outputs
        stacked_expert_outputs = tf.stack(expert_outputs, axis=1)
        
        # Get gating weights
        gating_weights = self.gating_network(expert_outputs, training=training)
        
        # Fuse expert outputs using gating weights
        fused_output = self.fusion_mechanism([stacked_expert_outputs, gating_weights])
        
        # Final classification
        output = self.classifier(fused_output)
        
        # Add load balancing loss during training
        if training:
            # Calculate load balancing loss
            # This encourages all experts to be used equally
            importance = tf.reduce_mean(gating_weights, axis=0)
            load_balancing_loss = tf.reduce_sum(importance * tf.math.log(importance + 1e-10))
            
            # Add to model losses
            self.add_loss(self.load_balance_coef * load_balancing_loss)
        
        return output
