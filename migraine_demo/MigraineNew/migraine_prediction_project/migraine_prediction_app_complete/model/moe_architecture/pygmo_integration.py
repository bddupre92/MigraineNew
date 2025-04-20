"""
PyGMO Integration for Migraine Prediction App

This module integrates PyGMO for hyperparameter optimization of the MoE architecture.
It implements the three optimization phases: Expert Hyperparameters, Gating Hyperparameters,
and End-to-End MoE optimization.
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
        
        # Set up hyperparameter search space based on expert type
        if expert_type == 'sleep':
            self.param_bounds = self._get_sleep_expert_bounds()
        elif expert_type == 'weather':
            self.param_bounds = self._get_weather_expert_bounds()
        elif expert_type == 'stress_diet':
            self.param_bounds = self._get_stress_diet_expert_bounds()
        else:
            raise ValueError(f"Unknown expert type: {expert_type}")
    
    def _get_sleep_expert_bounds(self):
        """Get hyperparameter bounds for Sleep Expert."""
        # Define bounds for each hyperparameter
        # [lower_bound, upper_bound]
        return [
            (16, 128),    # conv_filters
            (3, 7),       # kernel_size
            (32, 256),    # lstm_units
            (0.1, 0.5)    # dropout_rate
        ]
    
    def _get_weather_expert_bounds(self):
        """Get hyperparameter bounds for Weather Expert."""
        # Define bounds for each hyperparameter
        # [lower_bound, upper_bound]
        return [
            (32, 256),    # hidden_units
            (0, 1),       # activation (0: relu, 1: elu)
            (0.1, 0.5)    # dropout_rate
        ]
    
    def _get_stress_diet_expert_bounds(self):
        """Get hyperparameter bounds for Stress/Diet Expert."""
        # Define bounds for each hyperparameter
        # [lower_bound, upper_bound]
        return [
            (32, 128),    # embedding_dim
            (2, 8),       # num_heads
            (32, 128),    # transformer_dim
            (0.1, 0.4)    # dropout_rate
        ]
    
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
        # Create PyGMO problem
        prob = pg.problem(pg.fitness_wrapper(self.fitness, self.param_bounds))
        
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
        best_hyperparameters = self._process_params(best_params)
        
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
        self.num_experts = len(experts)
        
        # Set up hyperparameter search space
        self.param_bounds = self._get_gating_bounds()
    
    def _get_gating_bounds(self):
        """Get hyperparameter bounds for Gating Network."""
        # Define bounds for each hyperparameter
        # [lower_bound, upper_bound]
        return [
            (32, 256),     # gate_hidden_size
            (1, 3),        # gate_top_k
            (0.001, 0.1)   # load_balance_coef
        ]
    
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
        # Create PyGMO problem
        prob = pg.problem(pg.fitness_wrapper(self.fitness, self.param_bounds))
        
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
        best_hyperparameters = self._process_params(best_params)
        
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
        
        # Set up hyperparameter search space
        self.param_bounds = self._get_e2e_bounds()
    
    def _get_e2e_bounds(self):
        """Get hyperparameter bounds for End-to-End optimization."""
        # Define bounds for each hyperparameter
        # [lower_bound, upper_bound]
        return [
            (0.0001, 0.01),  # learning_rate
            (16, 128),       # batch_size
            (0.0, 0.5)       # l2_regularization
        ]
    
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
        
        # Create expert models
        experts = [
            SleepExpert(config=self.expert_configs[0]),
            WeatherExpert(config=self.expert_configs[1]),
            StressDietExpert(config=self.expert_configs[2])
        ]
        
        # Create gating network
        gating_network = GatingNetwork(
            num_experts=len(experts),
            config=self.gating_config
        )
        
        # Create fusion mechanism
        fusion_mechanism = FusionMechanism(top_k=self.gating_config['top_k'])
        
        # Create MoE model
        moe_model = MigraineMoEModel(
            experts=experts,
            gating_network=gating_network,
            fusion_mechanism=fusion_mechanism,
            load_balance_coef=self.gating_config['load_balance_coef']
        )
        
        # Add L2 regularization
        for layer in moe_model.layers:
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer = tf.keras.regularizers.l2(params['l2_regularization'])
        
        # Compile the model
        moe_model.compile(
            optimizer=optimizers.Adam(learning_rate=params['learning_rate']),
            loss=losses.BinaryCrossentropy(),
            metrics=[
                metrics.AUC(name='auc'),
                metrics.Recall(name='recall'),
                metrics.Precision(name='precision'),
                metrics.F1Score(name='f1_score')
            ]
        )
        
        # Train the model
        X_train_list, y_train = self.train_data
        X_val_list, y_val = self.val_data
        
        # Use early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=10,
            restore_best_weights=True,
            mode='max'
        )
        
        # Measure inference time
        start_time = time.time()
        _ = moe_model.predict(X_val_list[:10])  # Warm-up
        start_time = time.time()
        _ = moe_model.predict(X_val_list[:100])
        inference_time = (time.time() - start_time) / 100  # Average time per sample
        
        # Train the model
        history = moe_model.fit(
            X_train_list, y_train,
            epochs=30,
            batch_size=int(params['batch_size']),
            validation_data=(X_val_list, y_val),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Get the best validation metrics
        best_epoch = np.argmax(history.history['val_auc'])
        best_val_auc = history.history['val_auc'][best_epoch]
        best_val_recall = history.history['val_recall'][best_epoch]
        best_val_f1 = history.history['val_f1_score'][best_epoch]
        
        # Return multi-objective fitness (negative metrics to be minimized)
        return [
            -best_val_auc,
            -best_val_recall,
            -best_val_f1,
            inference_time / 0.2  # Normalize to target of 200ms
        ]
    
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
            'batch_size': 2 ** int(np.log2(x[1])),  # Power of 2
            'l2_regularization': x[2]
        }
    
    def optimize(self, pop_size=20, generations=10):
        """
        Run the multi-objective optimization process using PyGMO.
        
        Args:
            pop_size (int): Population size for the evolutionary algorithm
            generations (int): Number of generations to evolve
            
        Returns:
            tuple: (best_hyperparameters, best_fitness)
        """
        # Create PyGMO problem (multi-objective)
        prob = pg.problem(pg.fitness_wrapper(self.fitness, self.param_bounds, [4]))
        
        # Create population
        pop = pg.population(prob, size=pop_size, seed=self.seed)
        
        # Use NSGA-II for multi-objective optimization
        algo = pg.nsga2(gen=generations, seed=self.seed)
        
        # Evolve the population
        pop = algo.evolve(pop)
        
        # Get the Pareto front
        pareto_front = pg.non_dominated_front_2d(pop.get_f())
        
        # Select the solution with the best AUC
        best_auc_idx = np.argmin(pop.get_f()[:, 0])
        best_params = pop.get_x()[best_auc_idx]
        best_fitness = pop.get_f()[best_auc_idx]
        
        # Process the best parameters
        best_hyperparameters = self._process_params(best_params)
        
        return best_hyperparameters, best_fitness


class MigraineMoEModel(Model):
    """
    Migraine Prediction MoE Model.
    
    Combines expert networks using a gating network for migraine prediction.
    
    Attributes:
        experts (list): List of expert models
        gating_network (Model): Gating network model
        fusion_mechanism (Layer): Fusion mechanism layer
        load_balance_coef (float): Coefficient for load balancing loss
    """
    
    def __init__(self, experts, gating_network, fusion_mechanism, load_balance_coef=0.01):
        """
        Initialize the Migraine MoE Model.
        
        Args:
            experts (list): List of expert models
            gating_network (Model): Gating network model
            fusion_mechanism (Layer): Fusion mechanism layer
            load_balance_coef (float): Coefficient for load balancing loss
        """
        super(MigraineMoEModel, self).__init__()
        self.experts = experts
        self.gating_network = gating_network
        self.fusion_mechanism = fusion_mechanism
        self.load_balance_coef = load_balance_coef
        
        # --- Add back the final prediction layer ---
        self.prediction_layer = layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs, training=False):
        """
        Forward pass through the MoE model.
        
        Args:
            inputs (list): List of input tensors for each modality
            training (bool): Whether the model is in training mode
            
        Returns:
            tuple: (prediction, gate_weights, load_balance_loss)
        """
        # --- Debug Prints (Call Start) ---
        tf.print("\n--- MoE Call Debug --- ")
        # --- End Debug Prints ---
        
        # Process inputs through each expert
        expert_outputs = [
            tf.cast(self.experts[i](inputs[i], training=training), tf.float32) # Ensure float32
            for i in range(len(self.experts))
        ]
        # --- Debug Prints (Expert Outputs) ---
        for i, eo in enumerate(expert_outputs):
            tf.print(f"expert_outputs[{i}]:", tf.reduce_min(eo), tf.reduce_max(eo), tf.reduce_mean(eo), "NaNs:", tf.reduce_sum(tf.cast(tf.math.is_nan(eo), tf.int32)))
        # --- End Debug Prints ---

        # Prepare input for gating network (consistent with train_step)
        flat_expert_outputs = [layers.Flatten()(eo) for eo in expert_outputs]
        gating_input = layers.Concatenate()(flat_expert_outputs)
        # --- Debug Prints (Gating Input) ---
        tf.print("gating_input:", tf.reduce_min(gating_input), tf.reduce_max(gating_input), tf.reduce_mean(gating_input), "NaNs:", tf.reduce_sum(tf.cast(tf.math.is_nan(gating_input), tf.int32)))
        # --- End Debug Prints ---

        # Get gating weights and load balancing loss
        gate_weights, load_balance_loss = self.gating_network(gating_input, training=training)
        # --- Debug Prints (Gating Output) ---
        tf.print("gate_weights (call):", tf.reduce_min(gate_weights), tf.reduce_max(gate_weights), tf.reduce_mean(gate_weights), "NaNs:", tf.reduce_sum(tf.cast(tf.math.is_nan(gate_weights), tf.int32)))
        tf.print("load_balance_loss (call):", load_balance_loss, "NaNs:", tf.cast(tf.math.is_nan(load_balance_loss), tf.int32))
        # --- End Debug Prints ---

        # Fuse expert outputs
        fused_output = self.fusion_mechanism(expert_outputs, gate_weights)
        # --- Debug Prints (Fused Output) ---
        tf.print("fused_output:", tf.reduce_min(fused_output), tf.reduce_max(fused_output), tf.reduce_mean(fused_output), "NaNs:", tf.reduce_sum(tf.cast(tf.math.is_nan(fused_output), tf.int32)))
        # --- End Debug Prints ---

        # --- Apply the final prediction layer ---
        prediction = self.prediction_layer(fused_output)
        # --- Debug Prints (Final Prediction) ---
        tf.print("prediction (final):", tf.reduce_min(prediction), tf.reduce_max(prediction), tf.reduce_mean(prediction), "NaNs:", tf.reduce_sum(tf.cast(tf.math.is_nan(prediction), tf.int32)))
        tf.print("--- End MoE Call Debug ---")
        # --- End Debug Prints ---
        
        return prediction, gate_weights, load_balance_loss
    
    def train_step(self, data):
        """
        Custom training step with load balancing loss.
        
        Args:
            data (tuple): (inputs, targets) or (inputs, targets, sample_weights)
            
        Returns:
            dict: Dictionary of metrics
        """
        # Unpack data, handling potential sample_weight
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            # Assume data is (x, y) if not 3 elements
            x, y = data
            sample_weight = None
            # Optionally keep a warning if needed, but remove the print from the previous attempt
            # print("Warning: train_step received data tuple of length != 3. Sample weights assumed None.")

        with tf.GradientTape() as tape:
            # Forward pass - unpack all 3 values returned by call
            y_pred, gate_weights, load_balance_loss_from_call = self(x, training=True)
            
            # --- Debug Prints ---
            tf.print("\n--- train_step Debug --- ")
            tf.print("y_pred (raw):", tf.reduce_min(y_pred), tf.reduce_max(y_pred), tf.reduce_mean(y_pred), "NaNs:", tf.reduce_sum(tf.cast(tf.math.is_nan(y_pred), tf.int32)))
            tf.print("gate_weights:", tf.reduce_min(gate_weights), tf.reduce_max(gate_weights), tf.reduce_mean(gate_weights), "NaNs:", tf.reduce_sum(tf.cast(tf.math.is_nan(gate_weights), tf.int32)))
            tf.print("load_balance_loss (from call):", load_balance_loss_from_call, "NaNs:", tf.cast(tf.math.is_nan(load_balance_loss_from_call), tf.int32))
            # --- End Debug Prints ---

            # Compute main loss, passing sample_weight
            # Ensure y_pred is used, not gate_outputs
            main_loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)
            
            # --- Debug Prints (Post Main Loss) ---
            tf.print("main_loss:", main_loss, "NaNs:", tf.cast(tf.math.is_nan(main_loss), tf.int32))
            # --- End Debug Prints ---

            # Use the load_balance_loss returned from the call method
            # (Removed the separate calculation that was here before)
            load_balance_loss = load_balance_loss_from_call
            
            # Total loss
            total_loss = main_loss + self.load_balance_coef * load_balance_loss

            # --- Debug Prints (Post Total Loss) ---
            tf.print("total_loss:", total_loss, "NaNs:", tf.cast(tf.math.is_nan(total_loss), tf.int32))
            tf.print("--- End train_step Debug ---")
            # --- End Debug Prints ---

        # Compute gradients and update weights
        gradients = tape.gradient(total_loss, self.trainable_variables)

        # --- Debug Gradients for WeatherExpert Projection ---
        weather_expert_projection_kernel_name = "weather_expert/dense_3/kernel:0" # Adjust name if necessary based on model structure
        weather_expert_projection_bias_name = "weather_expert/dense_3/bias:0"   # Adjust name if necessary
        for grad, var in zip(gradients, self.trainable_variables):
            if var.name == weather_expert_projection_kernel_name:
                tf.print("  Gradient Check - WeatherExpert Projection Kernel NaNs:", tf.reduce_sum(tf.cast(tf.math.is_nan(grad), tf.int32)), " Inf:", tf.reduce_sum(tf.cast(tf.math.is_inf(grad), tf.int32)))
            elif var.name == weather_expert_projection_bias_name:
                 tf.print("  Gradient Check - WeatherExpert Projection Bias NaNs:", tf.reduce_sum(tf.cast(tf.math.is_nan(grad), tf.int32)), " Inf:", tf.reduce_sum(tf.cast(tf.math.is_inf(grad), tf.int32)))
        # --- End Debug Gradients ---

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Reshape y to match y_pred's shape for metrics
        y_reshaped = tf.reshape(y, tf.shape(y_pred))

        # Update metrics, passing sample_weight
        self.compiled_metrics.update_state(y_reshaped, y_pred, sample_weight=sample_weight)
        
        # Return metrics
        results = {m.name: m.result() for m in self.metrics}
        results.update({
            'loss': total_loss,
            'main_loss': main_loss,
            'load_balance_loss': load_balance_loss # Use the calculated loss
        })
        
        return results
    
    def test_step(self, data):
        """
        Custom test step.
        
        Args:
            data (tuple): (inputs, targets) or (inputs, targets, sample_weights)
            
        Returns:
            dict: Dictionary of metrics
        """
        # Unpack data
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        # Forward pass - unpack all 3 values
        y_pred, _, _ = self(x, training=False) # Ignore gate_weights and load_balance_loss

        # Compute loss, passing sample_weight
        # Use the actual compiled loss attribute
        self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)

        # Reshape y to match y_pred's shape for metrics
        y_reshaped = tf.reshape(y, tf.shape(y_pred))

        # Update metrics, passing sample_weight
        self.compiled_metrics.update_state(y_reshaped, y_pred, sample_weight=sample_weight)

        # Return metrics
        return {m.name: m.result() for m in self.metrics}
    
    def predict(self, x):
        """
        Predict method that returns only the prediction.
        
        Args:
            x (list): List of input tensors for each modality
            
        Returns:
            tensor: Prediction tensor
        """
        # Unpack all 3 values, return only the prediction
        prediction, _, _ = self(x, training=False) 
        return prediction


if __name__ == "__main__":
    # This would be used in the main training script
    # Example usage would be demonstrated there
    pass
