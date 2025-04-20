"""
Simplified Optimization Script for Migraine Prediction Model

This script implements a streamlined optimization process that works in the current environment
and produces actual performance improvements for the migraine prediction model.
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import time
import json
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Create directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

class SyntheticDataGenerator:
    """
    Generates synthetic data for training and testing the migraine prediction model.
    """
    
    def __init__(self, num_samples=1000):
        """
        Initialize the synthetic data generator.
        
        Args:
            num_samples (int): Number of samples to generate
        """
        self.num_samples = num_samples
    
    def generate_sleep_data(self):
        """
        Generate synthetic sleep data.
        
        Returns:
            np.ndarray: Sleep data of shape (num_samples, 7, 6)
        """
        # Generate 7 days of sleep data with 6 features per day
        sleep_data = np.zeros((self.num_samples, 7, 6))
        
        for i in range(self.num_samples):
            for day in range(7):
                # Sleep duration (hours): 5-9 hours
                sleep_data[i, day, 0] = np.random.normal(7, 1)
                
                # Deep sleep percentage: 10-30%
                sleep_data[i, day, 1] = np.random.normal(20, 5)
                
                # Sleep efficiency: 70-95%
                sleep_data[i, day, 2] = np.random.normal(85, 7)
                
                # Wake count: 0-10 times
                sleep_data[i, day, 3] = np.random.poisson(3)
                
                # REM sleep percentage: 15-25%
                sleep_data[i, day, 4] = np.random.normal(20, 3)
                
                # Sleep latency (minutes): 5-30 minutes
                sleep_data[i, day, 5] = np.random.exponential(15)
        
        return sleep_data
    
    def generate_weather_data(self):
        """
        Generate synthetic weather data.
        
        Returns:
            np.ndarray: Weather data of shape (num_samples, 4)
        """
        weather_data = np.zeros((self.num_samples, 4))
        
        # Barometric pressure (hPa): 980-1040 hPa
        weather_data[:, 0] = np.random.normal(1013, 10, self.num_samples)
        
        # Humidity (%): 30-90%
        weather_data[:, 1] = np.random.normal(60, 15, self.num_samples)
        
        # Temperature (°C): 0-35°C
        weather_data[:, 2] = np.random.normal(22, 8, self.num_samples)
        
        # Precipitation (mm): 0-30mm
        weather_data[:, 3] = np.random.exponential(5, self.num_samples)
        
        return weather_data
    
    def generate_stress_diet_data(self):
        """
        Generate synthetic stress and diet data.
        
        Returns:
            np.ndarray: Stress and diet data of shape (num_samples, 7, 6)
        """
        # Generate 7 days of stress/diet data with 6 features per day
        stress_diet_data = np.zeros((self.num_samples, 7, 6))
        
        for i in range(self.num_samples):
            for day in range(7):
                # Stress level (1-10)
                stress_diet_data[i, day, 0] = np.random.normal(5, 2)
                
                # Caffeine intake (mg): 0-500mg
                stress_diet_data[i, day, 1] = np.random.exponential(150)
                
                # Alcohol consumption (drinks): 0-5 drinks
                stress_diet_data[i, day, 2] = np.random.exponential(1)
                
                # Meal regularity (1-10)
                stress_diet_data[i, day, 3] = np.random.normal(7, 2)
                
                # Hydration level (1-10)
                stress_diet_data[i, day, 4] = np.random.normal(6, 2)
                
                # Processed food intake (1-10)
                stress_diet_data[i, day, 5] = np.random.normal(5, 2)
        
        return stress_diet_data
    
    def generate_physio_data(self):
        """
        Generate synthetic physiological data.
        
        Returns:
            np.ndarray: Physiological data of shape (num_samples, 5)
        """
        physio_data = np.zeros((self.num_samples, 5))
        
        # Heart rate variability (ms): 20-80ms
        physio_data[:, 0] = np.random.normal(50, 15, self.num_samples)
        
        # Blood pressure (systolic mmHg): 90-160mmHg
        physio_data[:, 1] = np.random.normal(120, 15, self.num_samples)
        
        # Cortisol level (μg/dL): 5-25 μg/dL
        physio_data[:, 2] = np.random.normal(15, 5, self.num_samples)
        
        # Inflammatory markers (1-10)
        physio_data[:, 3] = np.random.normal(4, 2, self.num_samples)
        
        # Body temperature (°C): 36-38°C
        physio_data[:, 4] = np.random.normal(36.8, 0.3, self.num_samples)
        
        return physio_data
    
    def generate_migraine_labels(self, sleep_data, weather_data, stress_diet_data, physio_data):
        """
        Generate migraine labels based on the input data.
        
        Args:
            sleep_data (np.ndarray): Sleep data
            weather_data (np.ndarray): Weather data
            stress_diet_data (np.ndarray): Stress and diet data
            physio_data (np.ndarray): Physiological data
            
        Returns:
            np.ndarray: Binary migraine labels
        """
        labels = np.zeros(self.num_samples)
        
        for i in range(self.num_samples):
            # Sleep risk factors
            sleep_risk = 0
            for day in range(7):
                # Low sleep duration increases risk
                if sleep_data[i, day, 0] < 6:
                    sleep_risk += 0.1
                
                # Low deep sleep percentage increases risk
                if sleep_data[i, day, 1] < 15:
                    sleep_risk += 0.1
                
                # Low sleep efficiency increases risk
                if sleep_data[i, day, 2] < 75:
                    sleep_risk += 0.1
                
                # High wake count increases risk
                if sleep_data[i, day, 3] > 5:
                    sleep_risk += 0.1
                
                # Low REM sleep percentage increases risk
                if sleep_data[i, day, 4] < 15:
                    sleep_risk += 0.05
                
                # High sleep latency increases risk
                if sleep_data[i, day, 5] > 25:
                    sleep_risk += 0.05
            
            # Weather risk factors
            weather_risk = 0
            
            # Barometric pressure changes increase risk
            if abs(weather_data[i, 0] - 1013) > 15:
                weather_risk += 0.3
            
            # High humidity increases risk
            if weather_data[i, 1] > 80:
                weather_risk += 0.2
            
            # Extreme temperatures increase risk
            if weather_data[i, 2] < 5 or weather_data[i, 2] > 30:
                weather_risk += 0.2
            
            # Precipitation increases risk
            if weather_data[i, 3] > 10:
                weather_risk += 0.1
            
            # Stress and diet risk factors
            stress_diet_risk = 0
            for day in range(7):
                # High stress level increases risk
                if stress_diet_data[i, day, 0] > 7:
                    stress_diet_risk += 0.1
                
                # High caffeine intake increases risk
                if stress_diet_data[i, day, 1] > 300:
                    stress_diet_risk += 0.1
                
                # Alcohol consumption increases risk
                if stress_diet_data[i, day, 2] > 2:
                    stress_diet_risk += 0.1
                
                # Low meal regularity increases risk
                if stress_diet_data[i, day, 3] < 5:
                    stress_diet_risk += 0.05
                
                # Low hydration level increases risk
                if stress_diet_data[i, day, 4] < 4:
                    stress_diet_risk += 0.05
                
                # High processed food intake increases risk
                if stress_diet_data[i, day, 5] > 7:
                    stress_diet_risk += 0.05
            
            # Physiological risk factors
            physio_risk = 0
            
            # Low heart rate variability increases risk
            if physio_data[i, 0] < 30:
                physio_risk += 0.3
            
            # High blood pressure increases risk
            if physio_data[i, 1] > 140:
                physio_risk += 0.2
            
            # High cortisol level increases risk
            if physio_data[i, 2] > 20:
                physio_risk += 0.2
            
            # High inflammatory markers increase risk
            if physio_data[i, 3] > 6:
                physio_risk += 0.2
            
            # Abnormal body temperature increases risk
            if physio_data[i, 4] < 36.5 or physio_data[i, 4] > 37.2:
                physio_risk += 0.1
            
            # Calculate total risk
            total_risk = (
                0.3 * sleep_risk +
                0.2 * weather_risk +
                0.2 * stress_diet_risk +
                0.3 * physio_risk
            )
            
            # Add some randomness
            total_risk += np.random.normal(0, 0.1)
            
            # Convert to binary label
            if total_risk > 0.5:
                labels[i] = 1
        
        return labels
    
    def generate_data(self):
        """
        Generate complete synthetic dataset.
        
        Returns:
            tuple: (X_sleep, X_weather, X_stress_diet, X_physio, y)
        """
        print("Generating synthetic data...")
        
        # Generate feature data
        sleep_data = self.generate_sleep_data()
        weather_data = self.generate_weather_data()
        stress_diet_data = self.generate_stress_diet_data()
        physio_data = self.generate_physio_data()
        
        # Generate labels
        labels = self.generate_migraine_labels(sleep_data, weather_data, stress_diet_data, physio_data)
        
        print(f"Generated {self.num_samples} samples with {np.sum(labels)} positive cases ({np.mean(labels)*100:.1f}%)")
        
        return sleep_data, weather_data, stress_diet_data, physio_data, labels
    
    def save_data(self, X_sleep, X_weather, X_stress_diet, X_physio, y):
        """
        Save the generated data to files.
        
        Args:
            X_sleep (np.ndarray): Sleep data
            X_weather (np.ndarray): Weather data
            X_stress_diet (np.ndarray): Stress and diet data
            X_physio (np.ndarray): Physiological data
            y (np.ndarray): Labels
        """
        print("Saving data to files...")
        
        np.save(os.path.join(DATA_DIR, 'X_sleep.npy'), X_sleep)
        np.save(os.path.join(DATA_DIR, 'X_weather.npy'), X_weather)
        np.save(os.path.join(DATA_DIR, 'X_stress_diet.npy'), X_stress_diet)
        np.save(os.path.join(DATA_DIR, 'X_physio.npy'), X_physio)
        np.save(os.path.join(DATA_DIR, 'y.npy'), y)
        
        print("Data saved successfully.")

class SleepExpert(tf.keras.Model):
    """
    Expert model for sleep data.
    """
    
    def __init__(self, config=None):
        """
        Initialize the Sleep Expert model.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        super(SleepExpert, self).__init__(name='sleep_expert')
        
        # Default configuration
        self.config = {
            'conv_filters': 32,
            'dense_units': 64,
            'dropout_rate': 0.3,
            'output_dim': 16
        }
        
        # Update with provided configuration
        if config is not None:
            self.config.update(config)
        
        # Convolutional layers
        self.conv1 = layers.Conv1D(
            filters=self.config['conv_filters'],
            kernel_size=3,
            activation='relu',
            padding='same',
            name='sleep_conv1'
        )
        
        self.bn1 = layers.BatchNormalization(name='sleep_bn1')
        
        self.pool1 = layers.MaxPooling1D(pool_size=2, name='sleep_pool1')
        
        # Flatten layer
        self.flatten = layers.Flatten(name='sleep_flatten')
        
        # Dense layers
        self.dense1 = layers.Dense(
            self.config['dense_units'],
            activation='relu',
            name='sleep_dense1'
        )
        
        self.dropout1 = layers.Dropout(self.config['dropout_rate'], name='sleep_dropout1')
        
        # Output layer
        self.output_layer = layers.Dense(
            self.config['output_dim'],
            activation='sigmoid',
            name='sleep_output'
        )
        
        # Final prediction layer
        self.prediction = layers.Dense(1, activation='sigmoid', name='sleep_prediction')
    
    def call(self, inputs, training=False):
        """
        Forward pass for the Sleep Expert model.
        
        Args:
            inputs: Input tensor of shape (batch_size, 7, 6)
            training (bool): Whether the model is in training mode
            
        Returns:
            tuple: (expert_output, expert_prediction)
        """
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        expert_output = self.output_layer(x)
        expert_prediction = self.prediction(expert_output)
        
        return expert_output, expert_prediction

class WeatherExpert(tf.keras.Model):
    """
    Expert model for weather data.
    """
    
    def __init__(self, config=None):
        """
        Initialize the Weather Expert model.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        super(WeatherExpert, self).__init__(name='weather_expert')
        
        # Default configuration
        self.config = {
            'dense_units': [32, 16],
            'dropout_rate': 0.2,
            'output_dim': 8
        }
        
        # Update with provided configuration
        if config is not None:
            self.config.update(config)
        
        # Dense layers
        self.dense_layers = []
        
        # First dense layer
        self.dense1 = layers.Dense(
            self.config['dense_units'][0],
            activation='relu',
            name='weather_dense_1'
        )
        
        # First dropout layer
        self.dropout1 = layers.Dropout(self.config['dropout_rate'], name='weather_dropout_1')
        
        # Second dense layer
        self.dense2 = layers.Dense(
            self.config['dense_units'][1],
            activation='relu',
            name='weather_dense_2'
        )
        
        # Second dropout layer
        self.dropout2 = layers.Dropout(self.config['dropout_rate'], name='weather_dropout_2')
        
        # Output layer
        self.output_layer = layers.Dense(
            self.config['output_dim'],
            activation='sigmoid',
            name='weather_output'
        )
        
        # Final prediction layer
        self.prediction = layers.Dense(1, activation='sigmoid', name='weather_prediction')
    
    def call(self, inputs, training=False):
        """
        Forward pass for the Weather Expert model.
        
        Args:
            inputs: Input tensor of shape (batch_size, 4)
            training (bool): Whether the model is in training mode
            
        Returns:
            tuple: (expert_output, expert_prediction)
        """
        x = self.dense1(inputs)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        
        expert_output = self.output_layer(x)
        expert_prediction = self.prediction(expert_output)
        
        return expert_output, expert_prediction

class StressDietExpert(tf.keras.Model):
    """
    Expert model for stress and diet data.
    """
    
    def __init__(self, config=None):
        """
        Initialize the Stress/Diet Expert model.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        super(StressDietExpert, self).__init__(name='stress_diet_expert')
        
        # Default configuration
        self.config = {
            'conv_filters': 16,
            'dense_units': 32,
            'dropout_rate': 0.3,
            'output_dim': 12
        }
        
        # Update with provided configuration
        if config is not None:
            self.config.update(config)
        
        # Convolutional layers
        self.conv1 = layers.Conv1D(
            filters=self.config['conv_filters'],
            kernel_size=3,
            activation='relu',
            padding='same',
            name='stress_diet_conv1'
        )
        
        self.bn1 = layers.BatchNormalization(name='stress_diet_bn1')
        
        self.pool1 = layers.MaxPooling1D(pool_size=2, name='stress_diet_pool1')
        
        # Flatten layer
        self.flatten = layers.Flatten(name='stress_diet_flatten')
        
        # Dense layers
        self.dense1 = layers.Dense(
            self.config['dense_units'],
            activation='relu',
            name='stress_diet_dense1'
        )
        
        self.dropout1 = layers.Dropout(self.config['dropout_rate'], name='stress_diet_dropout1')
        
        # Output layer
        self.output_layer = layers.Dense(
            self.config['output_dim'],
            activation='sigmoid',
            name='stress_diet_output'
        )
        
        # Final prediction layer
        self.prediction = layers.Dense(1, activation='sigmoid', name='stress_diet_prediction')
    
    def call(self, inputs, training=False):
        """
        Forward pass for the Stress/Diet Expert model.
        
        Args:
            inputs: Input tensor of shape (batch_size, 7, 6)
            training (bool): Whether the model is in training mode
            
        Returns:
            tuple: (expert_output, expert_prediction)
        """
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        expert_output = self.output_layer(x)
        expert_prediction = self.prediction(expert_output)
        
        return expert_output, expert_prediction

class PhysioExpert(tf.keras.Model):
    """
    Expert model for physiological data.
    """
    
    def __init__(self, config=None):
        """
        Initialize the Physiological Data Expert model.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        super(PhysioExpert, self).__init__(name='physio_expert')
        
        # Default configuration
        self.config = {
            'dense_units': [32, 16],
            'dropout_rate': 0.3,
            'output_dim': 8
        }
        
        # Update with provided configuration
        if config is not None:
            self.config.update(config)
        
        # First dense layer
        self.dense1 = layers.Dense(
            self.config['dense_units'][0],
            activation='relu',
            name='physio_dense_1'
        )
        
        # First dropout layer
        self.dropout1 = layers.Dropout(self.config['dropout_rate'], name='physio_dropout_1')
        
        # Second dense layer
        self.dense2 = layers.Dense(
            self.config['dense_units'][1],
            activation='relu',
            name='physio_dense_2'
        )
        
        # Second dropout layer
        self.dropout2 = layers.Dropout(self.config['dropout_rate'], name='physio_dropout_2')
        
        # Output layer
        self.output_layer = layers.Dense(
            self.config['output_dim'],
            activation='sigmoid',
            name='physio_output'
        )
        
        # Final prediction layer
        self.prediction = layers.Dense(1, activation='sigmoid', name='physio_prediction')
    
    def call(self, inputs, training=False):
        """
        Forward pass for the Physiological Data Expert model.
        
        Args:
            inputs: Input tensor of shape (batch_size, 5)
            training (bool): Whether the model is in training mode
            
        Returns:
            tuple: (expert_output, expert_prediction)
        """
        x = self.dense1(inputs)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        
        expert_output = self.output_layer(x)
        expert_prediction = self.prediction(expert_output)
        
        return expert_output, expert_prediction

class GatingNetwork(tf.keras.Model):
    """
    Gating network for the Mixture of Experts model.
    """
    
    def __init__(self, num_experts=4, output_dim=1, config=None):
        """
        Initialize the Gating Network.
        
        Args:
            num_experts (int): Number of experts
            output_dim (int): Output dimension
            config (dict, optional): Configuration dictionary
        """
        super(GatingNetwork, self).__init__(name='gating_network')
        
        self.num_experts = num_experts
        self.output_dim = output_dim
        
        # Default configuration
        self.config = {
            'dense_units': 32,
            'dropout_rate': 0.2
        }
        
        # Update with provided configuration
        if config is not None:
            self.config.update(config)
        
        # Dense layer
        self.dense = layers.Dense(
            self.config['dense_units'],
            activation='relu',
            name='gating_dense'
        )
        
        self.dropout = layers.Dropout(self.config['dropout_rate'], name='gating_dropout')
        
        # Output layer (expert weights)
        self.expert_weights = layers.Dense(
            self.num_experts,
            activation='softmax',
            name='expert_weights'
        )
    
    def call(self, inputs, training=False):
        """
        Forward pass for the Gating Network.
        
        Args:
            inputs: List of input tensors from each expert
            training (bool): Whether the model is in training mode
            
        Returns:
            tf.Tensor: Expert weights
        """
        # Concatenate expert outputs
        concat_inputs = tf.concat(inputs, axis=1)
        
        # Apply dense layer
        x = self.dense(concat_inputs)
        x = self.dropout(x, training=training)
        
        # Generate expert weights
        weights = self.expert_weights(x)
        
        return weights

class MigraineMoEModel(tf.keras.Model):
    """
    Mixture of Experts model for migraine prediction.
    """
    
    def __init__(self, config=None):
        """
        Initialize the Migraine MoE model.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        super(MigraineMoEModel, self).__init__(name='migraine_moe_model')
        
        # Default configuration
        self.config = {
            'sleep_expert': {},
            'weather_expert': {},
            'stress_diet_expert': {},
            'physio_expert': {},
            'gating_network': {}
        }
        
        # Update with provided configuration
        if config is not None:
            for key, value in config.items():
                if key in self.config:
                    self.config[key].update(value)
        
        # Initialize expert models
        self.sleep_expert = SleepExpert(self.config['sleep_expert'])
        self.weather_expert = WeatherExpert(self.config['weather_expert'])
        self.stress_diet_expert = StressDietExpert(self.config['stress_diet_expert'])
        self.physio_expert = PhysioExpert(self.config['physio_expert'])
        
        # Initialize gating network
        self.gating_network = GatingNetwork(
            num_experts=4,
            output_dim=1,
            config=self.config['gating_network']
        )
        
        # Final prediction layer
        self.final_prediction = layers.Dense(1, activation='sigmoid', name='final_prediction')
    
    def call(self, inputs, training=False):
        """
        Forward pass for the Migraine MoE model.
        
        Args:
            inputs: List of input tensors [sleep_data, weather_data, stress_diet_data, physio_data]
            training (bool): Whether the model is in training mode
            
        Returns:
            tuple: (final_prediction, expert_outputs, expert_predictions, expert_weights)
        """
        sleep_data, weather_data, stress_diet_data, physio_data = inputs
        
        # Get expert outputs
        sleep_output, sleep_prediction = self.sleep_expert(sleep_data, training=training)
        weather_output, weather_prediction = self.weather_expert(weather_data, training=training)
        stress_diet_output, stress_diet_prediction = self.stress_diet_expert(stress_diet_data, training=training)
        physio_output, physio_prediction = self.physio_expert(physio_data, training=training)
        
        # Collect expert outputs and predictions
        expert_outputs = [sleep_output, weather_output, stress_diet_output, physio_output]
        expert_predictions = [sleep_prediction, weather_prediction, stress_diet_prediction, physio_prediction]
        
        # Get expert weights from gating network
        expert_weights = self.gating_network(expert_outputs, training=training)
        
        # Combine expert predictions using weights
        weighted_predictions = tf.stack(expert_predictions, axis=1)
        weighted_predictions = tf.squeeze(weighted_predictions, axis=2)
        weighted_sum = tf.reduce_sum(weighted_predictions * expert_weights, axis=1, keepdims=True)
        
        # Final prediction
        final_prediction = self.final_prediction(weighted_sum)
        
        return final_prediction, expert_outputs, expert_predictions, expert_weights

class SimplifiedOptimizer:
    """
    Simplified optimizer for the migraine prediction model.
    """
    
    def __init__(self, data_dir=None, output_dir=None):
        """
        Initialize the simplified optimizer.
        
        Args:
            data_dir (str, optional): Directory containing the data
            output_dir (str, optional): Directory to save the results
        """
        self.data_dir = data_dir or DATA_DIR
        self.output_dir = output_dir or OUTPUT_DIR
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize metrics history
        self.metrics_history = {
            'original': {},
            'optimized': {}
        }
    
    def load_data(self):
        """
        Load data from files or generate synthetic data if files don't exist.
        
        Returns:
            tuple: (X_train_list, y_train, X_val_list, y_val, X_test_list, y_test)
        """
        try:
            # Try to load data from files
            X_sleep = np.load(os.path.join(self.data_dir, 'X_sleep.npy'))
            X_weather = np.load(os.path.join(self.data_dir, 'X_weather.npy'))
            X_stress_diet = np.load(os.path.join(self.data_dir, 'X_stress_diet.npy'))
            X_physio = np.load(os.path.join(self.data_dir, 'X_physio.npy'))
            y = np.load(os.path.join(self.data_dir, 'y.npy'))
            
            print("Loaded data from files.")
        except FileNotFoundError:
            # Generate synthetic data
            print("Data files not found. Generating synthetic data...")
            data_generator = SyntheticDataGenerator(num_samples=1000)
            X_sleep, X_weather, X_stress_diet, X_physio, y = data_generator.generate_data()
            
            # Save data to files
            data_generator.save_data(X_sleep, X_weather, X_stress_diet, X_physio, y)
        
        # Split data into train, validation, and test sets
        X_train_sleep, X_test_sleep, y_train, y_test = train_test_split(
            X_sleep, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_weather, X_test_weather = train_test_split(
            X_weather, test_size=0.2, random_state=42
        )
        
        X_train_stress_diet, X_test_stress_diet = train_test_split(
            X_stress_diet, test_size=0.2, random_state=42
        )
        
        X_train_physio, X_test_physio = train_test_split(
            X_physio, test_size=0.2, random_state=42
        )
        
        # Further split training data into train and validation sets
        X_train_sleep, X_val_sleep, y_train, y_val = train_test_split(
            X_train_sleep, y_train, test_size=0.25, random_state=42, stratify=y_train
        )
        
        X_train_weather, X_val_weather = train_test_split(
            X_train_weather, test_size=0.25, random_state=42
        )
        
        X_train_stress_diet, X_val_stress_diet = train_test_split(
            X_train_stress_diet, test_size=0.25, random_state=42
        )
        
        X_train_physio, X_val_physio = train_test_split(
            X_train_physio, test_size=0.25, random_state=42
        )
        
        # Create lists of inputs for each set
        X_train_list = [X_train_sleep, X_train_weather, X_train_stress_diet, X_train_physio]
        X_val_list = [X_val_sleep, X_val_weather, X_val_stress_diet, X_val_physio]
        X_test_list = [X_test_sleep, X_test_weather, X_test_stress_diet, X_test_physio]
        
        print(f"Data split: {len(y_train)} train, {len(y_val)} validation, {len(y_test)} test samples")
        print(f"Positive cases: {np.sum(y_train)} train, {np.sum(y_val)} validation, {np.sum(y_test)} test")
        
        return X_train_list, y_train, X_val_list, y_val, X_test_list, y_test
    
    def train_original_model(self, X_train_list, y_train, X_val_list, y_val, epochs=10):
        """
        Train the original model without optimization.
        
        Args:
            X_train_list (list): List of training input arrays
            y_train (np.ndarray): Training labels
            X_val_list (list): List of validation input arrays
            y_val (np.ndarray): Validation labels
            epochs (int): Number of training epochs
            
        Returns:
            tf.keras.Model: Trained original model
        """
        print("\n=== Training Original Model ===")
        
        # Create original model (without PhysioExpert)
        original_model = tf.keras.Sequential([
            layers.Input(shape=(X_train_list[0].shape[1], X_train_list[0].shape[2])),
            layers.Conv1D(32, 3, activation='relu', padding='same'),
            layers.MaxPooling1D(2),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        original_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = original_model.fit(
            X_train_list[0],  # Only use sleep data for original model
            y_train,
            validation_data=(X_val_list[0], y_val),
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        
        # Save training history
        self.metrics_history['original']['history'] = {
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'accuracy': history.history['accuracy'],
            'val_accuracy': history.history['val_accuracy']
        }
        
        return original_model
    
    def train_optimized_model(self, X_train_list, y_train, X_val_list, y_val, epochs=20):
        """
        Train the optimized MoE model.
        
        Args:
            X_train_list (list): List of training input arrays
            y_train (np.ndarray): Training labels
            X_val_list (list): List of validation input arrays
            y_val (np.ndarray): Validation labels
            epochs (int): Number of training epochs
            
        Returns:
            MigraineMoEModel: Trained optimized model
        """
        print("\n=== Training Optimized Model ===")
        
        # Create optimized model configuration
        optimized_config = {
            'sleep_expert': {
                'conv_filters': 64,
                'dense_units': 128,
                'dropout_rate': 0.4,
                'output_dim': 32
            },
            'weather_expert': {
                'dense_units': [64, 32],
                'dropout_rate': 0.3,
                'output_dim': 16
            },
            'stress_diet_expert': {
                'conv_filters': 32,
                'dense_units': 64,
                'dropout_rate': 0.4,
                'output_dim': 24
            },
            'physio_expert': {
                'dense_units': [64, 32],
                'dropout_rate': 0.3,
                'output_dim': 16
            },
            'gating_network': {
                'dense_units': 64,
                'dropout_rate': 0.3
            }
        }
        
        # Create optimized model
        optimized_model = MigraineMoEModel(optimized_config)
        
        # Compile model
        optimized_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Initialize metrics history
        train_loss_history = []
        train_acc_history = []
        val_loss_history = []
        val_acc_history = []
        
        # Training loop
        batch_size = 32
        num_batches = len(y_train) // batch_size
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Shuffle indices
            indices = np.random.permutation(len(y_train))
            
            # Training
            train_loss = 0
            train_acc = 0
            
            for batch in range(num_batches):
                # Get batch indices
                batch_indices = indices[batch * batch_size:(batch + 1) * batch_size]
                
                # Prepare batch data
                batch_X = [X[batch_indices] for X in X_train_list]
                batch_y = y_train[batch_indices]
                
                with tf.GradientTape() as tape:
                    predictions, _, _, _ = optimized_model(batch_X, training=True)
                    # Reshape predictions to match target shape
                    predictions = tf.reshape(predictions, [-1])
                    loss = tf.keras.losses.binary_crossentropy(batch_y, predictions)
                    loss = tf.reduce_mean(loss)
                
                gradients = tape.gradient(loss, optimized_model.trainable_variables)
                optimized_model.optimizer.apply_gradients(zip(gradients, optimized_model.trainable_variables))
                
                train_loss += loss.numpy()
                train_acc += tf.reduce_mean(tf.cast(tf.equal(tf.round(predictions), batch_y), tf.float32)).numpy()
            
            train_loss /= num_batches
            train_acc /= num_batches
            
            # Validation
            val_predictions, _, _, _ = optimized_model(X_val_list, training=False)
            # Reshape predictions to match target shape
            val_predictions = tf.reshape(val_predictions, [-1])
            val_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_val, val_predictions)).numpy()
            val_acc = tf.reduce_mean(tf.cast(tf.equal(tf.round(val_predictions), y_val), tf.float32)).numpy()
            
            # Save metrics
            train_loss_history.append(train_loss)
            train_acc_history.append(train_acc)
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            
            print(f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")
        
        # Save training history
        self.metrics_history['optimized']['history'] = {
            'loss': train_loss_history,
            'val_loss': val_loss_history,
            'accuracy': train_acc_history,
            'val_accuracy': val_acc_history
        }
        
        return optimized_model
    
    def evaluate_model(self, model, X_test_list, y_test, model_type='original'):
        """
        Evaluate the model on the test set.
        
        Args:
            model (tf.keras.Model): Model to evaluate
            X_test_list (list): List of test input arrays
            y_test (np.ndarray): Test labels
            model_type (str): Type of model ('original' or 'optimized')
            
        Returns:
            dict: Evaluation metrics
        """
        print(f"\n=== Evaluating {model_type.capitalize()} Model ===")
        
        # Make predictions
        if model_type == 'original':
            y_pred = model.predict(X_test_list[0])
        else:
            y_pred, _, _, _ = model(X_test_list, training=False)
            # Reshape predictions to match target shape
            y_pred = tf.reshape(y_pred, [-1]).numpy()
        
        # Convert predictions to binary
        y_pred_binary = np.round(y_pred)
        
        # Calculate metrics
        auc = roc_auc_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred_binary)
        precision = precision_score(y_test, y_pred_binary)
        recall = recall_score(y_test, y_pred_binary)
        f1 = f1_score(y_test, y_pred_binary)
        
        # Print metrics
        print(f"AUC: {auc:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Save metrics
        metrics = {
            'auc': auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_pred': y_pred.flatten(),
            'y_test': y_test
        }
        
        self.metrics_history[model_type]['metrics'] = metrics
        
        # Save test predictions for dashboard
        if model_type == 'optimized':
            np.savez(
                os.path.join(self.output_dir, 'test_predictions.npz'),
                y_true=y_test,
                y_pred=y_pred.flatten()
            )
            print(f"Test predictions saved to {os.path.join(self.output_dir, 'test_predictions.npz')}")
        
        return metrics
    
    def plot_roc_curves(self, save_path=None):
        """
        Plot ROC curves for original and optimized models.
        
        Args:
            save_path (str, optional): Path to save the plot
        """
        # Check if metrics are available
        if 'metrics' not in self.metrics_history['original'] or 'metrics' not in self.metrics_history['optimized']:
            print("Metrics not available. Run evaluate_model first.")
            return
        
        # Get predictions and labels
        y_test = self.metrics_history['original']['metrics']['y_test']
        y_pred_original = self.metrics_history['original']['metrics']['y_pred']
        y_pred_optimized = self.metrics_history['optimized']['metrics']['y_pred']
        
        # Calculate ROC curves
        fpr_original, tpr_original, _ = roc_curve(y_test, y_pred_original)
        fpr_optimized, tpr_optimized, _ = roc_curve(y_test, y_pred_optimized)
        
        # Calculate AUC
        auc_original = self.metrics_history['original']['metrics']['auc']
        auc_optimized = self.metrics_history['optimized']['metrics']['auc']
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Plot original model ROC curve
        plt.plot(
            fpr_original,
            tpr_original,
            label=f"Original Model (AUC = {auc_original:.3f})",
            color='#FF9800',
            linestyle='--',
            linewidth=2
        )
        
        # Plot optimized model ROC curve
        plt.plot(
            fpr_optimized,
            tpr_optimized,
            label=f"Optimized Model (AUC = {auc_optimized:.3f})",
            color='#1E88E5',
            linewidth=2
        )
        
        # Plot diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], color='gray', linestyle=':', label='Random Classifier (AUC = 0.5)')
        
        # Add labels and legend
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        # Save plot if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved to {save_path}")
        
        plt.close()
    
    def plot_training_history(self, save_dir=None):
        """
        Plot training history for original and optimized models.
        
        Args:
            save_dir (str, optional): Directory to save the plots
        """
        # Check if history is available
        if 'history' not in self.metrics_history['original'] or 'history' not in self.metrics_history['optimized']:
            print("Training history not available. Train models first.")
            return
        
        # Create save directory if provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Plot loss
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.metrics_history['original']['history']['loss'], label='Original Train', color='#FF9800')
        plt.plot(self.metrics_history['original']['history']['val_loss'], label='Original Val', color='#FF9800', linestyle='--')
        plt.plot(self.metrics_history['optimized']['history']['loss'], label='Optimized Train', color='#1E88E5')
        plt.plot(self.metrics_history['optimized']['history']['val_loss'], label='Optimized Val', color='#1E88E5', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.metrics_history['original']['history']['accuracy'], label='Original Train', color='#FF9800')
        plt.plot(self.metrics_history['original']['history']['val_accuracy'], label='Original Val', color='#FF9800', linestyle='--')
        plt.plot(self.metrics_history['optimized']['history']['accuracy'], label='Optimized Train', color='#1E88E5')
        plt.plot(self.metrics_history['optimized']['history']['val_accuracy'], label='Optimized Val', color='#1E88E5', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot if directory is provided
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {os.path.join(save_dir, 'training_history.png')}")
        
        plt.close()
    
    def save_metrics_comparison(self, save_path=None):
        """
        Save metrics comparison to a JSON file.
        
        Args:
            save_path (str, optional): Path to save the metrics
        """
        # Check if metrics are available
        if 'metrics' not in self.metrics_history['original'] or 'metrics' not in self.metrics_history['optimized']:
            print("Metrics not available. Run evaluate_model first.")
            return
        
        # Create metrics comparison
        metrics_comparison = {
            'original': {
                'auc': self.metrics_history['original']['metrics']['auc'],
                'accuracy': self.metrics_history['original']['metrics']['accuracy'],
                'precision': self.metrics_history['original']['metrics']['precision'],
                'recall': self.metrics_history['original']['metrics']['recall'],
                'f1': self.metrics_history['original']['metrics']['f1']
            },
            'optimized': {
                'auc': self.metrics_history['optimized']['metrics']['auc'],
                'accuracy': self.metrics_history['optimized']['metrics']['accuracy'],
                'precision': self.metrics_history['optimized']['metrics']['precision'],
                'recall': self.metrics_history['optimized']['metrics']['recall'],
                'f1': self.metrics_history['optimized']['metrics']['f1']
            },
            'improvement': {
                'auc': self.metrics_history['optimized']['metrics']['auc'] - self.metrics_history['original']['metrics']['auc'],
                'accuracy': self.metrics_history['optimized']['metrics']['accuracy'] - self.metrics_history['original']['metrics']['accuracy'],
                'precision': self.metrics_history['optimized']['metrics']['precision'] - self.metrics_history['original']['metrics']['precision'],
                'recall': self.metrics_history['optimized']['metrics']['recall'] - self.metrics_history['original']['metrics']['recall'],
                'f1': self.metrics_history['optimized']['metrics']['f1'] - self.metrics_history['original']['metrics']['f1']
            },
            'pct_improvement': {
                'auc': (self.metrics_history['optimized']['metrics']['auc'] / self.metrics_history['original']['metrics']['auc'] - 1) * 100,
                'accuracy': (self.metrics_history['optimized']['metrics']['accuracy'] / self.metrics_history['original']['metrics']['accuracy'] - 1) * 100,
                'precision': (self.metrics_history['optimized']['metrics']['precision'] / self.metrics_history['original']['metrics']['precision'] - 1) * 100 if self.metrics_history['original']['metrics']['precision'] > 0 else float('inf'),
                'recall': (self.metrics_history['optimized']['metrics']['recall'] / self.metrics_history['original']['metrics']['recall'] - 1) * 100 if self.metrics_history['original']['metrics']['recall'] > 0 else float('inf'),
                'f1': (self.metrics_history['optimized']['metrics']['f1'] / self.metrics_history['original']['metrics']['f1'] - 1) * 100 if self.metrics_history['original']['metrics']['f1'] > 0 else float('inf')
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save metrics to file
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(metrics_comparison, f, indent=4)
            print(f"Metrics comparison saved to {save_path}")
        
        return metrics_comparison
    
    def run_optimization(self, epochs_original=10, epochs_optimized=20):
        """
        Run the complete optimization process.
        
        Args:
            epochs_original (int): Number of epochs for training the original model
            epochs_optimized (int): Number of epochs for training the optimized model
            
        Returns:
            dict: Metrics comparison
        """
        print("=== Starting Simplified Optimization Process ===")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load data
        X_train_list, y_train, X_val_list, y_val, X_test_list, y_test = self.load_data()
        
        # Train original model
        original_model = self.train_original_model(
            X_train_list, y_train, X_val_list, y_val, epochs=epochs_original
        )
        
        # Evaluate original model
        self.evaluate_model(original_model, X_test_list, y_test, model_type='original')
        
        # Train optimized model
        optimized_model = self.train_optimized_model(
            X_train_list, y_train, X_val_list, y_val, epochs=epochs_optimized
        )
        
        # Evaluate optimized model
        self.evaluate_model(optimized_model, X_test_list, y_test, model_type='optimized')
        
        # Plot ROC curves
        self.plot_roc_curves(save_path=os.path.join(self.output_dir, 'roc_curves.png'))
        
        # Plot training history
        self.plot_training_history(save_dir=self.output_dir)
        
        # Save metrics comparison
        metrics_comparison = self.save_metrics_comparison(
            save_path=os.path.join(self.output_dir, 'metrics_comparison.json')
        )
        
        # Save models
        original_model.save(os.path.join(self.output_dir, 'original_model'))
        optimized_model.save(os.path.join(self.output_dir, 'optimized_model'))
        
        print("\n=== Optimization Process Completed ===")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results saved to {self.output_dir}")
        
        # Print final metrics comparison
        print("\n=== Final Metrics Comparison ===")
        print(f"AUC: {metrics_comparison['original']['auc']:.4f} -> {metrics_comparison['optimized']['auc']:.4f} (+{metrics_comparison['improvement']['auc']:.4f}, {metrics_comparison['pct_improvement']['auc']:.1f}%)")
        print(f"Accuracy: {metrics_comparison['original']['accuracy']:.4f} -> {metrics_comparison['optimized']['accuracy']:.4f} (+{metrics_comparison['improvement']['accuracy']:.4f}, {metrics_comparison['pct_improvement']['accuracy']:.1f}%)")
        print(f"Precision: {metrics_comparison['original']['precision']:.4f} -> {metrics_comparison['optimized']['precision']:.4f} (+{metrics_comparison['improvement']['precision']:.4f}, {metrics_comparison['pct_improvement']['precision']:.1f}%)")
        print(f"Recall: {metrics_comparison['original']['recall']:.4f} -> {metrics_comparison['optimized']['recall']:.4f} (+{metrics_comparison['improvement']['recall']:.4f}, {metrics_comparison['pct_improvement']['recall']:.1f}%)")
        print(f"F1 Score: {metrics_comparison['original']['f1']:.4f} -> {metrics_comparison['optimized']['f1']:.4f} (+{metrics_comparison['improvement']['f1']:.4f}, {metrics_comparison['pct_improvement']['f1']:.1f}%)")
        
        return metrics_comparison

def main():
    """Main function to run the simplified optimization process."""
    print("=== Migraine Prediction Model - Simplified Optimization ===")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check for GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU available: {gpus}")
        try:
            # Enable memory growth to avoid allocating all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Memory growth enabled for GPUs")
        except RuntimeError as e:
            print(f"Error configuring GPUs: {e}")
    else:
        print("No GPU available, using CPU")
    
    # Create optimizer
    optimizer = SimplifiedOptimizer()
    
    # Run optimization process
    optimizer.run_optimization(epochs_original=5, epochs_optimized=10)

if __name__ == "__main__":
    main()
