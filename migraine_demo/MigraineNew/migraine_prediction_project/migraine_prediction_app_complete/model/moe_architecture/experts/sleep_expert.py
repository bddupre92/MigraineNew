"""
Sleep Expert for Migraine Prediction App

This module implements the Sleep Expert network for the Mixture of Experts (MoE)
architecture, processing sleep data to predict migraine risk.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class SleepExpert(Model):
    """
    Sleep Expert network using 1D-CNN followed by Bidirectional LSTM.
    
    Processes sleep data features to extract patterns relevant for migraine prediction.
    
    Attributes:
        config (dict): Configuration parameters for the expert
        name (str): Name of the expert
    """
    
    def __init__(self, config=None, name="sleep_expert"):
        """
        Initialize the Sleep Expert network.
        
        Args:
            config (dict): Configuration parameters for the expert
                - conv_filters (int): Number of filters in convolutional layers
                - kernel_size (int): Size of convolutional kernels
                - lstm_units (int): Number of units in LSTM layers
                - dropout_rate (float): Dropout rate for regularization
                - output_dim (int): Dimension of output feature vector
            name (str): Name of the expert
        """
        super(SleepExpert, self).__init__(name=name)
        
        # Default configuration
        self.config = {
            'conv_filters': 64,
            'kernel_size': 5,
            'lstm_units': 128,
            'dropout_rate': 0.3,
            'output_dim': 64
        }
        
        # Update with provided configuration
        if config:
            self.config.update(config)
        
        # Build the model
        self._build_model()
        
    def _build_model(self):
        """Build the Sleep Expert model architecture."""
        # 1D Convolutional layers
        self.conv1 = layers.Conv1D(
            filters=self.config['conv_filters'],
            kernel_size=self.config['kernel_size'],
            padding='same',
            activation=None
        )
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.Activation('relu')
        self.pool1 = layers.MaxPooling1D(pool_size=2)
        
        self.conv2 = layers.Conv1D(
            filters=self.config['conv_filters'] * 2,
            kernel_size=self.config['kernel_size'],
            padding='same',
            activation=None
        )
        self.bn2 = layers.BatchNormalization()
        self.act2 = layers.Activation('relu')
        self.pool2 = layers.MaxPooling1D(pool_size=2)
        
        # Bidirectional LSTM layer
        self.bilstm = layers.Bidirectional(
            layers.LSTM(
                units=self.config['lstm_units'],
                return_sequences=False
            )
        )
        
        # Dropout for regularization
        self.dropout = layers.Dropout(rate=self.config['dropout_rate'])
        
        # Output layer
        self.output_layer = layers.Dense(
            units=self.config['output_dim'],
            activation='relu'
        )
        
    def call(self, inputs, training=False):
        """
        Forward pass through the Sleep Expert network.
        
        Args:
            inputs (tensor): Input tensor of shape (batch_size, sequence_length, num_features)
            training (bool): Whether the model is in training mode
            
        Returns:
            tensor: Output tensor of shape (batch_size, output_dim)
        """
        # Convolutional blocks
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)
        x = self.pool2(x)
        
        # Bidirectional LSTM
        x = self.bilstm(x)
        
        # Dropout
        x = self.dropout(x, training=training)
        
        # Output layer
        output = self.output_layer(x)
        
        return output
    
    def preprocess(self, sleep_data, sequence_length=7):
        """
        Preprocess sleep data for input to the Sleep Expert.
        
        Args:
            sleep_data (DataFrame): DataFrame containing sleep data
                Must have columns: ['total_sleep_hours', 'deep_sleep_pct', 'rem_sleep_pct', 
                                   'light_sleep_pct', 'awake_time_mins', 'sleep_quality']
            sequence_length (int): Number of days to include in each sequence
            
        Returns:
            tensor: Preprocessed data of shape (num_samples, sequence_length, num_features)
        """
        # Extract relevant features
        features = [
            'total_sleep_hours', 'deep_sleep_pct', 'rem_sleep_pct',
            'light_sleep_pct', 'awake_time_mins', 'sleep_quality'
        ]
        
        # Convert to numpy array
        data = sleep_data[features].values
        
        # Normalize data
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        normalized_data = (data - data_mean) / (data_std + 1e-8)
        
        # Create sequences
        sequences = []
        for i in range(len(normalized_data) - sequence_length + 1):
            sequences.append(normalized_data[i:i+sequence_length])
        
        return np.array(sequences)
    
    def get_config(self):
        """Get the configuration of the Sleep Expert."""
        config = super(SleepExpert, self).get_config()
        config.update({"config": self.config})
        return config

if __name__ == "__main__":
    # Example usage
    config = {
        'conv_filters': 32,
        'kernel_size': 3,
        'lstm_units': 64,
        'dropout_rate': 0.2,
        'output_dim': 64
    }
    
    # Create the model
    sleep_expert = SleepExpert(config)
    
    # Build the model with sample input
    sample_input = tf.random.normal((32, 7, 6))  # (batch_size, sequence_length, num_features)
    output = sleep_expert(sample_input)
    
    print(f"Model built successfully!")
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Summary
    sleep_expert.summary()
