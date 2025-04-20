"""
Physiological Data Expert Model for Migraine Prediction

This module implements a specialized expert model for physiological data
that contributes to the Mixture of Experts (MoE) architecture for migraine prediction.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

class PhysioExpert(tf.keras.Model):
    """
    Expert model for physiological data features.
    
    This model processes physiological data such as heart rate, blood pressure,
    sleep quality metrics, and other biometric measurements to predict migraine risk.
    
    The architecture uses multiple dense layers with residual connections to
    effectively capture complex relationships in physiological measurements.
    """
    
    def __init__(self, config=None):
        """
        Initialize the Physiological Data Expert model.
        
        Args:
            config (dict, optional): Configuration dictionary with the following keys:
                - hidden_units (int): Number of units in hidden layers (default: 128)
                - num_layers (int): Number of dense layers (default: 3)
                - activation (str): Activation function (default: 'relu')
                - dropout_rate (float): Dropout rate (default: 0.3)
                - output_dim (int): Output dimension (default: 64)
        """
        super(PhysioExpert, self).__init__(name='physio_expert')
        
        # Set default configuration
        self.config = {
            'hidden_units': 128,
            'num_layers': 3,
            'activation': 'relu',
            'dropout_rate': 0.3,
            'output_dim': 64
        }
        
        # Update with provided configuration
        if config is not None:
            self.config.update(config)
        
        # Input layer
        self.input_layer = layers.InputLayer(input_shape=(None,), name='physio_input')
        
        # Feature normalization
        self.normalization = layers.BatchNormalization(name='physio_normalization')
        
        # Dense layers with residual connections
        self.dense_layers = []
        for i in range(self.config['num_layers']):
            self.dense_layers.append(
                layers.Dense(
                    self.config['hidden_units'],
                    activation=self.config['activation'],
                    name=f'physio_dense_{i+1}'
                )
            )
            self.dense_layers.append(
                layers.BatchNormalization(name=f'physio_bn_{i+1}')
            )
            self.dense_layers.append(
                layers.Dropout(self.config['dropout_rate'], name=f'physio_dropout_{i+1}')
            )
        
        # Residual connection
        self.residual = layers.Dense(
            self.config['hidden_units'],
            activation=None,
            name='physio_residual'
        )
        
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
            inputs: Input tensor of shape (batch_size, features)
            training (bool): Whether the model is in training mode
            
        Returns:
            tuple: (expert_output, expert_prediction)
                - expert_output: Output tensor for MoE fusion (batch_size, output_dim)
                - expert_prediction: Binary prediction tensor (batch_size, 1)
        """
        x = self.input_layer(inputs)
        x = self.normalization(x, training=training)
        
        # Save input for residual connection
        residual_input = self.residual(x)
        
        # Apply dense layers
        for layer in self.dense_layers:
            x = layer(x, training=training)
        
        # Add residual connection
        x = x + residual_input
        
        # Apply output layer
        expert_output = self.output_layer(x)
        
        # Generate prediction
        expert_prediction = self.prediction(expert_output)
        
        return expert_output, expert_prediction
    
    def get_config(self):
        """Get model configuration."""
        config = super(PhysioExpert, self).get_config()
        config.update({"config": self.config})
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create model from configuration."""
        return cls(**config)
