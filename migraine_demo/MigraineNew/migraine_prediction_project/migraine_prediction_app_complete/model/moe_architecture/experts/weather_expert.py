"""
Weather Expert for Migraine Prediction App

This module implements the Weather Expert network for the Mixture of Experts (MoE)
architecture, processing weather data to predict migraine risk.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class WeatherExpert(Model):
    """
    Weather Expert network using 3-layer MLP with residual connections.
    
    Processes weather data features to extract patterns relevant for migraine prediction.
    
    Attributes:
        config (dict): Configuration parameters for the expert
        name (str): Name of the expert
    """
    
    def __init__(self, config=None, name="weather_expert"):
        """
        Initialize the Weather Expert network.
        
        Args:
            config (dict): Configuration parameters for the expert
                - hidden_units (int): Number of units in hidden layers
                - activation (str): Activation function to use
                - dropout_rate (float): Dropout rate for regularization
                - output_dim (int): Dimension of output feature vector
            name (str): Name of the expert
        """
        super(WeatherExpert, self).__init__(name=name)
        
        # Default configuration
        self.config = {
            'hidden_units': 128,
            'activation': 'relu',
            'dropout_rate': 0.3,
            'output_dim': 64
        }
        
        # Update with provided configuration
        if config:
            self.config.update(config)
        
        # Build the model
        self._build_model()
        
    def _build_model(self):
        """Build the Weather Expert model architecture."""
        # First MLP block with residual connection
        self.dense1 = layers.Dense(
            units=self.config['hidden_units'],
            activation=None
        )
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.Activation(self.config['activation'])
        self.dropout1 = layers.Dropout(rate=self.config['dropout_rate'])
        
        # Second MLP block with residual connection
        self.dense2 = layers.Dense(
            units=self.config['hidden_units'],
            activation=None
        )
        self.bn2 = layers.BatchNormalization()
        self.act2 = layers.Activation(self.config['activation'])
        self.dropout2 = layers.Dropout(rate=self.config['dropout_rate'])
        
        # Third MLP block with residual connection
        self.dense3 = layers.Dense(
            units=self.config['hidden_units'],
            activation=None
        )
        self.bn3 = layers.BatchNormalization()
        self.act3 = layers.Activation(self.config['activation'])
        self.dropout3 = layers.Dropout(rate=self.config['dropout_rate'])
        
        # Projection layer for residual connections
        self.projection = layers.Dense(
            units=self.config['hidden_units'],
            activation='tanh',
            kernel_initializer='he_normal',
            bias_initializer=tf.keras.initializers.Constant(0.01)
        )
        
        # Output layer
        self.output_layer = layers.Dense(
            units=self.config['output_dim'],
            activation=self.config['activation']
        )
        
    def call(self, inputs, training=False):
        """
        Forward pass through the Weather Expert network.
        
        Args:
            inputs (tensor): Input tensor of shape (batch_size, num_features)
            training (bool): Whether the model is in training mode
            
        Returns:
            tensor: Output tensor of shape (batch_size, output_dim)
        """
        # Project input for residual connection (this builds the layer)
        residual = self.projection(inputs)

        # --- Debug Print (AFTER projection is built/called) ---
        tf.print("  WeatherExpert Call - Input Received Min/Max/Mean:", tf.reduce_min(inputs), tf.reduce_max(inputs), tf.reduce_mean(inputs), "NaNs:", tf.reduce_sum(tf.cast(tf.math.is_nan(inputs), tf.int32)))
        tf.print("  WeatherExpert Call - Projection Kernel Min/Max/Mean:", tf.reduce_min(self.projection.kernel), tf.reduce_max(self.projection.kernel), tf.reduce_mean(self.projection.kernel), "NaNs:", tf.reduce_sum(tf.cast(tf.math.is_nan(self.projection.kernel), tf.int32)))
        tf.print("  WeatherExpert Call - Projection Bias Min/Max/Mean:", tf.reduce_min(self.projection.bias), tf.reduce_max(self.projection.bias), tf.reduce_mean(self.projection.bias), "NaNs:", tf.reduce_sum(tf.cast(tf.math.is_nan(self.projection.bias), tf.int32)))
        tf.print("  WeatherExpert Call - After Projection (residual):", tf.reduce_min(residual), tf.reduce_max(residual), tf.reduce_mean(residual), "NaNs:", tf.reduce_sum(tf.cast(tf.math.is_nan(residual), tf.int32)))
        # --- End Debug Print ---
        
        # First MLP block with residual connection
        x = self.dense1(inputs) # NOTE: inputs is used here, not residual
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.dropout1(x, training=training)
        x = x + residual  # Residual connection
        # --- Debug Print ---
        tf.print("  WeatherExpert Call - After Block 1:", tf.reduce_sum(x), "NaNs:", tf.reduce_sum(tf.cast(tf.math.is_nan(x), tf.int32)))
        # --- End Debug Print ---
        
        # Second MLP block with residual connection
        residual = x
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)
        x = self.dropout2(x, training=training)
        x = x + residual  # Residual connection
        # --- Debug Print ---
        tf.print("  WeatherExpert Call - After Block 2:", tf.reduce_sum(x), "NaNs:", tf.reduce_sum(tf.cast(tf.math.is_nan(x), tf.int32)))
        # --- End Debug Print ---
        
        # Third MLP block with residual connection
        residual = x
        x = self.dense3(x)
        x = self.bn3(x, training=training)
        x = self.act3(x)
        x = self.dropout3(x, training=training)
        x = x + residual  # Residual connection
        # --- Debug Print ---
        tf.print("  WeatherExpert Call - After Block 3 (Before Output Layer):", tf.reduce_sum(x), "NaNs:", tf.reduce_sum(tf.cast(tf.math.is_nan(x), tf.int32)))
        # --- End Debug Print ---
        
        # Output layer
        output = self.output_layer(x)
        # --- Debug Print ---
        tf.print("  WeatherExpert Call - Final Output:", tf.reduce_sum(output), "NaNs:", tf.reduce_sum(tf.cast(tf.math.is_nan(output), tf.int32)))
        # --- End Debug Print ---
        
        return output
    
    def preprocess(self, weather_data):
        """
        Preprocess weather data for input to the Weather Expert.
        
        Args:
            weather_data (DataFrame): DataFrame containing weather data
                Must have columns: ['temperature', 'humidity', 'pressure', 'pressure_change_24h']
            
        Returns:
            tensor: Preprocessed data of shape (num_samples, num_features)
        """
        # Extract relevant features
        features = [
            'temperature', 'humidity', 'pressure', 'pressure_change_24h'
        ]
        
        # Convert to numpy array
        data = weather_data[features].values
        
        # Normalize data
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)

        # --- Check for zero or near-zero std dev ---
        zero_std_mask = data_std < 1e-7
        if np.any(zero_std_mask):
            print(f"Warning: WeatherExpert encountered near-zero standard deviation for features: {np.array(features)[zero_std_mask]}")
            print(f"Std Dev values: {data_std}")
            # Optional: Handle this case, e.g., by setting std dev to 1 for these features to avoid division issues
            # data_std[zero_std_mask] = 1.0 
        # --- End Check ---

        normalized_data = (data - data_mean) / (data_std + 1e-8)
        
        # --- Check for NaNs/Infs AFTER normalization ---
        if np.any(np.isnan(normalized_data)) or np.any(np.isinf(normalized_data)):
            print("Warning: NaNs or Infs detected in WeatherExpert output AFTER normalization!")
            print(f"NaN count: {np.isnan(normalized_data).sum()}, Inf count: {np.isinf(normalized_data).sum()}")
        # --- End Check ---

        return normalized_data
    
    def get_config(self):
        """Get the configuration of the Weather Expert."""
        config = super(WeatherExpert, self).get_config()
        config.update({"config": self.config})
        return config

if __name__ == "__main__":
    # Example usage
    config = {
        'hidden_units': 64,
        'activation': 'elu',
        'dropout_rate': 0.2,
        'output_dim': 64
    }
    
    # Create the model
    weather_expert = WeatherExpert(config)
    
    # Build the model with sample input
    sample_input = tf.random.normal((32, 4))  # (batch_size, num_features)
    output = weather_expert(sample_input)
    
    print(f"Model built successfully!")
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Summary
    weather_expert.summary()
