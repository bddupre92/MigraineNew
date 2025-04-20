"""
Stress/Diet Expert for Migraine Prediction App

This module implements the Stress/Diet Expert network for the Mixture of Experts (MoE)
architecture, processing stress and dietary data to predict migraine risk.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import math
import pandas as pd

class StressDietExpert(Model):
    """
    Stress/Diet Expert network using a small Transformer encoder.
    
    Processes stress and dietary features to extract patterns relevant for migraine prediction.
    
    Attributes:
        config (dict): Configuration parameters for the expert
        name (str): Name of the expert
    """
    
    def __init__(self, config=None, name="stress_diet_expert"):
        """
        Initialize the Stress/Diet Expert network.
        
        Args:
            config (dict): Configuration parameters for the expert
                - embedding_dim (int): Dimension of embedding layer
                - num_heads (int): Number of attention heads in transformer
                - transformer_dim (int): Dimension of transformer layers
                - dropout_rate (float): Dropout rate for regularization
                - output_dim (int): Dimension of output feature vector
            name (str): Name of the expert
        """
        super(StressDietExpert, self).__init__(name=name)
        
        # Default configuration
        self.config = {
            'embedding_dim': 64,
            'num_heads': 4,
            'transformer_dim': 64,
            'dropout_rate': 0.2,
            'output_dim': 64
        }
        
        # Update with provided configuration
        if config:
            self.config.update(config)
        
        # Build the model
        self._build_model()
        
    def _build_model(self):
        """Build the Stress/Diet Expert model architecture."""
        # Embedding layer
        self.embedding = layers.Dense(
            units=self.config['embedding_dim'],
            activation='relu'
        )
        
        # Transformer encoder layer
        self.transformer_encoder = TransformerEncoder(
            embed_dim=self.config['embedding_dim'],
            num_heads=self.config['num_heads'],
            ff_dim=self.config['transformer_dim'],
            rate=self.config['dropout_rate']
        )
        
        # Global average pooling
        self.global_avg_pool = layers.GlobalAveragePooling1D()
        
        # Output layer
        self.output_layer = layers.Dense(
            units=self.config['output_dim'],
            activation='relu'
        )
        
    def call(self, inputs, training=False):
        """
        Forward pass through the Stress/Diet Expert network.
        
        Args:
            inputs (tensor): Input tensor of shape (batch_size, sequence_length, num_features)
            training (bool): Whether the model is in training mode
            
        Returns:
            tensor: Output tensor of shape (batch_size, output_dim)
        """
        # Embedding
        x = self.embedding(inputs)
        
        # Add positional encoding
        x = self.add_positional_encoding(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x, training=training)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        
        # Output layer
        output = self.output_layer(x)
        
        return output
    
    def add_positional_encoding(self, inputs):
        """
        Add positional encoding to the input embeddings using TensorFlow operations.

        Args:
            inputs (tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim)

        Returns:
            tensor: Tensor with positional encoding added
        """
        seq_length = tf.shape(inputs)[1]
        embed_dim = tf.shape(inputs)[2]
        # Ensure embed_dim is treated as float for division
        float_embed_dim = tf.cast(embed_dim, dtype=tf.float32)

        # Calculate positional encoding (modified from original TF tutorial)
        # Create positions tensor [seq_length, 1]
        positions = tf.range(start=0, limit=seq_length, delta=1, dtype=tf.float32)
        positions = tf.expand_dims(positions, axis=1)

        # Create angle rates tensor [1, embed_dim/2]
        angle_rads_base = tf.range(0, embed_dim, 2, dtype=tf.float32) # Shape (embed_dim/2,)
        angle_rads = 1 / tf.pow(10000.0, (angle_rads_base / float_embed_dim))
        angle_rads = tf.expand_dims(angle_rads, axis=0) # Shape (1, embed_dim/2)

        # Calculate angles [seq_length, embed_dim/2]
        angle_rads = positions * angle_rads

        # Apply sin to even indices (0, 2, ...) and cos to odd indices (1, 3, ...)
        # Calculate sines and cosines [seq_length, embed_dim/2]
        sines = tf.sin(angle_rads)
        cosines = tf.cos(angle_rads)

        # Interleave sines and cosines
        # Reshape sines and cosines to [seq_length, embed_dim/2, 1]
        sines = tf.expand_dims(sines, axis=2)
        cosines = tf.expand_dims(cosines, axis=2)
        # Concatenate along the last axis [seq_length, embed_dim/2, 2]
        pos_encoding = tf.concat([sines, cosines], axis=2)
        # Reshape to [seq_length, embed_dim]
        # Ensure the shape is compatible even if embed_dim is odd
        # This reshape flattens the last two dimensions
        pos_encoding = tf.reshape(pos_encoding, [-1, embed_dim]) # Shape [seq_length, embed_dim]
        # Ensure the shape matches the input up to embed_dim if it was odd
        pos_encoding = pos_encoding[:, :embed_dim]

        # Add batch dimension and broadcast
        pos_encoding = tf.expand_dims(pos_encoding, axis=0) # Shape [1, seq_length, embed_dim]

        # Ensure pos_encoding has the same dtype as inputs
        pos_encoding = tf.cast(pos_encoding, dtype=inputs.dtype)

        # Add positional encoding to inputs
        return inputs + pos_encoding
    
    def preprocess(self, stress_diet_data, sequence_length=7):
        """
        Preprocess stress and dietary data for input to the Stress/Diet Expert.

        Args:
            stress_diet_data (DataFrame): DataFrame containing stress and dietary data
                Must have columns: ['stress_level', 'alcohol_consumed', 'caffeine_consumed',
                                   'chocolate_consumed', 'processed_food_consumed', 'water_consumed_liters']
            sequence_length (int): Number of days to include in each sequence

        Returns:
            numpy.ndarray: Preprocessed data of shape (num_samples, sequence_length, num_features) with float64 dtype.
        """
        # Extract relevant features
        features = [
            'stress_level', 'alcohol_consumed', 'caffeine_consumed',
            'chocolate_consumed', 'processed_food_consumed', 'water_consumed_liters'
        ]

        # Select and explicitly convert to numeric, coercing errors to NaN
        data_numeric = stress_diet_data[features].apply(pd.to_numeric, errors='coerce')

        # Fill any NaNs resulting from coercion or original data (e.g., with 0 or mean/median)
        # Using 0 for simplicity here, consider imputation if more appropriate
        data_filled = data_numeric.fillna(0)

        # Convert to numpy array with a specific numeric dtype
        data = data_filled.values.astype(np.float64)

        # Normalize numerical columns (stress_level and water_consumed_liters)
        # Indices might change based on 'features' list above
        num_cols_indices = [features.index(col) for col in ['stress_level', 'water_consumed_liters'] if col in features]

        if not num_cols_indices:
             print("Warning: Could not find numerical columns for normalization in StressDietExpert.")
        else:
            # Apply normalization only to existing numerical columns
            data_subset_to_normalize = data[:, num_cols_indices]
            data_mean = np.mean(data_subset_to_normalize, axis=0)
            data_std = np.std(data_subset_to_normalize, axis=0)
            # Avoid division by zero if std is zero
            safe_std = np.where(data_std == 0, 1e-8, data_std)
            normalized_subset = (data_subset_to_normalize - data_mean) / safe_std
            # Place normalized data back into the main array
            data[:, num_cols_indices] = normalized_subset

        # Create sequences
        sequences = []
        num_samples = len(data)
        for i in range(num_samples - sequence_length + 1):
            sequences.append(data[i:i+sequence_length, :])

        # Convert the list of sequences to a NumPy array with explicit float dtype
        # Check if sequences list is empty before converting
        if not sequences:
            # Return an empty array with the correct shape and dtype
            num_features = len(features)
            return np.empty((0, sequence_length, num_features), dtype=np.float64)

        return np.array(sequences, dtype=np.float64)
    
    def get_config(self):
        """Get the configuration of the Stress/Diet Expert."""
        config = super(StressDietExpert, self).get_config()
        config.update({"config": self.config})
        return config


class TransformerEncoder(layers.Layer):
    """
    Transformer Encoder layer.
    
    Attributes:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        ff_dim (int): Feed-forward dimension
        rate (float): Dropout rate
    """
    
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        """
        Initialize the Transformer Encoder layer.
        
        Args:
            embed_dim (int): Embedding dimension
            num_heads (int): Number of attention heads
            ff_dim (int): Feed-forward dimension
            rate (float): Dropout rate
        """
        super(TransformerEncoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        
        # Multi-head attention
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads
        )
        
        # Feed-forward network
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim)
        ])
        
        # Layer normalization
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        # Dropout
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
    
    def call(self, inputs, training=False):
        """
        Forward pass through the Transformer Encoder layer.
        
        Args:
            inputs (tensor): Input tensor
            training (bool): Whether the layer is in training mode
            
        Returns:
            tensor: Output tensor
        """
        # Multi-head attention with residual connection and layer normalization
        attn_output = self.mha(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward network with residual connection and layer normalization
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


if __name__ == "__main__":
    # Example usage
    config = {
        'embedding_dim': 32,
        'num_heads': 2,
        'transformer_dim': 64,
        'dropout_rate': 0.1,
        'output_dim': 64
    }
    
    # Create the model
    stress_diet_expert = StressDietExpert(config)
    
    # Build the model with sample input
    sample_input = tf.random.normal((32, 7, 6))  # (batch_size, sequence_length, num_features)
    output = stress_diet_expert(sample_input)
    
    print(f"Model built successfully!")
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Summary
    stress_diet_expert.summary()
