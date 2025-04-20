"""
Stress/Diet Expert for Migraine Prediction App

This module implements the Stress/Diet Expert network for the Mixture of Experts (MoE)
architecture, processing stress and dietary data to predict migraine risk.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import math

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
        Add positional encoding to the input embeddings.
        
        Args:
            inputs (tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim)
            
        Returns:
            tensor: Tensor with positional encoding added
        """
        seq_length = tf.shape(inputs)[1]
        embed_dim = tf.shape(inputs)[2]
        
        # Create positional encoding
        positions = tf.range(start=0, limit=seq_length, delta=1, dtype=tf.float32)
        positions = tf.expand_dims(positions, axis=1)
        
        # Calculate the angles for the positional encoding
        div_term = tf.exp(
            tf.range(0, embed_dim, 2, dtype=tf.float32) * 
            (-math.log(10000.0) / embed_dim)
        )
        
        # Apply sin to even indices and cos to odd indices
        pos_encoding = tf.zeros_like(inputs)
        
        # Create indices for even and odd positions
        even_indices = tf.range(0, embed_dim, 2, dtype=tf.int32)
        odd_indices = tf.range(1, embed_dim, 2, dtype=tf.int32)
        
        # Calculate sin and cos values
        sin_values = tf.sin(positions * div_term)
        cos_values = tf.cos(positions * div_term)
        
        # Expand dimensions for broadcasting
        sin_values = tf.expand_dims(sin_values, axis=0)
        cos_values = tf.expand_dims(cos_values, axis=0)
        
        # Create updates for even and odd indices
        updates_sin = tf.zeros_like(inputs[:, :, even_indices]) + sin_values
        updates_cos = tf.zeros_like(inputs[:, :, odd_indices]) + cos_values
        
        # Create positional encoding tensor
        pos_encoding = tf.zeros_like(inputs)
        
        # Update even and odd indices
        pos_encoding_list = []
        for i in range(embed_dim):
            if i % 2 == 0:
                pos_encoding_list.append(sin_values[:, :, i//2])
            else:
                pos_encoding_list.append(cos_values[:, :, i//2])
        
        pos_encoding = tf.stack(pos_encoding_list, axis=-1)
        
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
            tensor: Preprocessed data of shape (num_samples, sequence_length, num_features)
        """
        # Extract relevant features
        features = [
            'stress_level', 'alcohol_consumed', 'caffeine_consumed',
            'chocolate_consumed', 'processed_food_consumed', 'water_consumed_liters'
        ]
        
        # Convert to numpy array
        data = stress_diet_data[features].values
        
        # Convert boolean columns to float
        bool_cols = [1, 2, 3, 4]  # Indices of boolean columns
        for i in range(data.shape[0]):
            for j in bool_cols:
                data[i, j] = float(data[i, j])
        
        # Normalize numerical columns (stress_level and water_consumed_liters)
        num_cols = [0, 5]  # Indices of numerical columns
        for j in num_cols:
            col_mean = np.mean(data[:, j])
            col_std = np.std(data[:, j])
            data[:, j] = (data[:, j] - col_mean) / (col_std + 1e-8)
        
        # Create sequences
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i+sequence_length])
        
        return np.array(sequences)
    
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
