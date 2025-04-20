"""
Gating Network for Migraine Prediction App

This module implements the Gating Network for the Mixture of Experts (MoE)
architecture, determining the contribution of each expert to the final prediction.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class GatingNetwork(Model):
    """
    Gating Network that determines the contribution of each expert to the final prediction.
    
    Attributes:
        config (dict): Configuration parameters for the gating network
        name (str): Name of the gating network
        num_experts (int): Number of experts in the MoE architecture
    """
    
    def __init__(self, num_experts=3, config=None, name="gating_network"):
        """
        Initialize the Gating Network.
        
        Args:
            num_experts (int): Number of experts in the MoE architecture
            config (dict): Configuration parameters for the gating network
                - hidden_size (int): Number of units in hidden layers
                - top_k (int): Number of experts to select
                - dropout_rate (float): Dropout rate for regularization
            name (str): Name of the gating network
        """
        super(GatingNetwork, self).__init__(name=name)
        
        self.num_experts = num_experts
        
        # Default configuration
        self.config = {
            'hidden_size': 128,
            'top_k': 2,
            'dropout_rate': 0.2
        }
        
        # Update with provided configuration
        if config:
            self.config.update(config)
        
        # Build the model
        self._build_model()
        
    def _build_model(self):
        """Build the Gating Network model architecture."""
        # First hidden layer
        self.dense1 = layers.Dense(
            units=self.config['hidden_size'],
            activation=None
        )
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.Activation('relu')
        self.dropout1 = layers.Dropout(rate=self.config['dropout_rate'])
        
        # Second hidden layer
        self.dense2 = layers.Dense(
            units=self.config['hidden_size'] // 2,
            activation=None
        )
        self.bn2 = layers.BatchNormalization()
        self.act2 = layers.Activation('relu')
        self.dropout2 = layers.Dropout(rate=self.config['dropout_rate'])
        
        # Output layer (gating weights)
        self.gate_output = layers.Dense(
            units=self.num_experts,
            activation='softmax'
        )
        
    def call(self, inputs, training=False):
        """
        Forward pass through the Gating Network.
        
        Args:
            inputs (tensor): Input tensor for the gating network. 
                             (Note: Previous version expected a list, this needs consistency check with MigraineMoEModel)
            training (bool): Whether the model is in training mode
            
        Returns:
            tuple: (gate_weights, load_balance_loss)
        """
        # Ensure input is a single tensor (consistency check needed)
        if isinstance(inputs, list):
            # This indicates the inconsistency from MigraineMoEModel's train_step persists.
            # For now, using the first element assuming it's the intended flattened input.
            # TODO: Fix the input preparation in MigraineMoEModel.train_step
            print("Warning: GatingNetwork received list input, expected tensor. Using first element.")
            inputs = inputs[0] 
            
        # First hidden layer
        x = self.dense1(inputs) # Apply directly to the input tensor
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.dropout1(x, training=training)
        
        # Second hidden layer
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)
        x = self.dropout2(x, training=training)
        
        # Output layer (gating weights)
        gate_weights = self.gate_output(x) # Softmax output, shape (batch_size, num_experts)
        
        # Calculate Load Balancing Loss using tf.cond for graph compatibility
        batch_size = tf.cast(tf.shape(gate_weights)[0], tf.float32)

        def compute_loss():
            f_i = tf.reduce_mean(gate_weights, axis=0) # Shape (num_experts,)
            P_i = tf.reduce_mean(gate_weights, axis=0) # Shape (num_experts,)
            # Loss calculation: N * sum(f_i * P_i)
            return tf.cast(self.num_experts, tf.float32) * tf.reduce_sum(f_i * P_i)

        def zero_loss():
            return tf.constant(0.0, dtype=tf.float32)

        # Use tf.cond to handle potential zero batch size
        load_balance_loss = tf.cond(tf.equal(batch_size, 0.0), zero_loss, compute_loss)
            
        # Add small epsilon to prevent potential NaN in gradient if loss is exactly zero?
        # load_balance_loss += 1e-8 
        
        return gate_weights, load_balance_loss
    
    def apply_sparse_gating(self, gate_weights):
        """
        Apply sparse gating to select top-k experts.
        
        Args:
            gate_weights (tensor): Gating weights for each expert
            
        Returns:
            tensor: Sparse gating weights (zeros for non-selected experts)
        """
        # Get top-k experts
        top_k = min(self.config['top_k'], self.num_experts)
        _, indices = tf.math.top_k(gate_weights, k=top_k)
        
        # Create mask for selected experts
        mask = tf.reduce_sum(
            tf.one_hot(indices, depth=self.num_experts),
            axis=1
        )
        
        # Apply mask to gate weights (zero out non-selected experts)
        sparse_gates = gate_weights * mask
        
        # Normalize gates to sum to 1
        normalized_gates = sparse_gates / (tf.reduce_sum(sparse_gates, axis=1, keepdims=True) + 1e-8)
        
        return normalized_gates
    
    def get_config(self):
        """Get the configuration of the Gating Network."""
        config = super(GatingNetwork, self).get_config()
        config.update({
            "config": self.config,
            "num_experts": self.num_experts
        })
        return config


class FusionMechanism(layers.Layer):
    """
    Fusion Mechanism that combines expert outputs based on gating weights.
    
    Attributes:
        top_k (int): Number of experts to select
    """
    
    def __init__(self, top_k=2, name="fusion_mechanism"):
        """
        Initialize the Fusion Mechanism.
        
        Args:
            top_k (int): Number of experts to select
            name (str): Name of the fusion mechanism
        """
        super(FusionMechanism, self).__init__(name=name)
        self.top_k = top_k
        
    def call(self, expert_outputs, gate_outputs):
        """
        Fuse expert outputs using sparse gating (top-k selection).
        
        Args:
            expert_outputs (list): List of tensors from each expert
            gate_outputs (tensor): Tensor of gating weights for each expert
            
        Returns:
            tensor: Fused output tensor
        """
        # Get top-k experts
        _, indices = tf.math.top_k(gate_outputs, k=self.top_k)
        
        # Create mask for selected experts
        mask = tf.reduce_sum(
            tf.one_hot(indices, depth=tf.shape(gate_outputs)[1]),
            axis=1
        )
        
        # Apply mask to gate outputs (zero out non-selected experts)
        sparse_gates = gate_outputs * mask
        
        # Normalize gates to sum to 1
        normalized_gates = sparse_gates / (tf.reduce_sum(sparse_gates, axis=1, keepdims=True) + 1e-8)
        
        # Apply gates to expert outputs and sum
        gated_outputs = []
        for i in range(len(expert_outputs)):
            # Expand dimensions for broadcasting
            gates_expanded = tf.expand_dims(normalized_gates[:, i], axis=1)
            gated_output = expert_outputs[i] * gates_expanded
            gated_outputs.append(gated_output)
        
        # Sum all gated outputs
        fused_output = tf.add_n(gated_outputs)
        
        return fused_output
    
    def get_config(self):
        """Get the configuration of the Fusion Mechanism."""
        config = super(FusionMechanism, self).get_config()
        config.update({"top_k": self.top_k})
        return config


if __name__ == "__main__":
    # Example usage
    num_experts = 3
    
    config = {
        'hidden_size': 64,
        'top_k': 2,
        'dropout_rate': 0.2
    }
    
    # Create the gating network
    gating_network = GatingNetwork(num_experts=num_experts, config=config)
    
    # Create the fusion mechanism
    fusion_mechanism = FusionMechanism(top_k=config['top_k'])
    
    # Build the models with sample inputs
    # Sample inputs for each modality
    sleep_input = tf.random.normal((32, 10))  # (batch_size, sleep_features)
    weather_input = tf.random.normal((32, 8))  # (batch_size, weather_features)
    stress_diet_input = tf.random.normal((32, 12))  # (batch_size, stress_diet_features)
    
    # Get gating weights
    gate_weights, load_balance_loss = gating_network([sleep_input, weather_input, stress_diet_input])
    
    # Sample expert outputs
    expert_outputs = [
        tf.random.normal((32, 64)),  # Sleep expert output
        tf.random.normal((32, 64)),  # Weather expert output
        tf.random.normal((32, 64))   # Stress/Diet expert output
    ]
    
    # Fuse expert outputs
    fused_output = fusion_mechanism(expert_outputs, gate_weights)
    
    print(f"Models built successfully!")
    print(f"Gate weights shape: {gate_weights.shape}")
    print(f"Fused output shape: {fused_output.shape}")
    
    # Summary
    gating_network.summary()
