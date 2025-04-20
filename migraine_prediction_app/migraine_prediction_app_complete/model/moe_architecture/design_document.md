# Mixture of Experts (MoE) Architecture Design for Migraine Prediction

## Overview

This document outlines the design for the Mixture of Experts (MoE) architecture that will be used for migraine prediction. The architecture will integrate multiple expert networks, each specialized for a different data modality, with a gating network that determines the contribution of each expert to the final prediction.

## Architecture Components

The MoE architecture consists of the following components:

1. **Expert Networks**: Specialized neural networks for each data modality
2. **Gating Network**: Determines the weight/contribution of each expert
3. **Fusion Mechanism**: Combines expert outputs based on gating weights
4. **Training Framework**: Handles end-to-end training of the entire model

## Expert Networks

For Phase 1 (MVP), we will implement three expert networks as specified in the requirements:

### 1. Sleep Expert

```
Input: Sleep data features
- total_sleep_hours
- deep_sleep_pct
- rem_sleep_pct
- light_sleep_pct
- awake_time_mins
- sleep_quality

Architecture: 1D-CNN → Bi-LSTM
- Input layer: Shape (sequence_length, num_features)
- 1D Convolutional layers:
  - Conv1D(filters=16-128, kernel_size=3-7)
  - BatchNormalization()
  - Activation('relu')
  - MaxPooling1D()
- Bidirectional LSTM layers:
  - Bidirectional(LSTM(units=32-256))
  - Dropout(0.1-0.5)
- Output layer: Dense(units=64)

Hyperparameters to optimize:
- Conv filters: [16, 32, 64, 128]
- Kernel size: [3, 5, 7]
- LSTM units: [32, 64, 128, 256]
- Dropout rate: [0.1, 0.2, 0.3, 0.4, 0.5]
```

### 2. Weather Expert

```
Input: Weather data features
- temperature
- humidity
- pressure
- pressure_change_24h

Architecture: 3-layer MLP with residual connections
- Input layer: Shape (num_features,)
- MLP layers with residual connections:
  - Dense(units=32-256)
  - BatchNormalization()
  - Activation('relu')
  - Residual connection
- Output layer: Dense(units=64)

Hyperparameters to optimize:
- Hidden units: [32, 64, 128, 256]
- Activation function: ['relu', 'elu']
- Dropout rate: [0.1, 0.2, 0.3, 0.4, 0.5]
```

### 3. Stress/Diet Expert

```
Input: Stress and dietary features
- stress_level
- alcohol_consumed
- caffeine_consumed
- chocolate_consumed
- processed_food_consumed
- water_consumed_liters

Architecture: Small Transformer encoder
- Input layer: Shape (sequence_length, num_features)
- Embedding layer: Dense(units=32-128)
- Positional encoding
- Transformer encoder layers:
  - MultiHeadAttention(heads=2-8)
  - Dense(units=32-128)
  - LayerNormalization()
  - Dropout(0.1-0.4)
- Output layer: Dense(units=64)

Hyperparameters to optimize:
- Embedding dimension: [32, 64, 128]
- Number of heads: [2, 4, 8]
- Transformer dimension: [32, 64, 128]
- Dropout rate: [0.1, 0.2, 0.3, 0.4]
```

## Gating Network

The gating network determines the contribution of each expert to the final prediction:

```
Input: Features from all modalities
- Sleep features
- Weather features
- Stress/Diet features

Architecture: Feed-forward network
- Input layer: Concatenated features from all modalities
- Hidden layers:
  - Dense(units=32-256)
  - BatchNormalization()
  - Activation('relu')
  - Dropout(0.1-0.4)
- Output layer: Dense(units=num_experts, activation='softmax')

Hyperparameters to optimize:
- Gate hidden size: [32, 64, 128, 256]
- Gate top-k: [1, 2, 3] (number of experts to select)
- Load balance coefficient: [0.001, 0.01, 0.1] (penalty for uneven expert usage)
```

## Fusion Mechanism

We will implement a sparse fusion mechanism that selects the top-k experts based on the gating network outputs:

```python
def sparse_fusion(expert_outputs, gate_outputs, top_k=2):
    """
    Fuse expert outputs using sparse gating (top-k selection).
    
    Args:
        expert_outputs: List of tensors from each expert
        gate_outputs: Tensor of gating weights for each expert
        top_k: Number of experts to select
        
    Returns:
        Fused output tensor
    """
    # Get top-k experts
    _, indices = tf.math.top_k(gate_outputs, k=top_k)
    
    # Create mask for selected experts
    mask = tf.reduce_sum(tf.one_hot(indices, depth=len(expert_outputs)), axis=1)
    
    # Apply mask to gate outputs (zero out non-selected experts)
    sparse_gates = gate_outputs * mask
    
    # Normalize gates to sum to 1
    normalized_gates = sparse_gates / tf.reduce_sum(sparse_gates, axis=1, keepdims=True)
    
    # Apply gates to expert outputs and sum
    gated_outputs = [expert_outputs[i] * normalized_gates[:, i:i+1] for i in range(len(expert_outputs))]
    fused_output = tf.add_n(gated_outputs)
    
    return fused_output
```

## Training Framework

The training framework will handle end-to-end training of the entire MoE model:

```python
class MigraineMoEModel(tf.keras.Model):
    def __init__(self, experts, gating_network, top_k=2, load_balance_coef=0.01):
        super().__init__()
        self.experts = experts
        self.gating_network = gating_network
        self.top_k = top_k
        self.load_balance_coef = load_balance_coef
        
    def call(self, inputs, training=False):
        # Process inputs through each expert
        expert_outputs = [expert(inputs[i], training=training) for i, expert in enumerate(self.experts)]
        
        # Get gating weights
        gate_outputs = self.gating_network(inputs, training=training)
        
        # Fuse expert outputs
        fused_output = sparse_fusion(expert_outputs, gate_outputs, self.top_k)
        
        # Final prediction layer
        prediction = tf.keras.layers.Dense(1, activation='sigmoid')(fused_output)
        
        return prediction, gate_outputs
        
    def train_step(self, data):
        # Custom training step with load balancing loss
        x, y = data
        
        with tf.GradientTape() as tape:
            # Forward pass
            y_pred, gate_outputs = self(x, training=True)
            
            # Compute main loss
            main_loss = self.compiled_loss(y, y_pred)
            
            # Compute load balancing loss
            # Encourages uniform expert utilization
            expert_usage = tf.reduce_mean(gate_outputs, axis=0)
            target_usage = tf.ones_like(expert_usage) / tf.cast(tf.shape(expert_usage)[0], tf.float32)
            load_balance_loss = tf.reduce_sum(tf.square(expert_usage - target_usage))
            
            # Total loss
            total_loss = main_loss + self.load_balance_coef * load_balance_loss
        
        # Compute gradients and update weights
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)
        
        # Return metrics
        results = {m.name: m.result() for m in self.metrics}
        results.update({'load_balance_loss': load_balance_loss})
        
        return results
```

## Integration with PyGMO

The MoE architecture will be optimized using PyGMO's evolutionary algorithms:

1. **Phase 1: Expert Hyperparameters (DE/CMA-ES)**
   - Optimize each expert's architecture independently
   - Fitness function: Validation accuracy on respective modality

2. **Phase 2: Gating Hyperparameters (PSO/ACO)**
   - Optimize gating network with fixed experts
   - Fitness function: Combined validation accuracy and expert utilization balance

3. **Phase 3: End-to-End MoE (Mixed Algorithm)**
   - Fine-tune entire model with best configurations
   - Fitness function: Multi-objective (accuracy, sensitivity, F1-score, inference time)

## Data Flow

1. **Input Processing**:
   - Sleep data → Sleep Expert
   - Weather data → Weather Expert
   - Stress/Diet data → Stress/Diet Expert

2. **Expert Processing**:
   - Each expert processes its modality-specific data
   - Experts output feature vectors of the same dimension (64)

3. **Gating Network**:
   - Takes features from all modalities
   - Outputs weights for each expert

4. **Fusion**:
   - Combines expert outputs based on gating weights
   - Produces final feature representation

5. **Prediction**:
   - Final dense layer converts fused representation to migraine probability

## Performance Metrics

As specified in the requirements, we will track:

1. **Validation AUC**: Target ≥ 0.80
2. **Sensitivity (Recall)** for "high-risk" days: Target ≥ 0.95
3. **F1-score** on the positive class: Target ≥ 0.75
4. **Inference latency**: Target < 200 ms per sample

## Implementation Plan

1. Implement base expert networks
2. Implement gating network
3. Implement fusion mechanism
4. Create end-to-end training framework
5. Integrate with PyGMO for hyperparameter optimization
6. Implement performance metrics tracking

## Next Steps

After completing this design, we will:

1. Implement the expert networks and gating network
2. Integrate PyGMO for optimization
3. Develop the complete migraine prediction model
4. Test and optimize the model to meet performance targets
