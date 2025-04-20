# Hierarchical Mixture of Experts (HMoE) Implementation for Migraine Prediction

This document outlines a proposed implementation of Hierarchical Mixture of Experts (HMoE) architecture to enhance the migraine prediction model.

## Overview

The Hierarchical Mixture of Experts (HMoE) extends the current Mixture of Experts (MoE) architecture by introducing multiple levels of expert specialization. This allows the model to capture both domain-specific patterns and cross-domain interactions at different levels of abstraction.

## Architecture Design

### Level 1: Temporal Hierarchy
The first level separates data by time scale:
- **Immediate Factors Experts**: Focus on short-term triggers (24-48 hours before migraine)
- **Medium-term Factors Experts**: Focus on patterns over 3-7 days
- **Long-term Factors Experts**: Focus on patterns over weeks to months

### Level 2: Domain Hierarchy
Within each temporal expert, we have domain-specific experts:
- **Sleep Expert**
- **Weather Expert**
- **Stress/Diet Expert**
- **Physiological Expert**

### Gating Networks
- **Level 1 Gating**: Routes inputs to appropriate temporal experts
- **Level 2 Gating**: Routes temporal features to domain experts
- **Final Gating**: Combines all expert outputs for final prediction

## Implementation Plan

### 1. Data Preprocessing
- Segment data into different temporal windows
- Apply feature engineering specific to each temporal scale
- Create specialized feature sets for each expert

### 2. Expert Models
```python
class TemporalExpert(tf.keras.Model):
    def __init__(self, domain_experts, gating_network):
        super(TemporalExpert, self).__init__()
        self.domain_experts = domain_experts
        self.gating_network = gating_network
        
    def call(self, inputs):
        # Get expert outputs
        expert_outputs = [expert(inputs) for expert in self.domain_experts]
        expert_outputs = tf.stack(expert_outputs, axis=1)
        
        # Get gating weights
        gating_weights = self.gating_network(inputs)
        
        # Combine expert outputs
        combined_output = tf.reduce_sum(expert_outputs * gating_weights, axis=1)
        return combined_output
```

### 3. Hierarchical Gating
```python
class HierarchicalGating(tf.keras.Model):
    def __init__(self, temporal_experts, final_gating):
        super(HierarchicalGating, self).__init__()
        self.temporal_experts = temporal_experts
        self.final_gating = final_gating
        
    def call(self, inputs):
        # Get temporal expert outputs
        temporal_outputs = [expert(inputs) for expert in self.temporal_experts]
        temporal_outputs = tf.stack(temporal_outputs, axis=1)
        
        # Get final gating weights
        final_weights = self.final_gating(inputs)
        
        # Combine temporal outputs
        final_output = tf.reduce_sum(temporal_outputs * final_weights, axis=1)
        return final_output
```

### 4. PyGMO Optimization
Extend the current PyGMO optimization to handle hierarchical architecture:

```python
class HierarchicalMoEOptimizationProblem:
    """
    PyGMO-compatible problem class for hierarchical MoE optimization.
    """
    
    def __init__(self, temporal_configs, domain_configs, train_data, val_data):
        self.temporal_configs = temporal_configs
        self.domain_configs = domain_configs
        self.train_data = train_data
        self.val_data = val_data
        
        # Set up hyperparameter search space
        self.param_bounds = self._get_hierarchical_bounds()
        self.integer_dims = 3  # Number of integer parameters
    
    def _get_hierarchical_bounds(self):
        # Define bounds for hierarchical parameters
        lb = [16, 16, 2, 0.001]  # First 3 are integers
        ub = [128, 128, 5, 0.1]
        return (lb, ub)
    
    def get_bounds(self):
        return self.param_bounds
    
    def get_nobj(self):
        return 2  # Multi-objective: AUC and latency
    
    def get_nix(self):
        return self.integer_dims
    
    def fitness(self, x):
        # Process parameters
        params = self._process_params(x)
        
        # Create hierarchical model with parameters
        model = self._create_hierarchical_model(params)
        
        # Train and evaluate model
        # ...
        
        # Return multi-objective fitness
        return [-auc, latency]
```

## Expected Benefits

1. **Improved Accuracy**: By specializing experts at multiple levels, the model can better capture complex patterns.
2. **Better Temporal Understanding**: Explicit modeling of different time scales helps identify both immediate triggers and long-term patterns.
3. **Enhanced Interpretability**: The hierarchical structure provides insights into which temporal scales and domains are most predictive.
4. **Reduced Overfitting**: The hierarchical structure acts as a form of regularization, reducing overfitting on limited data.

## Performance Metrics

We expect the following improvements over the current MoE architecture:
- AUC: Increase from 0.9325 to 0.95+ (2-3% improvement)
- F1 Score: Increase from 0.8659 to 0.89+ (3-4% improvement)
- Precision: Increase from 0.8571 to 0.88+ (3% improvement)
- Recall: Increase from 0.8750 to 0.90+ (3% improvement)

## Implementation Timeline

1. Data preprocessing and feature engineering: 1 week
2. Temporal expert implementation: 1 week
3. Hierarchical gating implementation: 1 week
4. PyGMO optimization extension: 1 week
5. Testing and validation: 1 week

Total estimated time: 5 weeks
