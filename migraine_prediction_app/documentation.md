# Migraine Prediction App Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
3. [Synthetic Data Generation](#synthetic-data-generation)
4. [Mixture of Experts (MoE) Architecture](#mixture-of-experts-moe-architecture)
5. [PyGMO Integration for Optimization](#pygmo-integration-for-optimization)
6. [Performance Metrics](#performance-metrics)
7. [Optimized Model](#optimized-model)
8. [Usage Instructions](#usage-instructions)
9. [API Reference](#api-reference)
10. [Troubleshooting](#troubleshooting)

## Introduction

The Migraine Prediction App is a machine learning system designed to predict migraine occurrences based on multiple data modalities including sleep patterns, weather conditions, and stress/dietary factors. The system uses a Mixture of Experts (MoE) architecture implemented with FuseMoE, synthetic data generation with Synthea-inspired techniques, and hyperparameter optimization with PyGMO.

### Key Features

- Multi-modal data fusion using specialized expert networks
- High-sensitivity prediction of migraine events (≥95% for high-risk days)
- Optimized performance with AUC ≥0.80 and F1-score ≥0.75
- Fast inference with latency <200ms
- Interpretable predictions with expert contribution analysis

### System Requirements

- Python 3.10+
- TensorFlow 2.19+
- PyGMO 2.19+
- NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn

## System Architecture

The Migraine Prediction App consists of several interconnected components:

1. **Data Generation Layer**: Creates synthetic data for training and testing
2. **Data Processing Layer**: Preprocesses and transforms data for each expert
3. **Model Layer**: Contains the MoE architecture with expert networks and gating
4. **Optimization Layer**: Uses PyGMO for hyperparameter optimization
5. **Evaluation Layer**: Implements performance metrics and visualization
6. **Interface Layer**: Provides prediction API and visualization tools

![System Architecture](system_architecture.png)

## Synthetic Data Generation

The synthetic data generator creates realistic migraine-related data across multiple modalities.

### Data Modalities

1. **Sleep Data**:
   - Total sleep hours
   - Deep sleep percentage
   - REM sleep percentage
   - Light sleep percentage
   - Awake time minutes
   - Sleep quality score

2. **Weather Data**:
   - Temperature
   - Humidity
   - Barometric pressure
   - Pressure change over 24 hours

3. **Stress/Dietary Data**:
   - Stress level (1-10 scale)
   - Alcohol consumption (boolean/amount)
   - Caffeine consumption (boolean/amount)
   - Chocolate consumption (boolean/amount)
   - Processed food consumption (boolean/amount)
   - Water consumption (liters)

### Migraine Triggers

The data generator implements the following migraine triggers with specified correlations:

- **Barometric pressure drops** ≥ 5 hPa within 24h (increases next-day migraine probability by 15-20%)
- **Sleep disruptions** (< 5h or > 9h sleep) (increases next-day migraine probability by 15-20%)
- **Stress spikes** (sudden +3-5 points on a 1-10 scale) (increases next-day migraine probability by 15-20%)
- **Dietary triggers** (alcohol, caffeine, chocolate intake) (increases next-day migraine probability by 15-20%)

### Patient Distribution

- 15% chronic patients (≥15 migraine days/month)
- 85% episodic patients (<15 migraine days/month)
- 30% with aura, 70% without aura

### Implementation Details

The synthetic data generator is implemented in the following modules:

- `patient_profile_generator.py`: Creates patient profiles with appropriate distributions
- `migraine_event_generator.py`: Generates migraine events based on patient susceptibility and triggers
- `sleep_data_generator.py`: Generates sleep data with disruption patterns
- `weather_data_generator.py`: Generates weather data with barometric pressure patterns
- `stress_diet_generator.py`: Generates stress levels and dietary trigger patterns
- `correlation_engine.py`: Ensures proper correlation between modalities and migraine events
- `data_export_module.py`: Exports data to CSV files for model training
- `synthetic_data_generator.py`: Main module that orchestrates the data generation process

## Mixture of Experts (MoE) Architecture

The Mixture of Experts architecture divides the prediction task among specialized expert networks, each focusing on a specific data modality.

### Expert Networks

1. **Sleep Expert**:
   - Architecture: 1D-CNN → Bi-LSTM
   - Input: 7-day sequence of sleep metrics
   - Output: 64-dimensional embedding
   - Optimized hyperparameters: conv_filters=96, kernel_size=5, lstm_units=192, dropout_rate=0.3

2. **Weather Expert**:
   - Architecture: 3-layer MLP with residual connections
   - Input: Weather metrics
   - Output: 64-dimensional embedding
   - Optimized hyperparameters: hidden_units=192, activation='elu', dropout_rate=0.3

3. **Stress/Diet Expert**:
   - Architecture: Small Transformer encoder
   - Input: 7-day sequence of stress and dietary metrics
   - Output: 64-dimensional embedding
   - Optimized hyperparameters: embedding_dim=96, num_heads=6, transformer_dim=96, dropout_rate=0.3

### Gating Network

- Architecture: Feed-forward network
- Input: Concatenated features from all modalities
- Output: Weight distribution across experts
- Optimized hyperparameters: hidden_size=192, top_k=3, dropout_rate=0.2

### Fusion Mechanism

- Sparse fusion that selects the top-k experts based on gating network outputs
- Weighted combination of expert outputs
- Final prediction through a sigmoid activation layer

### Implementation Details

The MoE architecture is implemented in the following modules:

- `experts/sleep_expert.py`: Implementation of the Sleep Expert
- `experts/weather_expert.py`: Implementation of the Weather Expert
- `experts/stress_diet_expert.py`: Implementation of the Stress/Diet Expert
- `gating_network.py`: Implementation of the Gating Network and Fusion Mechanism
- `migraine_prediction_model.py`: Main model that integrates all components

## PyGMO Integration for Optimization

PyGMO (Python Parallel Global Multiobjective Optimizer) is used for hyperparameter optimization of the MoE architecture.

### Optimization Phases

1. **Expert Hyperparameter Optimization**:
   - Algorithm: Differential Evolution (DE) or CMA-ES
   - Objective: Maximize validation AUC for each expert independently
   - Parameters: Network architecture parameters (filters, units, etc.)

2. **Gating Hyperparameter Optimization**:
   - Algorithm: Particle Swarm Optimization (PSO)
   - Objective: Maximize validation AUC with fixed expert networks
   - Parameters: Gating network parameters and load balancing coefficient

3. **End-to-End MoE Optimization**:
   - Algorithm: NSGA-II (multi-objective)
   - Objectives: Maximize AUC, recall, F1-score; minimize inference time
   - Parameters: Learning rate, batch size, L2 regularization

### Implementation Details

The PyGMO integration is implemented in the `pygmo_integration.py` module, which defines:

- `ExpertHyperparamOptimization`: Class for optimizing expert hyperparameters
- `GatingHyperparamOptimization`: Class for optimizing gating hyperparameters
- `EndToEndMoEOptimization`: Class for end-to-end optimization
- `MigraineMoEModel`: Custom model class with load balancing loss

## Performance Metrics

The performance metrics module implements specialized metrics for evaluating the migraine prediction model.

### Key Metrics

1. **Standard Classification Metrics**:
   - AUC (Area Under ROC Curve) - Target: ≥0.80
   - Precision
   - Recall
   - F1-score - Target: ≥0.75
   - Specificity

2. **High-Risk Day Metrics**:
   - High-risk sensitivity (recall) - Target: ≥0.95
   - High-risk precision
   - High-risk F1-score
   - Percentage of days classified as high-risk

3. **Threshold-Optimized Metrics**:
   - Optimal threshold for F1-score
   - Metrics at optimal threshold

4. **Performance Targets**:
   - Overall performance score (percentage of targets met)
   - Target achievement status (>95% required)

### Visualization

The metrics module generates the following visualizations:

- ROC curve
- Precision-recall curve
- Confusion matrices (standard and high-risk thresholds)
- Threshold analysis plot
- Performance summary radar chart
- Metrics history during training

### Implementation Details

The performance metrics are implemented in the `performance_metrics.py` module, which defines:

- `MigrainePerformanceMetrics`: Class for calculating and visualizing metrics
- `PerformanceMetricsCallback`: Callback for monitoring metrics during training

## Optimized Model

The optimized model implements advanced techniques to achieve >95% performance accuracy.

### Optimization Techniques

1. **Enhanced Architecture**:
   - Increased capacity in expert networks
   - Optimized hyperparameters for each component
   - Improved gating network with better fusion

2. **Advanced Training**:
   - Focal loss for handling class imbalance
   - Class weighting for emphasizing migraine detection
   - Learning rate scheduling for optimal convergence
   - Increased regularization to prevent overfitting

3. **High-Risk Day Optimization**:
   - Threshold optimization for ≥0.95 sensitivity
   - Precision-sensitivity trade-off analysis

4. **Ensemble Approach**:
   - Multiple models with different random seeds
   - Weighted averaging of predictions

### Implementation Details

The optimized model is implemented in the `optimized_model.py` module, which defines:

- `OptimizedMigrainePredictionModel`: Class with optimized configurations and techniques
- `focal_loss`: Custom loss function for handling class imbalance
- `optimize_high_risk_threshold`: Method for threshold optimization
- `ensemble_predictions`: Method for creating ensemble predictions

## Usage Instructions

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/migraine-prediction-app.git
cd migraine-prediction-app

# Install dependencies
pip install -r requirements.txt
```

### Data Generation

```python
from data_generator.synthetic_data_generator import SyntheticDataGenerator

# Create data generator
data_generator = SyntheticDataGenerator(
    num_patients=200,
    days=180,
    output_dir='./data',
    seed=42
)

# Generate data
data_generator.generate_data()
```

### Model Training

```python
from migraine_prediction_model import MigrainePredictionModel

# Create and train model
model = MigrainePredictionModel(
    data_dir='./data',
    output_dir='./output',
    config={
        'optimize_experts': True,
        'optimize_gating': True,
        'optimize_end_to_end': True
    },
    seed=42
)

# Run the complete pipeline
metrics = model.run_pipeline()

# Print evaluation metrics
print(f"AUC: {metrics['auc']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1 Score: {metrics['f1_score']:.4f}")
```

### Optimized Model Training

```python
from optimized_model import OptimizedMigrainePredictionModel

# Create optimized model
optimizer = OptimizedMigrainePredictionModel(
    data_dir='./data',
    output_dir='./output',
    seed=42
)

# Run optimization
final_metrics = optimizer.run_optimization()

# Print final results
print(f"Final Performance Score: {final_metrics['performance_score']:.1f}%")
print(f"Target Met: {'Yes' if final_metrics['overall_target_met'] else 'No'}")
```

### Making Predictions

```python
import numpy as np
import tensorflow as tf

# Load saved model
model = tf.keras.models.load_model('./output/optimized_model')

# Prepare input data (example)
sleep_data = np.random.randn(1, 7, 6)  # 7-day sequence, 6 features
weather_data = np.random.randn(1, 4)   # 4 features
stress_diet_data = np.random.randn(1, 7, 6)  # 7-day sequence, 6 features

# Make prediction
prediction = model.predict([sleep_data, weather_data, stress_diet_data])
print(f"Migraine probability: {prediction[0][0]:.4f}")
```

## API Reference

### Data Generator API

#### `SyntheticDataGenerator`

```python
SyntheticDataGenerator(
    num_patients=100,
    days=180,
    output_dir='./data',
    seed=None
)
```

**Methods**:
- `generate_data()`: Generate synthetic data for all modalities
- `save_data()`: Save generated data to CSV files

### Model API

#### `MigrainePredictionModel`

```python
MigrainePredictionModel(
    data_dir='./data',
    output_dir='./output',
    config=None,
    seed=None
)
```

**Methods**:
- `load_data()`: Load and preprocess data
- `optimize_experts()`: Optimize expert hyperparameters
- `optimize_gating()`: Optimize gating hyperparameters
- `optimize_end_to_end()`: Perform end-to-end optimization
- `build_model()`: Build the MoE model
- `train_model()`: Train the model
- `evaluate_model()`: Evaluate the model on test data
- `save_model()`: Save the trained model
- `load_model()`: Load a trained model
- `predict()`: Make predictions with the model
- `run_pipeline()`: Run the complete pipeline

#### `OptimizedMigrainePredictionModel`

```python
OptimizedMigrainePredictionModel(
    data_dir='./data',
    output_dir='./output',
    seed=42
)
```

**Methods**:
- `focal_loss()`: Create a focal loss function
- `build_optimized_model()`: Build the optimized model
- `train_optimized_model()`: Train the optimized model
- `optimize_high_risk_threshold()`: Optimize the high-risk threshold
- `ensemble_predictions()`: Create ensemble predictions
- `run_optimization()`: Run the complete optimization process

### Performance Metrics API

#### `MigrainePerformanceMetrics`

```python
MigrainePerformanceMetrics(
    output_dir='./output',
    config=None
)
```

**Methods**:
- `calculate_metrics()`: Calculate comprehensive performance metrics
- `_calculate_standard_metrics()`: Calculate standard classification metrics
- `_calculate_high_risk_metrics()`: Calculate metrics for high-risk days
- `_calculate_threshold_optimized_metrics()`: Calculate metrics with optimized threshold
- `_calculate_performance_targets()`: Calculate whether performance targets are met
- `_save_metrics()`: Save metrics and generate plots

#### `PerformanceMetricsCallback`

```python
PerformanceMetricsCallback(
    metrics_tracker,
    validation_data,
    log_dir='./logs'
)
```

**Methods**:
- `on_epoch_end()`: Calculate and log metrics at the end of each epoch
- `_plot_metrics_history()`: Generate and save metrics history plot

## Troubleshooting

### Common Issues

1. **Missing Dependencies**:
   - Error: `ModuleNotFoundError: No module named 'tensorflow'`
   - Solution: Install required packages with `pip install tensorflow numpy pandas matplotlib seaborn scikit-learn pygmo`

2. **Memory Issues**:
   - Error: `ResourceExhaustedError: OOM when allocating tensor`
   - Solution: Reduce batch size or model complexity in configuration

3. **Convergence Issues**:
   - Symptom: Model performance plateaus at suboptimal level
   - Solution: Try different learning rates, increase training epochs, or adjust regularization parameters

4. **Low Sensitivity**:
   - Symptom: High-risk day sensitivity below target (0.95)
   - Solution: Lower the high-risk threshold, increase class weight for positive class, or use focal loss with higher gamma

5. **Slow Inference**:
   - Symptom: Inference time exceeds target (200ms)
   - Solution: Reduce model complexity, optimize tensor operations, or consider model quantization

### Getting Help

For additional help or to report issues, please contact the development team or open an issue on the GitHub repository.

---

© 2025 Migraine Prediction App Team
