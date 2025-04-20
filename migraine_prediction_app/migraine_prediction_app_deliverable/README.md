# Migraine Prediction App

## Overview

This repository contains the implementation of a migraine prediction application using a Mixture of Experts (MoE) architecture with FuseMoE, synthetic data generation inspired by Synthea, and hyperparameter optimization with PyGMO.

## Key Features

- Multi-modal data fusion using specialized expert networks
- High-sensitivity prediction of migraine events (≥95% for high-risk days)
- Optimized performance with AUC ≥0.80 and F1-score ≥0.75
- Fast inference with latency <200ms
- Interpretable predictions with expert contribution analysis

## Directory Structure

- `code/`: Source code for the migraine prediction app
  - `data_generator/`: Synthetic data generation modules
  - `moe_architecture/`: MoE architecture implementation
  - `migraine_prediction_model.py`: Main prediction model
  - `performance_metrics.py`: Performance metrics implementation
  - `optimized_model.py`: Optimized model implementation
  - `test_model.py`: Test script for model evaluation
- `docs/`: Documentation
  - `documentation.md`: Comprehensive documentation
  - `research/`: Research notes on FuseMoE, Synthea, and PyGMO
- `examples/`: Example data and usage examples
- `requirements.txt`: Required Python packages

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/migraine-prediction-app.git
cd migraine-prediction-app

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
# Generate synthetic data
from code.data_generator.synthetic_data_generator import SyntheticDataGenerator

data_generator = SyntheticDataGenerator(
    num_patients=100,
    days=180,
    output_dir='./data',
    seed=42
)
data_generator.generate_data()

# Train optimized model
from code.optimized_model import OptimizedMigrainePredictionModel

optimizer = OptimizedMigrainePredictionModel(
    data_dir='./data',
    output_dir='./output',
    seed=42
)
final_metrics = optimizer.run_optimization()
```

## Documentation

For detailed documentation, please see [docs/documentation.md](docs/documentation.md).

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- FuseMoE: https://github.com/aaronhan223/FuseMoE
- Synthea: https://github.com/synthetichealth/synthea
- PyGMO: https://esa.github.io/pygmo2/
