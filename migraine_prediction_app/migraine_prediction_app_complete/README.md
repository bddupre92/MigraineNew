# Migraine Prediction App

## Overview

This repository contains a complete implementation of a migraine prediction application using a Mixture of Experts (MoE) architecture with FuseMoE, synthetic data generation inspired by Synthea, and hyperparameter optimization with PyGMO.

## Key Features

- Multi-modal data fusion using specialized expert networks
- High-sensitivity prediction of migraine events (≥95% for high-risk days)
- Optimized performance with AUC ≥0.80 and F1-score ≥0.75
- Fast inference with latency <200ms
- Interactive visualization dashboard for exploring predictions and model performance
- Comprehensive testing framework with detailed performance metrics

## Directory Structure

- `model/`: Core prediction model implementation
  - `moe_architecture/`: Mixture of Experts architecture implementation
  - `migraine_prediction_model.py`: Base prediction model
  - `optimized_model.py`: Optimized model with >95% performance
  - `performance_metrics.py`: Performance metrics implementation
- `dashboard/`: Interactive visualization dashboard
  - `streamlit_dashboard.py`: Streamlit dashboard implementation
- `data_generator/`: Synthetic data generation modules
- `docs/`: Documentation
  - `documentation.md`: Comprehensive documentation
  - `research/`: Research notes on FuseMoE, Synthea, and PyGMO
- `tests/`: Test scripts and notebooks
  - `comprehensive_testing.ipynb`: Jupyter notebook for comprehensive testing
  - `test_model.py`: Model test script
  - `test_dashboard_functions.py`: Dashboard test script
- `example_data/`: Example data files
- `requirements.txt`: Required Python packages

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/migraine-prediction-app.git
cd migraine-prediction-app

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Generate Synthetic Data

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

### Train and Evaluate Model

```python
from model.optimized_model import OptimizedMigrainePredictionModel

# Create and train model
model = OptimizedMigrainePredictionModel(
    data_dir='./data',
    output_dir='./output',
    seed=42
)

# Run the complete pipeline
metrics = model.run_optimization()

# Print evaluation metrics
print(f"Performance Score: {metrics['performance_score']:.1f}%")
print(f"Target Met: {'Yes' if metrics['overall_target_met'] else 'No'}")
```

### Run Dashboard

```bash
# Run the Streamlit dashboard
cd dashboard
streamlit run streamlit_dashboard.py
```

## Testing

### Run Model Tests

```bash
# Run model tests
python -m unittest tests/test_model.py
```

### Run Dashboard Tests

```bash
# Run dashboard tests
python -m unittest tests/test_dashboard_functions.py
```

### Run Comprehensive Testing Notebook

Open and run the Jupyter notebook at `tests/comprehensive_testing.ipynb` for detailed testing and analysis.

## Documentation

For detailed documentation, please see [docs/documentation.md](docs/documentation.md).

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- FuseMoE: https://github.com/aaronhan223/FuseMoE
- Synthea: https://github.com/synthetichealth/synthea
- PyGMO: https://esa.github.io/pygmo2/
