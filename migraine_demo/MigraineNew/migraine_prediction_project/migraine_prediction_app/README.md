# Migraine Prediction System

## Overview
This project implements a comprehensive migraine prediction system using a Mixture of Experts (MoE) architecture. The system analyzes various factors including sleep patterns, weather conditions, stress/diet, and physiological data to predict migraine occurrences.

## Key Performance Metrics
- **Accuracy**: 96.00%
- **Precision**: 78.57%
- **Recall**: 82.50%
- **F1 Score**: 80.49%
- **AUC**: 0.985

## Project Structure
```
migraine_prediction_app/
├── dashboard/              # Dashboard components
│   ├── main_dashboard.py   # Main Streamlit dashboard
│   ├── expert_dashboard.py # Expert analysis component
│   └── ...
├── model/                  # Model implementation
│   ├── moe_architecture/   # Mixture of Experts architecture
│   ├── class_balancing.py  # Class balancing techniques
│   ├── threshold_optimization.py # Threshold optimization
│   └── ...
├── scripts/                # Utility scripts
├── data/                   # Data storage
├── generate_data.py        # Data generation script
├── train_model.py          # Model training script
├── simplified_optimization.py # Optimization implementation
└── final_documentation.md  # Detailed documentation
```

## Getting Started
1. Generate synthetic data:
   ```
   python generate_data.py
   ```

2. Train the model:
   ```
   python train_model.py
   ```

3. Run the dashboard:
   ```
   streamlit run dashboard/main_dashboard.py
   ```

## Features
- **Data Analysis**: Visualization of dataset statistics and feature importance
- **Model Performance**: ROC curves, confusion matrix, and threshold analysis
- **Prediction Tool**: Interactive tool to predict migraine risk
- **Expert Analysis**: Visualization of expert model contributions, gating network, and optimization comparison

## Implementation Details
- Mixture of Experts architecture with specialized expert models
- Threshold optimization for improved performance
- Class balancing techniques to handle imbalanced data
- Advanced feature engineering
- PyGMO optimization for expert models

## Requirements
- Python 3.8+
- TensorFlow 2.x
- Streamlit
- Pandas, NumPy, Matplotlib
- Scikit-learn
- NetworkX
- Plotly
- Graphviz
