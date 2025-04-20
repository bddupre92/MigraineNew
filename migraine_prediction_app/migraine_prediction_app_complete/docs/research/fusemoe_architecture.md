# FuseMoE Architecture Research

## Overview
FuseMoE is a multimodal Mixture-of-Experts framework that allows for modular expertise per modality for scalable data fusion. The repository contains implementations for various MoE architectures including sparse MoE and hierarchical MoE.

## Key Components

### 1. Sparse Mixture of Experts (sparse_moe.py)
- Implements the Sparsely-Gated Mixture-of-Experts as described in "Outrageously Large Neural Networks" (https://arxiv.org/abs/1701.06538)
- Key class: `SparseDispatcher`
  - Handles routing of inputs to experts based on gates
  - Combines outputs from experts weighted by gate values
  - Supports efficient handling of sparse gate matrices

### 2. Model Architecture (model.py)
- Contains implementations for various model architectures
- Includes `BertForRepresentation` for text representation

### 3. Training Framework (train.py)
- Implements training and evaluation functions
- Supports different modality combinations (TS_Text, TS_CXR, TS_CXR_Text)
- Handles model checkpointing and result tracking

### 4. Data Processing (preprocessing/data.py)
- Implements data preparation and loading
- Includes `TSNote_Irg` dataset class for handling time-series data
- Supports data imputation for handling missing values

## Integration Points for Migraine Prediction

For our migraine prediction application, we'll need to:

1. **Modify Expert Definitions**: Create specialized experts for each modality (Sleep, Weather, Physio, Stress/Diet, EHR)
2. **Parameterize Configuration**: Extend configuration to accept hyperparameters from PyGMO optimization
3. **Adapt Training Loop**: Modify to emit performance metrics for PyGMO fitness evaluation
4. **Implement Data Generators**: Create synthetic data generators for each modality

## Files to Refactor (as per PRD)

1. `train.py`: Parameterize via JSON configs; emit results.json with validation metrics
2. `modeling_fusemoe.py`: Abstract expert definitions into classes; accept hyperparams from config
3. `configuration_fusemoe.py`: Extend to parse external optimized_config.json
4. `run_fusemoe.py`: Accept --config_path and --mode flags; write results.json on completion

## Next Steps

1. Research Synthea capabilities for generating synthetic EHR data
2. Research PyGMO optimization framework for hyperparameter tuning
3. Design modality-specific experts for migraine prediction
4. Implement integration between FuseMoE and PyGMO
