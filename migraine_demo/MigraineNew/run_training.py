import sys
import os
import json
import tensorflow as tf
import numpy as np
import pandas as pd

# Enable numeric checks
# tf.debugging.enable_check_numerics() # Commented out for now

# Ensure the project root is in the Python path
project_root_dir = '/app/migraine_prediction_project/migraine_prediction_app_complete'
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

print(f'Sys path: {sys.path}')

try:
    # Import the necessary class
    from model.optimized_model import OptimizedMigrainePredictionModel
    print('Import successful.')

    # Define directories (absolute paths inside container)
    DATA_DIR = '/app/data'
    OUTPUT_DIR = '/app/output'
    SEED = 42

    print('Running model optimization...')
    # Instantiate and run the optimization
    model_optimizer = OptimizedMigrainePredictionModel(data_dir=DATA_DIR, output_dir=OUTPUT_DIR, seed=SEED)
    metrics = model_optimizer.run_optimization()

    print('Model training complete.')
    print('Final Metrics:', metrics)

    # Optionally save metrics to a file
    try:
        metrics_file = os.path.join(OUTPUT_DIR, 'final_metrics.json')
        # Convert numpy arrays in metrics if necessary before saving to JSON
        serializable_metrics = {k: (v.tolist() if isinstance(v, (np.ndarray, np.generic)) else v) for k, v in metrics.items()}
        with open(metrics_file, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)
        print(f'Final metrics saved to {metrics_file}')
    except Exception as e:
        print(f'Could not save metrics to JSON: {e}')

except ImportError as e:
    print(f"Import Error: {e}")
    print("Please check the PYTHONPATH and module availability.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during training: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Python training script finished successfully.")
sys.exit(0) 