# Migraine Prediction App: Fixes Documentation

This document outlines the issues identified in the migraine prediction app and the fixes implemented to resolve them.

## Issue 1: Model Loading Problem

### Problem Description
The Streamlit dashboard was unable to load the trained model, resulting in the error:
```
Failed to load model from /home/ubuntu/migraine_demo/MigraineNew/migraine_prediction_project/migraine_prediction_app_complete/output/optimized_model.keras
Model loading returned None. Check model file integrity.
```

### Root Cause
The migraine prediction model uses a complex Mixture of Experts (MoE) architecture with custom TensorFlow components:
- GatingNetwork
- FusionMechanism
- MigraineMoEModel
- Expert models (SleepExpert, WeatherExpert, StressDietExpert)

These custom components need to be registered with TensorFlow's serialization system when loading the model, but the dashboard was trying to load the model without providing these custom objects.

### Fix Implemented
Modified the `load_model` function in `dashboard/streamlit_dashboard.py` to:
1. Import all necessary custom components
2. Register them in a `custom_objects` dictionary
3. Pass this dictionary to `tf.keras.models.load_model()`
4. Add better error handling with detailed exception information

```python
def load_model(model_path):
    """Load a saved model with custom objects."""
    try:
        # Import custom model components
        from model.moe_architecture.gating_network import GatingNetwork, FusionMechanism
        from model.moe_architecture.pygmo_integration import MigraineMoEModel
        from model.moe_architecture.experts.sleep_expert import SleepExpert
        from model.moe_architecture.experts.weather_expert import WeatherExpert
        from model.moe_architecture.experts.stress_diet_expert import StressDietExpert
        
        # Create custom objects dictionary for model loading
        custom_objects = {
            'GatingNetwork': GatingNetwork,
            'FusionMechanism': FusionMechanism,
            'MigraineMoEModel': MigraineMoEModel,
            'SleepExpert': SleepExpert,
            'WeatherExpert': WeatherExpert,
            'StressDietExpert': StressDietExpert
        }
        
        # Load the model with custom objects
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        st.success(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        st.error(f"Failed to load model from {model_path}")
        st.exception(e)
        return None
```

## Issue 2: Missing Test Predictions File

### Problem Description
The dashboard was looking for a test predictions file that wasn't being generated during model training:
```
Test predictions file not found at /home/ubuntu/migraine_demo/MigraineNew/migraine_prediction_project/migraine_prediction_app_complete/output/test_predictions.npz. Cannot display performance metrics.
```

### Root Cause
The performance metrics code wasn't saving test predictions to a file that the dashboard could use for visualization.

### Fix Implemented
1. Modified the `_save_metrics` method in `model/performance_metrics.py` to:
   - Accept an `X_test_list` parameter
   - Save test predictions, actual values, and test data to a `.npz` file
   - Add error handling for the saving process

```python
def _save_metrics(self, metrics, y_true, y_pred_prob, X_test_list=None):
    """
    Save metrics and generate plots.
    
    Args:
        metrics (dict): Dictionary of metrics
        y_true (array): True labels
        y_pred_prob (array): Predicted probabilities
        X_test_list (list, optional): List of test data for each expert
    """
    # Create metrics directory
    metrics_dir = os.path.join(self.output_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({k: [v] for k, v in metrics.items() 
                              if not isinstance(v, (np.ndarray, list, tuple))})
    metrics_df.to_csv(os.path.join(metrics_dir, 'performance_metrics.csv'), index=False)
    
    # Save test predictions for dashboard visualization
    if X_test_list is not None:
        try:
            # Convert to numpy arrays if they're not already
            y_true_np = y_true.numpy() if hasattr(y_true, 'numpy') else np.array(y_true)
            y_pred_prob_np = y_pred_prob.numpy() if hasattr(y_pred_prob, 'numpy') else np.array(y_pred_prob)
            
            # Save test predictions, actual values, and test data
            np.savez(
                os.path.join(self.output_dir, 'test_predictions.npz'),
                y_true=y_true_np,
                y_pred=y_pred_prob_np,
                # Save a simplified version of X_test_list for visualization
                X_test_sleep=X_test_list[0] if len(X_test_list) > 0 else np.array([]),
                X_test_weather=X_test_list[1] if len(X_test_list) > 1 else np.array([]),
                X_test_stress_diet=X_test_list[2] if len(X_test_list) > 2 else np.array([])
            )
            print(f"Test predictions saved to {os.path.join(self.output_dir, 'test_predictions.npz')}")
        except Exception as e:
            print(f"Error saving test predictions: {e}")
    
    # Generate and save plots
    self._generate_roc_curve_plot(metrics, metrics_dir)
    self._generate_precision_recall_curve(y_true, y_pred_prob, metrics_dir)
    self._generate_confusion_matrix_plot(metrics, metrics_dir)
    self._generate_threshold_analysis_plot(y_true, y_pred_prob, metrics_dir)
    self._generate_performance_summary_plot(metrics, metrics_dir)
```

2. Updated the `calculate_metrics` method to pass `X_test_list` to `_save_metrics`:

```python
# Save metrics and test predictions
self._save_metrics(all_metrics, y_test, y_pred_prob, X_test_list)
```

3. Created a script (`generate_test_predictions.py`) to generate the test predictions file:
   - Loads the trained model with custom objects
   - Loads test data
   - Runs model evaluation to generate predictions
   - Saves test predictions to a file using the updated code

## Remaining Issues

When running the test predictions generation script, we encountered an input shape incompatibility error:

```
ValueError: Exception encountered when calling Sequential.call().
Invalid input shape for input Tensor("data:0", shape=(32, 7, 6), dtype=float32). Expected shape (None, 10), but input has incompatible shape (32, 7, 6)
```

This suggests there's a mismatch between:
1. The model's expected input shape (a single tensor with shape (None, 10))
2. The actual data being provided (a list of three tensors with shapes (32, 7, 6), (32, 4), and (32, 7, 6))

### Potential Solutions

1. **Model Input Wrapper**: Create a wrapper function that properly formats the input data to match what the model expects.

2. **Model Adaptation**: Modify how the model is loaded to handle the list of inputs correctly.

3. **Custom Prediction Function**: Implement a custom prediction function that handles the input format conversion.

4. **Dashboard Adaptation**: Update the dashboard to work with the available data without requiring the test predictions file.

## Next Steps

1. Investigate the input shape incompatibility issue
2. Implement one of the potential solutions
3. Complete the test predictions file generation
4. Verify the dashboard works with all fixes implemented
5. Finalize the documentation and deliver the complete solution
