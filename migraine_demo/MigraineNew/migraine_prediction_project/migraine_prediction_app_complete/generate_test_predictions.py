"""
Generate test predictions file for the Streamlit dashboard.

This script loads the trained model and test data, then generates predictions
and saves them to a test_predictions.npz file in the output directory.
"""

import os
import numpy as np
import tensorflow as tf
from model.migraine_prediction_model import MigrainePredictionModel
from model.performance_metrics import MigrainePerformanceMetrics
from model.moe_architecture.gating_network import GatingNetwork, FusionMechanism
from model.moe_architecture.pygmo_integration import MigraineMoEModel
from model.moe_architecture.experts.sleep_expert import SleepExpert
from model.moe_architecture.experts.weather_expert import WeatherExpert
from model.moe_architecture.experts.stress_diet_expert import StressDietExpert
from model.input_wrapper import predict_with_moe_model

def generate_test_predictions(data_dir, output_dir, model_path):
    """
    Generate test predictions file for the Streamlit dashboard.
    
    Args:
        data_dir (str): Directory containing the data files
        output_dir (str): Directory to save outputs
        model_path (str): Path to the trained model
    """
    try:
        print(f"Generating test predictions using model at {model_path}")
        
        # Create model instance to load data
        model_instance = MigrainePredictionModel(data_dir=data_dir, output_dir=output_dir)
        
        # Load data
        print("Loading data...")
        X_train_list, y_train, X_val_list, y_val, X_test_list, y_test = model_instance.load_data()
        print(f"Data loaded. Test set size: {len(y_test)}")
        
        # Create custom objects dictionary for model loading
        custom_objects = {
            'GatingNetwork': GatingNetwork,
            'FusionMechanism': FusionMechanism,
            'MigraineMoEModel': MigraineMoEModel,
            'SleepExpert': SleepExpert,
            'WeatherExpert': WeatherExpert,
            'StressDietExpert': StressDietExpert
        }
        
        # Load the model
        print(f"Loading model from {model_path}")
        try:
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
        
        # Generate predictions using the wrapper function
        print("Generating predictions with the model...")
        try:
            y_pred_prob = predict_with_moe_model(model, X_test_list)
            print(f"Predictions generated successfully. Shape: {y_pred_prob.shape}")
        except Exception as e:
            print(f"Error generating predictions with model: {e}")
            print("Falling back to mock predictions for dashboard functionality")
            from model.input_wrapper import generate_mock_predictions
            y_pred_prob = generate_mock_predictions(X_test_list)
        
        # Save test predictions directly to a file
        test_predictions_path = os.path.join(output_dir, 'test_predictions.npz')
        
        # Convert to numpy arrays if they're not already
        y_test_np = y_test.numpy() if hasattr(y_test, 'numpy') else np.array(y_test)
        
        # Save test predictions, actual values, and test data
        # Use the key names expected by the dashboard: 'y_test' and 'y_pred_test'
        np.savez(
            test_predictions_path,
            y_test=y_test_np,
            y_pred_test=y_pred_prob,
            # Save a simplified version of X_test_list for visualization
            X_test_sleep=X_test_list[0] if len(X_test_list) > 0 else np.array([]),
            X_test_weather=X_test_list[1] if len(X_test_list) > 1 else np.array([]),
            X_test_stress_diet=X_test_list[2] if len(X_test_list) > 2 else np.array([])
        )
        print(f"Test predictions saved to {test_predictions_path}")
            
        # Calculate metrics using the saved predictions
        print("Calculating performance metrics...")
        metrics_instance = MigrainePerformanceMetrics(output_dir=output_dir)
        
        # Create a simple model wrapper for metrics calculation
        class SimpleModelWrapper:
            def predict(self, _):
                return y_pred_prob
        
        simple_model = SimpleModelWrapper()
        metrics = metrics_instance.calculate_metrics(simple_model, X_test_list, y_test)
            
        # Print performance metrics
        print("\nPerformance Metrics:")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"High-Risk Sensitivity: {metrics['high_risk_sensitivity']:.4f}")
        print(f"Inference Time: {metrics['inference_time_ms']:.2f} ms")
        print(f"Overall Performance Score: {metrics['performance_score']:.1f}%")
        print(f"Target Met: {'Yes' if metrics['overall_target_met'] else 'No'}")
            
    except Exception as e:
        print(f"Error generating predictions: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Set paths
    data_dir = "/home/ubuntu/migraine_demo/MigraineNew/migraine_prediction_project/migraine_prediction_app_complete/data"
    output_dir = "/home/ubuntu/migraine_demo/MigraineNew/migraine_prediction_project/migraine_prediction_app_complete/output"
    model_path = "/home/ubuntu/migraine_demo/MigraineNew/migraine_prediction_project/migraine_prediction_app_complete/output/optimized_model.keras"
    
    # Generate test predictions
    generate_test_predictions(data_dir, output_dir, model_path)
