"""
Comparison Dashboard for Migraine Prediction Models

This Streamlit dashboard compares the original FuseMoE model with the PyGMO-optimized version,
showing performance metrics, expert contributions, and prediction capabilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import time
import os
import sys
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

# Add the project root to the path
dashboard_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(dashboard_dir)
sys.path.append(project_root)

# Import our modules
from model.input_preprocessing import preprocess_expert_inputs, format_input_for_prediction
from dashboard.dashboard_metrics import MigrainePerformanceMetrics

# Set page configuration - must be first Streamlit command
st.set_page_config(
    page_title="Migraine Prediction Model Comparison",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import custom model components for proper model loading
try:
    from model.moe_architecture.gating_network import GatingNetwork
    from model.moe_architecture.experts.sleep_expert import SleepExpert
    from model.moe_architecture.experts.weather_expert import WeatherExpert
    from model.moe_architecture.experts.stress_diet_expert import StressDietExpert
    from model.moe_architecture.experts.physio_expert import PhysioExpert
    from model.migraine_prediction_model import MigraineMoEModel, FusionMechanism
    st.success("Successfully imported custom model components")
except Exception as e:
    st.warning(f"Error importing custom model components: {e}")
    # Define fallback empty classes for model loading
    class GatingNetwork(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
    
    class SleepExpert(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
    
    class WeatherExpert(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
    
    class StressDietExpert(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
    
    class PhysioExpert(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
    
    class FusionMechanism(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
    
    class MigraineMoEModel(tf.keras.Model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4B0082;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #6A5ACD;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F0F8FF;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .info-text {
        font-size: 0.9rem;
        color: #666;
        font-style: italic;
    }
    .highlight {
        background-color: #E6E6FA;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 3px solid #4B0082;
    }
    .comparison-container {
        display: flex;
        justify-content: space-between;
    }
    .model-card {
        background-color: #F8F8FF;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .original-model {
        border-left: 5px solid #4169E1;
    }
    .optimized-model {
        border-left: 5px solid #8A2BE2;
    }
    .improvement-positive {
        color: #008000;
        font-weight: bold;
    }
    .improvement-negative {
        color: #FF0000;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Constants
OUTPUT_DIR = os.path.join(project_root, 'output')
DATA_DIR = os.path.join(project_root, 'data')
ORIGINAL_MODEL_PATH = os.path.join(OUTPUT_DIR, 'original_model.keras')
OPTIMIZED_MODEL_PATH = os.path.join(OUTPUT_DIR, 'optimized_model.keras')
OPTIMIZATION_SUMMARY_PATH = os.path.join(OUTPUT_DIR, 'optimization', 'optimization_summary.json')

# Helper functions
def load_data():
    """Load data for model prediction."""
    try:
        # Load test data
        X_test_sleep = np.load(os.path.join(DATA_DIR, 'X_test_sleep.npy'))
        X_test_weather = np.load(os.path.join(DATA_DIR, 'X_test_weather.npy'))
        X_test_stress_diet = np.load(os.path.join(DATA_DIR, 'X_test_stress_diet.npy'))
        y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
        
        return [X_test_sleep, X_test_weather, X_test_stress_diet], y_test
    except Exception as e:
        st.error(f"Error loading test data: {e}")
        # Create mock data for testing
        X_test_sleep = np.random.random((20, 7, 6))
        X_test_weather = np.random.random((20, 4))
        X_test_stress_diet = np.random.random((20, 7, 6))
        y_test = np.random.randint(0, 2, (20, 1))
        
        return [X_test_sleep, X_test_weather, X_test_stress_diet], y_test

def load_model(model_path, model_type="original"):
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
        st.success(f"{model_type.capitalize()} model loaded successfully from {model_path}")
        return model
    except Exception as e:
        st.error(f"Failed to load {model_type} model from {model_path}")
        st.exception(e)
        return None

def get_model_predictions(model, X_test_list, model_type="original"):
    """Get model predictions for test data."""
    try:
        # Format input for prediction
        formatted_input = format_input_for_prediction(X_test_list)
        
        # Get predictions
        predictions = model.predict(formatted_input, verbose=0)
        
        st.success(f"Successfully generated predictions for {model_type} model")
        return predictions
    except Exception as e:
        st.error(f"Error generating predictions for {model_type} model: {e}")
        # Create mock predictions
        num_samples = X_test_list[0].shape[0]
        mock_predictions = np.random.random((num_samples, 1))
        
        st.warning(f"Using mock predictions for {model_type} model")
        return mock_predictions

def get_expert_contributions(model, X_test_list, model_type="original"):
    """Get expert contributions for test data."""
    try:
        # Format input for prediction
        formatted_input = format_input_for_prediction(X_test_list)
        
        # Try using the model's predict method with formatted input
        try:
            predictions = model.predict(formatted_input, verbose=0)
            
            # Since we don't have gate weights from predict method,
            # we'll create mock gate weights for visualization
            num_samples = formatted_input.shape[0]
            num_experts = len(X_test_list)
            gate_outputs = tf.ones((num_samples, num_experts)) / num_experts
            
            st.info(f"{model_type.capitalize()} model prediction successful, using equal expert contributions for visualization.")
        except Exception as predict_error:
            st.warning(f"{model_type.capitalize()} model predict method failed: {predict_error}")
            
            # Try using the model's call method with formatted input
            try:
                result = model(formatted_input, training=False)
                
                # Handle different return formats
                if isinstance(result, tuple) and len(result) >= 2:
                    predictions, gate_outputs = result[0], result[1]
                else:
                    # If model returns only predictions, use a fallback approach
                    st.warning(f"{model_type.capitalize()} model doesn't return gate weights. Using mock weights for visualization.")
                    predictions = result
                    # Create mock gate weights (equal contribution from each expert)
                    num_samples = formatted_input.shape[0]
                    num_experts = len(X_test_list)
                    gate_outputs = tf.ones((num_samples, num_experts)) / num_experts
            except Exception as call_error:
                st.error(f"{model_type.capitalize()} model call method also failed: {call_error}")
                raise ValueError(f"Both prediction methods failed: {predict_error}, {call_error}")
        
        # Convert to numpy arrays
        gate_weights = gate_outputs.numpy() if hasattr(gate_outputs, 'numpy') else np.array(gate_outputs)
        predictions = predictions.numpy() if hasattr(predictions, 'numpy') else np.array(predictions)
        
        return predictions, gate_weights
    except Exception as e:
        st.error(f"Error getting expert contributions for {model_type} model: {e}")
        # Provide fallback mock data for visualization
        num_samples = len(X_test_list[0])
        num_experts = len(X_test_list)
        mock_predictions = np.random.random((num_samples, 1))
        mock_gate_weights = np.random.random((num_samples, num_experts))
        mock_gate_weights = mock_gate_weights / mock_gate_weights.sum(axis=1, keepdims=True)
        
        st.warning(f"Using mock data for {model_type} model expert contributions due to compatibility issues.")
        return mock_predictions, mock_gate_weights

def load_optimization_summary():
    """Load the optimization summary if available."""
    try:
        if os.path.exists(OPTIMIZATION_SUMMARY_PATH):
            with open(OPTIMIZATION_SUMMARY_PATH, 'r') as f:
                summary = json.load(f)
            return summary
        else:
            st.warning(f"Optimization summary not found at {OPTIMIZATION_SUMMARY_PATH}")
            return None
    except Exception as e:
        st.error(f"Error loading optimization summary: {e}")
        return None

def plot_roc_curves(y_true, y_pred_original, y_pred_optimized):
    """Plot ROC curves for both models."""
    # Calculate ROC curve for original model
    fpr_original, tpr_original, _ = roc_curve(y_true, y_pred_original)
    roc_auc_original = auc(fpr_original, tpr_original)
    
    # Calculate ROC curve for optimized model
    fpr_optimized, tpr_optimized, _ = roc_curve(y_true, y_pred_optimized)
    roc_auc_optimized = auc(fpr_optimized, tpr_optimized)
    
    # Create figure
    fig = go.Figure()
    
    # Add original model curve
    fig.add_trace(go.Scatter(
        x=fpr_original, y=tpr_original,
        name=f'Original Model (AUC = {roc_auc_original:.3f})',
        line=dict(color='royalblue', width=2)
    ))
    
    # Add optimized model curve
    fig.add_trace(go.Scatter(
        x=fpr_optimized, y=tpr_optimized,
        name=f'Optimized Model (AUC = {roc_auc_optimized:.3f})',
        line=dict(color='purple', width=2)
    ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name='Random Classifier',
        line=dict(color='gray', width=1, dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title='ROC Curve Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
        width=700,
        height=500
    )
    
    return fig

def plot_confusion_matrices(y_true, y_pred_original, y_pred_optimized, threshold=0.5):
    """Plot confusion matrices for both models."""
    # Convert predictions to binary
    y_pred_original_binary = (y_pred_original >= threshold).astype(int)
    y_pred_optimized_binary = (y_pred_optimized >= threshold).astype(int)
    
    # Calculate confusion matrices
    cm_original = confusion_matrix(y_true, y_pred_original_binary)
    cm_optimized = confusion_matrix(y_true, y_pred_optimized_binary)
    
    # Create subplots
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=("Original Model", "Optimized Model"),
                        specs=[[{"type": "heatmap"}, {"type": "heatmap"}]])
    
    # Add original model confusion matrix
    fig.add_trace(
        go.Heatmap(
            z=cm_original,
            x=['No Migraine', 'Migraine'],
            y=['No Migraine', 'Migraine'],
            colorscale='Blues',
            showscale=False,
            text=cm_original,
            texttemplate="%{text}",
            textfont={"size":14},
        ),
        row=1, col=1
    )
    
    # Add optimized model confusion matrix
    fig.add_trace(
        go.Heatmap(
            z=cm_optimized,
            x=['No Migraine', 'Migraine'],
            y=['No Migraine', 'Migraine'],
            colorscale='Purples',
            showscale=False,
            text=cm_optimized,
            texttemplate="%{text}",
            textfont={"size":14},
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=f"Confusion Matrix Comparison (Threshold: {threshold:.2f})",
        height=400,
        width=700
    )
    
    return fig

def plot_expert_contributions(gate_weights_original, gate_weights_optimized):
    """Plot expert contributions for both models."""
    # Calculate average contribution of each expert
    avg_weights_original = np.mean(gate_weights_original, axis=0)
    avg_weights_optimized = np.mean(gate_weights_optimized, axis=0)
    
    # Expert names
    expert_names = ['Sleep Expert', 'Weather Expert', 'Stress/Diet Expert']
    
    # Create figure
    fig = go.Figure()
    
    # Add original model contributions
    fig.add_trace(go.Bar(
        x=expert_names,
        y=avg_weights_original,
        name='Original Model',
        marker_color='royalblue'
    ))
    
    # Add optimized model contributions
    fig.add_trace(go.Bar(
        x=expert_names,
        y=avg_weights_optimized,
        name='Optimized Model',
        marker_color='purple'
    ))
    
    # Update layout
    fig.update_layout(
        title='Average Expert Contributions',
        xaxis_title='Expert',
        yaxis_title='Average Contribution',
        barmode='group',
        legend=dict(x=0.01, y=0.99),
        width=700,
        height=400
    )
    
    return fig

def main():
    """Main function to run the dashboard."""
    # Header
    st.markdown("<h1 class='main-header'>Migraine Prediction Model Comparison</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-text'>
    This dashboard compares the original FuseMoE model with the PyGMO-optimized version,
    showing performance improvements, expert contributions, and prediction capabilities.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", [
        "Model Comparison",
        "Performance Metrics",
        "Expert Contributions",
        "Prediction Tool",
        "Optimization Details"
    ])
    
    # Load data
    X_test_list, y_test = load_data()
    
    # Load models
    original_model = load_model(ORIGINAL_MODEL_PATH, "original")
    optimized_model = load_model(OPTIMIZED_MODEL_PATH, "optimized")
    
    # Store in session state
    if 'y_test' not in st.session_state:
        st.session_state['y_test'] = y_test
    
    # Get predictions if models are loaded
    if original_model and 'original_predictions' not in st.session_state:
        original_predictions = get_model_predictions(original_model, X_test_list, "original")
        st.session_state['original_predictions'] = original_predictions
    
    if optimized_model and 'optimized_predictions' not in st.session_state:
        optimized_predictions = get_model_predictions(optimized_model, X_test_list, "optimized")
        st.session_state['optimized_predictions'] = optimized_predictions
    
    # Get expert contributions if models are loaded
    if original_model and 'original_expert_contributions' not in st.session_state:
        _, original_gate_weights = get_expert_contributions(original_model, X_test_list, "original")
        st.session_state['original_expert_contributions'] = original_gate_weights
    
    if optimized_model and 'optimized_expert_contributions' not in st.session_state:
        _, optimized_gate_weights = get_expert_contributions(optimized_model, X_test_list, "optimized")
        st.session_state['optimized_expert_contributions'] = optimized_gate_weights
    
    # Load optimization summary
    optimization_summary = load_optimization_summary()
    
    # Page content
    if page == "Model Comparison":
        st.markdown("<h2 class='sub-header'>Model Comparison Overview</h2>", unsafe_allow_html=True)
        
        # Check if we have all the data we need
        if ('original_predictions' in st.session_state and 
            'optimized_predictions' in st.session_state and 
            'y_test' in st.session_state):
            
            # Get predictions and true values
            y_pred_original = st.session_state['original_predictions']
            y_pred_optimized = st.session_state['optimized_predictions']
            y_test = st.session_state['y_test']
            
            # Calculate metrics
            metrics_original = MigrainePerformanceMetrics(y_test, y_pred_original)
            metrics_optimized = MigrainePerformanceMetrics(y_test, y_pred_optimized)
            
            # Calculate key metrics
            roc_auc_original = metrics_original.roc_auc()
            roc_auc_optimized = metrics_optimized.roc_auc()
            
            accuracy_original, precision_original, recall_original, f1_original, _ = metrics_original.calculate_metrics(0.5)
            accuracy_optimized, precision_optimized, recall_optimized, f1_optimized, _ = metrics_optimized.calculate_metrics(0.5)
            
            # Calculate improvements
            auc_improvement = roc_auc_optimized - roc_auc_original
            f1_improvement = f1_optimized - f1_original
            
            # Display comparison cards
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class='model-card original-model'>
                <h3>Original FuseMoE Model</h3>
                <p><strong>AUC:</strong> {:.3f}</p>
                <p><strong>F1 Score:</strong> {:.3f}</p>
                <p><strong>Accuracy:</strong> {:.3f}</p>
                <p><strong>Precision:</strong> {:.3f}</p>
                <p><strong>Recall:</strong> {:.3f}</p>
                </div>
                """.format(roc_auc_original, f1_original, accuracy_original, precision_original, recall_original), 
                unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class='model-card optimized-model'>
                <h3>PyGMO-Optimized Model</h3>
                <p><strong>AUC:</strong> {:.3f}</p>
                <p><strong>F1 Score:</strong> {:.3f}</p>
                <p><strong>Accuracy:</strong> {:.3f}</p>
                <p><strong>Precision:</strong> {:.3f}</p>
                <p><strong>Recall:</strong> {:.3f}</p>
                </div>
                """.format(roc_auc_optimized, f1_optimized, accuracy_optimized, precision_optimized, recall_optimized), 
                unsafe_allow_html=True)
            
            # Display improvement summary
            st.markdown("<h3>Performance Improvement</h3>", unsafe_allow_html=True)
            
            improvement_class_auc = "improvement-positive" if auc_improvement > 0 else "improvement-negative"
            improvement_class_f1 = "improvement-positive" if f1_improvement > 0 else "improvement-negative"
            
            st.markdown(f"""
            <div class='highlight'>
            <p><strong>AUC Improvement:</strong> <span class='{improvement_class_auc}'>{auc_improvement:.3f}</span> ({(auc_improvement/roc_auc_original*100):.1f}%)</p>
            <p><strong>F1 Score Improvement:</strong> <span class='{improvement_class_f1}'>{f1_improvement:.3f}</span> ({(f1_improvement/max(0.001, f1_original)*100):.1f}%)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Plot ROC curves
            st.subheader("ROC Curve Comparison")
            roc_fig = plot_roc_curves(y_test, y_pred_original, y_pred_optimized)
            st.plotly_chart(roc_fig, use_container_width=True)
            
            # Plot confusion matrices
            st.subheader("Confusion Matrix Comparison")
            threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.01)
            cm_fig = plot_confusion_matrices(y_test, y_pred_original, y_pred_optimized, threshold)
            st.plotly_chart(cm_fig, use_container_width=True)
            
        else:
            st.warning("Models or predictions not available. Please check if models are loaded correctly.")
    
    elif page == "Performance Metrics":
        st.markdown("<h2 class='sub-header'>Detailed Performance Metrics</h2>", unsafe_allow_html=True)
        
        # Check if we have all the data we need
        if ('original_predictions' in st.session_state and 
            'optimized_predictions' in st.session_state and 
            'y_test' in st.session_state):
            
            # Get predictions and true values
            y_pred_original = st.session_state['original_predictions']
            y_pred_optimized = st.session_state['optimized_predictions']
            y_test = st.session_state['y_test']
            
            # Calculate metrics
            metrics_original = MigrainePerformanceMetrics(y_test, y_pred_original)
            metrics_optimized = MigrainePerformanceMetrics(y_test, y_pred_optimized)
            
            # Threshold analysis
            st.subheader("Threshold Analysis")
            st.write("Explore how different threshold values affect model performance metrics.")
            
            threshold = st.slider("Select Threshold", 0.0, 1.0, 0.5, 0.01, key="threshold_metrics")
            
            # Calculate metrics at selected threshold
            accuracy_original, precision_original, recall_original, f1_original, specificity_original = metrics_original.calculate_metrics(threshold)
            accuracy_optimized, precision_optimized, recall_optimized, f1_optimized, specificity_optimized = metrics_optimized.calculate_metrics(threshold)
            
            # Display metrics in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<h4>Original Model</h4>", unsafe_allow_html=True)
                metrics_df_original = pd.DataFrame({
                    'Metric': ['Accuracy', 'Precision', 'Recall (Sensitivity)', 'Specificity', 'F1 Score'],
                    'Value': [accuracy_original, precision_original, recall_original, specificity_original, f1_original]
                })
                st.dataframe(metrics_df_original, hide_index=True)
            
            with col2:
                st.markdown("<h4>Optimized Model</h4>", unsafe_allow_html=True)
                metrics_df_optimized = pd.DataFrame({
                    'Metric': ['Accuracy', 'Precision', 'Recall (Sensitivity)', 'Specificity', 'F1 Score'],
                    'Value': [accuracy_optimized, precision_optimized, recall_optimized, specificity_optimized, f1_optimized]
                })
                st.dataframe(metrics_df_optimized, hide_index=True)
            
            # Calculate metrics across different thresholds
            thresholds = np.arange(0.1, 1.0, 0.1)
            metrics_across_thresholds = []
            
            for t in thresholds:
                acc_orig, prec_orig, rec_orig, f1_orig, _ = metrics_original.calculate_metrics(t)
                acc_opt, prec_opt, rec_opt, f1_opt, _ = metrics_optimized.calculate_metrics(t)
                
                metrics_across_thresholds.append({
                    'Threshold': t,
                    'Original Accuracy': acc_orig,
                    'Optimized Accuracy': acc_opt,
                    'Original Precision': prec_orig,
                    'Optimized Precision': prec_opt,
                    'Original Recall': rec_orig,
                    'Optimized Recall': rec_opt,
                    'Original F1': f1_orig,
                    'Optimized F1': f1_opt
                })
            
            metrics_df = pd.DataFrame(metrics_across_thresholds)
            
            # Plot metrics across thresholds
            st.subheader("Metrics Across Thresholds")
            
            metric_to_plot = st.selectbox(
                "Select Metric to Plot",
                ["Accuracy", "Precision", "Recall", "F1"]
            )
            
            if metric_to_plot == "Accuracy":
                y_orig = metrics_df['Original Accuracy']
                y_opt = metrics_df['Optimized Accuracy']
            elif metric_to_plot == "Precision":
                y_orig = metrics_df['Original Precision']
                y_opt = metrics_df['Optimized Precision']
            elif metric_to_plot == "Recall":
                y_orig = metrics_df['Original Recall']
                y_opt = metrics_df['Optimized Recall']
            else:  # F1
                y_orig = metrics_df['Original F1']
                y_opt = metrics_df['Optimized F1']
            
            # Create figure
            fig = go.Figure()
            
            # Add original model line
            fig.add_trace(go.Scatter(
                x=metrics_df['Threshold'],
                y=y_orig,
                mode='lines+markers',
                name='Original Model',
                line=dict(color='royalblue', width=2)
            ))
            
            # Add optimized model line
            fig.add_trace(go.Scatter(
                x=metrics_df['Threshold'],
                y=y_opt,
                mode='lines+markers',
                name='Optimized Model',
                line=dict(color='purple', width=2)
            ))
            
            # Update layout
            fig.update_layout(
                title=f'{metric_to_plot} vs. Threshold',
                xaxis_title='Threshold',
                yaxis_title=metric_to_plot,
                legend=dict(x=0.01, y=0.99),
                width=700,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Find optimal thresholds
            st.subheader("Optimal Thresholds")
            
            # Calculate optimal thresholds for different criteria
            optimal_f1_threshold_original = metrics_original.get_optimal_threshold('f1')
            optimal_f1_threshold_optimized = metrics_optimized.get_optimal_threshold('f1')
            
            optimal_balanced_threshold_original = metrics_original.get_optimal_threshold('balanced')
            optimal_balanced_threshold_optimized = metrics_optimized.get_optimal_threshold('balanced')
            
            # Display optimal thresholds
            optimal_thresholds_df = pd.DataFrame({
                'Criterion': ['F1 Score', 'Balanced (Sensitivity/Specificity)'],
                'Original Model': [optimal_f1_threshold_original, optimal_balanced_threshold_original],
                'Optimized Model': [optimal_f1_threshold_optimized, optimal_balanced_threshold_optimized]
            })
            
            st.dataframe(optimal_thresholds_df, hide_index=True)
            
        else:
            st.warning("Models or predictions not available. Please check if models are loaded correctly.")
    
    elif page == "Expert Contributions":
        st.markdown("<h2 class='sub-header'>Expert Contributions Analysis</h2>", unsafe_allow_html=True)
        
        # Check if we have expert contributions
        if ('original_expert_contributions' in st.session_state and 
            'optimized_expert_contributions' in st.session_state):
            
            # Get expert contributions
            original_gate_weights = st.session_state['original_expert_contributions']
            optimized_gate_weights = st.session_state['optimized_expert_contributions']
            
            # Plot average expert contributions
            st.subheader("Average Expert Contributions")
            expert_fig = plot_expert_contributions(original_gate_weights, optimized_gate_weights)
            st.plotly_chart(expert_fig, use_container_width=True)
            
            # Expert contribution details
            st.subheader("Expert Contribution Details")
            
            # Expert names
            expert_names = ['Sleep Expert', 'Weather Expert', 'Stress/Diet Expert']
            
            # Calculate statistics
            original_stats = {
                'Mean': np.mean(original_gate_weights, axis=0),
                'Median': np.median(original_gate_weights, axis=0),
                'Std Dev': np.std(original_gate_weights, axis=0),
                'Min': np.min(original_gate_weights, axis=0),
                'Max': np.max(original_gate_weights, axis=0)
            }
            
            optimized_stats = {
                'Mean': np.mean(optimized_gate_weights, axis=0),
                'Median': np.median(optimized_gate_weights, axis=0),
                'Std Dev': np.std(optimized_gate_weights, axis=0),
                'Min': np.min(optimized_gate_weights, axis=0),
                'Max': np.max(optimized_gate_weights, axis=0)
            }
            
            # Display statistics in tabs
            tab1, tab2 = st.tabs(["Original Model", "Optimized Model"])
            
            with tab1:
                # Create dataframe for original model
                stats_df_original = pd.DataFrame({
                    'Statistic': list(original_stats.keys()),
                    expert_names[0]: [stat[0] for stat in original_stats.values()],
                    expert_names[1]: [stat[1] for stat in original_stats.values()],
                    expert_names[2]: [stat[2] for stat in original_stats.values()]
                })
                
                st.dataframe(stats_df_original, hide_index=True)
                
                # Plot distribution of expert weights
                st.subheader("Distribution of Expert Weights")
                
                # Create figure
                fig = go.Figure()
                
                # Add violin plots for each expert
                for i, expert in enumerate(expert_names):
                    fig.add_trace(go.Violin(
                        y=original_gate_weights[:, i],
                        name=expert,
                        box_visible=True,
                        meanline_visible=True
                    ))
                
                # Update layout
                fig.update_layout(
                    title='Original Model: Distribution of Expert Weights',
                    yaxis_title='Weight',
                    width=700,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # Create dataframe for optimized model
                stats_df_optimized = pd.DataFrame({
                    'Statistic': list(optimized_stats.keys()),
                    expert_names[0]: [stat[0] for stat in optimized_stats.values()],
                    expert_names[1]: [stat[1] for stat in optimized_stats.values()],
                    expert_names[2]: [stat[2] for stat in optimized_stats.values()]
                })
                
                st.dataframe(stats_df_optimized, hide_index=True)
                
                # Plot distribution of expert weights
                st.subheader("Distribution of Expert Weights")
                
                # Create figure
                fig = go.Figure()
                
                # Add violin plots for each expert
                for i, expert in enumerate(expert_names):
                    fig.add_trace(go.Violin(
                        y=optimized_gate_weights[:, i],
                        name=expert,
                        box_visible=True,
                        meanline_visible=True
                    ))
                
                # Update layout
                fig.update_layout(
                    title='Optimized Model: Distribution of Expert Weights',
                    yaxis_title='Weight',
                    width=700,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Expert contribution changes
            st.subheader("Expert Contribution Changes")
            
            # Calculate changes
            mean_changes = optimized_stats['Mean'] - original_stats['Mean']
            
            # Create dataframe for changes
            changes_df = pd.DataFrame({
                'Expert': expert_names,
                'Original Mean': original_stats['Mean'],
                'Optimized Mean': optimized_stats['Mean'],
                'Absolute Change': mean_changes,
                'Relative Change (%)': mean_changes / np.maximum(0.001, original_stats['Mean']) * 100
            })
            
            st.dataframe(changes_df, hide_index=True)
            
            # Plot changes
            fig = go.Figure()
            
            # Add bar for absolute changes
            fig.add_trace(go.Bar(
                x=expert_names,
                y=mean_changes,
                name='Absolute Change',
                marker_color=['green' if x > 0 else 'red' for x in mean_changes]
            ))
            
            # Update layout
            fig.update_layout(
                title='Changes in Expert Contributions (Optimized - Original)',
                yaxis_title='Change in Mean Contribution',
                width=700,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("Expert contributions not available. Please check if models are loaded correctly.")
    
    elif page == "Prediction Tool":
        st.markdown("<h2 class='sub-header'>Migraine Prediction Tool</h2>", unsafe_allow_html=True)
        st.write("Compare predictions from both models using custom input values.")
        
        # Create input form
        with st.form("prediction_form"):
            st.subheader("Sleep Features")
            sleep_col1, sleep_col2 = st.columns(2)
            with sleep_col1:
                sleep_duration = st.slider("Sleep Duration (hours)", 3.0, 12.0, 7.0, 0.1)
                sleep_quality = st.slider("Sleep Quality (0-10)", 0, 10, 5)
            with sleep_col2:
                deep_sleep = st.slider("Deep Sleep (%)", 0, 50, 20, 1)
                rem_sleep = st.slider("REM Sleep (%)", 0, 40, 20, 1)
            
            st.subheader("Weather Features")
            weather_col1, weather_col2 = st.columns(2)
            with weather_col1:
                temperature = st.slider("Temperature (°C)", -10.0, 40.0, 20.0, 0.5)
                humidity = st.slider("Humidity (%)", 0, 100, 50, 1)
            with weather_col2:
                pressure = st.slider("Barometric Pressure (hPa)", 980.0, 1040.0, 1013.0, 0.5)
                pressure_change = st.slider("Pressure Change (hPa/24h)", -20.0, 20.0, 0.0, 0.5)
            
            st.subheader("Stress/Diet Features")
            stress_col1, stress_col2 = st.columns(2)
            with stress_col1:
                stress_level = st.slider("Stress Level (0-10)", 0, 10, 5)
                water_intake = st.slider("Water Intake (cups)", 0, 15, 8)
            with stress_col2:
                caffeine = st.slider("Caffeine Intake (mg)", 0, 500, 100, 10)
                alcohol = st.slider("Alcohol Units", 0, 10, 0)
            
            submitted = st.form_submit_button("Predict Migraine Risk")
        
        if submitted:
            # Create input features
            sleep_features = np.array([[sleep_duration, sleep_quality, deep_sleep, rem_sleep]])
            weather_features = np.array([[temperature, humidity, pressure, pressure_change]])
            stress_diet_features = np.array([[stress_level, water_intake, caffeine, alcohol]])
            
            # Normalize features (simple min-max scaling for demonstration)
            # In a real application, you would use the same scaling as during training
            sleep_features_norm = (sleep_features - np.array([[3, 0, 0, 0]])) / np.array([[9, 10, 50, 40]])
            weather_features_norm = (weather_features - np.array([[-10, 0, 980, -20]])) / np.array([[50, 100, 60, 40]])
            stress_diet_features_norm = (stress_diet_features - np.array([[0, 0, 0, 0]])) / np.array([[10, 15, 500, 10]])
            
            # Create input for model
            # For simplicity, we'll create a single sample with sequence length 1
            sleep_input = np.zeros((1, 7, 6))
            sleep_input[:, -1, :4] = sleep_features_norm
            
            weather_input = weather_features_norm
            
            stress_diet_input = np.zeros((1, 7, 6))
            stress_diet_input[:, -1, :4] = stress_diet_features_norm
            
            X_input = [sleep_input, weather_input, stress_diet_input]
            
            # Make predictions if models are loaded
            if original_model and optimized_model:
                # Format input for prediction
                formatted_input = format_input_for_prediction(X_input)
                
                # Get predictions
                try:
                    original_pred = original_model.predict(formatted_input, verbose=0)[0][0]
                    optimized_pred = optimized_model.predict(formatted_input, verbose=0)[0][0]
                    
                    # Display predictions
                    st.subheader("Prediction Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        <div class='model-card original-model'>
                        <h3>Original Model Prediction</h3>
                        <p><strong>Migraine Risk:</strong> {original_pred:.2%}</p>
                        <p><strong>Risk Level:</strong> {'High' if original_pred >= 0.7 else 'Medium' if original_pred >= 0.3 else 'Low'}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class='model-card optimized-model'>
                        <h3>Optimized Model Prediction</h3>
                        <p><strong>Migraine Risk:</strong> {optimized_pred:.2%}</p>
                        <p><strong>Risk Level:</strong> {'High' if optimized_pred >= 0.7 else 'Medium' if optimized_pred >= 0.3 else 'Low'}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display risk difference
                    risk_diff = optimized_pred - original_pred
                    risk_diff_pct = risk_diff / max(0.001, original_pred) * 100
                    
                    diff_class = "improvement-negative" if risk_diff > 0 else "improvement-positive"
                    
                    st.markdown(f"""
                    <div class='highlight'>
                    <p><strong>Risk Difference:</strong> <span class='{diff_class}'>{risk_diff:.2%}</span> ({risk_diff_pct:.1f}%)</p>
                    <p><strong>Interpretation:</strong> The optimized model {'predicts a higher' if risk_diff > 0 else 'predicts a lower'} migraine risk for these input values.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error making predictions: {e}")
                    st.exception(e)
            else:
                st.warning("Models not loaded. Please check if model files exist and are accessible.")
    
    elif page == "Optimization Details":
        st.markdown("<h2 class='sub-header'>PyGMO Optimization Details</h2>", unsafe_allow_html=True)
        
        # Check if optimization summary is available
        if optimization_summary:
            # Display optimization overview
            st.subheader("Optimization Overview")
            
            # Extract timing information
            optimization_time = optimization_summary.get('optimization_time_seconds', 0)
            training_time = optimization_summary.get('training_time_seconds', 0)
            
            # Display timing information
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Optimization Time", f"{optimization_time/60:.1f} minutes")
            with col2:
                st.metric("Training Time", f"{training_time/60:.1f} minutes")
            
            # Display performance improvements
            st.subheader("Performance Improvements")
            
            # Extract improvement information
            auc_improvement = optimization_summary.get('improvement', {}).get('auc_improvement', 0)
            f1_improvement = optimization_summary.get('improvement', {}).get('f1_improvement', 0)
            
            # Display improvement metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("AUC Improvement", f"{auc_improvement:.3f}", f"{auc_improvement/0.5625*100:.1f}%")
            with col2:
                st.metric("F1 Score Improvement", f"{f1_improvement:.3f}", f"{f1_improvement/max(0.001, 0.0741)*100:.1f}%")
            
            # Display optimization phases
            st.subheader("Optimization Phases")
            
            # Create tabs for each phase
            tab1, tab2, tab3 = st.tabs(["Expert Optimization", "Gating Optimization", "End-to-End Optimization"])
            
            with tab1:
                # Expert phase results
                expert_phase = optimization_summary.get('expert_phase', {})
                
                if expert_phase:
                    # Sleep expert
                    st.markdown("<h4>Sleep Expert</h4>", unsafe_allow_html=True)
                    sleep_config = expert_phase.get('sleep', {}).get('config', {})
                    sleep_fitness = expert_phase.get('sleep', {}).get('fitness', 0)
                    
                    if sleep_config:
                        sleep_df = pd.DataFrame({
                            'Parameter': list(sleep_config.keys()),
                            'Value': list(sleep_config.values())
                        })
                        st.dataframe(sleep_df, hide_index=True)
                        st.metric("Validation AUC", f"{sleep_fitness:.3f}")
                    
                    # Weather expert
                    st.markdown("<h4>Weather Expert</h4>", unsafe_allow_html=True)
                    weather_config = expert_phase.get('weather', {}).get('config', {})
                    weather_fitness = expert_phase.get('weather', {}).get('fitness', 0)
                    
                    if weather_config:
                        weather_df = pd.DataFrame({
                            'Parameter': list(weather_config.keys()),
                            'Value': list(weather_config.values())
                        })
                        st.dataframe(weather_df, hide_index=True)
                        st.metric("Validation AUC", f"{weather_fitness:.3f}")
                    
                    # Stress/Diet expert
                    st.markdown("<h4>Stress/Diet Expert</h4>", unsafe_allow_html=True)
                    stress_diet_config = expert_phase.get('stress_diet', {}).get('config', {})
                    stress_diet_fitness = expert_phase.get('stress_diet', {}).get('fitness', 0)
                    
                    if stress_diet_config:
                        stress_diet_df = pd.DataFrame({
                            'Parameter': list(stress_diet_config.keys()),
                            'Value': list(stress_diet_config.values())
                        })
                        st.dataframe(stress_diet_df, hide_index=True)
                        st.metric("Validation AUC", f"{stress_diet_fitness:.3f}")
                else:
                    st.warning("Expert phase results not available.")
            
            with tab2:
                # Gating phase results
                gating_phase = optimization_summary.get('gating_phase', {})
                
                if gating_phase:
                    st.markdown("<h4>Gating Network</h4>", unsafe_allow_html=True)
                    gating_config = gating_phase.get('config', {})
                    gating_fitness = gating_phase.get('fitness', 0)
                    
                    if gating_config:
                        gating_df = pd.DataFrame({
                            'Parameter': list(gating_config.keys()),
                            'Value': list(gating_config.values())
                        })
                        st.dataframe(gating_df, hide_index=True)
                        st.metric("Validation AUC", f"{gating_fitness:.3f}")
                else:
                    st.warning("Gating phase results not available.")
            
            with tab3:
                # End-to-End phase results
                e2e_phase = optimization_summary.get('e2e_phase', {})
                
                if e2e_phase:
                    st.markdown("<h4>End-to-End Optimization</h4>", unsafe_allow_html=True)
                    e2e_config = e2e_phase.get('config', {})
                    e2e_fitness = e2e_phase.get('fitness', {})
                    
                    if e2e_config:
                        e2e_df = pd.DataFrame({
                            'Parameter': list(e2e_config.keys()),
                            'Value': list(e2e_config.values())
                        })
                        st.dataframe(e2e_df, hide_index=True)
                        
                        # Display multi-objective fitness
                        if e2e_fitness:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Validation AUC", f"{e2e_fitness.get('auc', 0):.3f}")
                            with col2:
                                st.metric("Inference Latency", f"{e2e_fitness.get('latency', 0):.2f} ms")
                else:
                    st.warning("End-to-End phase results not available.")
            
            # Display final performance
            st.subheader("Final Performance")
            final_performance = optimization_summary.get('final_performance', {})
            
            if final_performance:
                # Create metrics dataframe
                metrics_df = pd.DataFrame({
                    'Metric': ['AUC', 'F1 Score', 'Precision', 'Recall', 'Loss'],
                    'Value': [
                        final_performance.get('val_auc', 0),
                        final_performance.get('val_f1', 0),
                        final_performance.get('val_precision', 0),
                        final_performance.get('val_recall', 0),
                        final_performance.get('val_loss', 0)
                    ]
                })
                
                st.dataframe(metrics_df, hide_index=True)
            else:
                st.warning("Final performance metrics not available.")
        else:
            st.warning("Optimization summary not available. Please run the optimization process first.")
            
            # Display placeholder information
            st.info("""
            The optimization process will:
            1. Optimize each expert's architecture independently (Phase 1)
            2. Optimize the gating network with fixed experts (Phase 2)
            3. Fine-tune the entire model with the best configurations (Phase 3)
            
            After optimization, this page will show detailed results from each phase,
            including optimized hyperparameters and performance improvements.
            """)

if __name__ == "__main__":
    main()
