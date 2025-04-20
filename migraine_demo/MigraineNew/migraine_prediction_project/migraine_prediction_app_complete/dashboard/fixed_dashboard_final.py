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

# Import our modules - use the fixed preprocessing module
from model.input_preprocessing_fixed import preprocess_expert_inputs, format_input_for_prediction
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
        # Load test data from the npz file with correct keys
        test_data_path = os.path.join(OUTPUT_DIR, 'data', 'test_data.npz')
        if os.path.exists(test_data_path):
            data = np.load(test_data_path)
            
            # Extract data with the correct keys
            X_test_sleep = data['X_sleep']
            X_test_weather = data['X_weather']
            X_test_stress_diet = data['X_stress_diet']
            X_test_physio = data['X_physio']
            y_test = data['y']
            
            st.success(f"Successfully loaded test data from {test_data_path}")
            return [X_test_sleep, X_test_weather, X_test_stress_diet, X_test_physio], y_test
        else:
            st.warning(f"Test data file not found at {test_data_path}")
            # Create mock data for testing
            X_test_sleep = np.random.random((20, 7, 6))
            X_test_weather = np.random.random((20, 4))
            X_test_stress_diet = np.random.random((20, 7, 6))
            X_test_physio = np.random.random((20, 5))
            y_test = np.random.randint(0, 2, (20, 1))
            
            st.warning("Using mock data for testing")
            return [X_test_sleep, X_test_weather, X_test_stress_diet, X_test_physio], y_test
    except Exception as e:
        st.error(f"Error loading test data: {e}")
        # Create mock data for testing
        X_test_sleep = np.random.random((20, 7, 6))
        X_test_weather = np.random.random((20, 4))
        X_test_stress_diet = np.random.random((20, 7, 6))
        X_test_physio = np.random.random((20, 5))
        y_test = np.random.randint(0, 2, (20, 1))
        
        st.warning("Using mock data due to error")
        return [X_test_sleep, X_test_weather, X_test_stress_diet, X_test_physio], y_test

def load_model(model_path, model_type="original"):
    """Load a saved model with custom objects."""
    try:
        # Create custom objects dictionary for model loading
        custom_objects = {
            'GatingNetwork': GatingNetwork,
            'FusionMechanism': FusionMechanism,
            'MigraineMoEModel': MigraineMoEModel,
            'SleepExpert': SleepExpert,
            'WeatherExpert': WeatherExpert,
            'StressDietExpert': StressDietExpert,
            'PhysioExpert': PhysioExpert
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
        # Format input for prediction - use only the first 3 arrays for original model
        if model_type == "original":
            formatted_input = format_input_for_prediction(X_test_list[:3])
        else:
            # For optimized model, use all arrays
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
        # Format input for prediction - use only the first 3 arrays for original model
        if model_type == "original":
            formatted_input = format_input_for_prediction(X_test_list[:3])
            num_experts = 3
        else:
            # For optimized model, use all arrays
            formatted_input = format_input_for_prediction(X_test_list)
            num_experts = 4
        
        # Try using the model's predict method with formatted input
        try:
            predictions = model.predict(formatted_input, verbose=0)
            
            # Since we don't have gate weights from predict method,
            # we'll create mock gate weights for visualization
            num_samples = formatted_input.shape[0]
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
        num_experts = 3 if model_type == "original" else 4
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
    expert_names_original = ['Sleep Expert', 'Weather Expert', 'Stress/Diet Expert']
    expert_names_optimized = ['Sleep Expert', 'Weather Expert', 'Stress/Diet Expert', 'Physio Expert']
    
    # Create figure
    fig = go.Figure()
    
    # Add original model contributions
    fig.add_trace(go.Bar(
        x=expert_names_original,
        y=avg_weights_original,
        name='Original Model',
        marker_color='royalblue'
    ))
    
    # Add optimized model contributions
    fig.add_trace(go.Bar(
        x=expert_names_optimized[:len(avg_weights_optimized)],
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
    page = st.sidebar.radio(
        "Select Page",
        ["Model Comparison", "Performance Metrics", "Expert Contributions", "Prediction Tool", "Optimization Details"]
    )
    
    # Load optimization summary
    optimization_summary = load_optimization_summary()
    
    # Load data
    X_test_list, y_test = load_data()
    
    # Load models
    original_model = load_model(ORIGINAL_MODEL_PATH, "original")
    optimized_model = load_model(OPTIMIZED_MODEL_PATH, "optimized")
    
    # Get predictions
    if original_model is not None:
        original_predictions = get_model_predictions(original_model, X_test_list, "original")
        st.session_state['original_predictions'] = original_predictions
    
    if optimized_model is not None:
        optimized_predictions = get_model_predictions(optimized_model, X_test_list, "optimized")
        st.session_state['optimized_predictions'] = optimized_predictions
    
    # Get expert contributions
    if original_model is not None:
        original_pred, original_gate_weights = get_expert_contributions(original_model, X_test_list, "original")
        st.session_state['original_expert_contributions'] = original_gate_weights
    
    if optimized_model is not None:
        optimized_pred, optimized_gate_weights = get_expert_contributions(optimized_model, X_test_list, "optimized")
        st.session_state['optimized_expert_contributions'] = optimized_gate_weights
    
    # Page content
    if page == "Model Comparison":
        st.markdown("<h2 class='sub-header'>Model Architecture Comparison</h2>", unsafe_allow_html=True)
        
        # Display model architecture
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='model-card original-model'>", unsafe_allow_html=True)
            st.markdown("<h3>Original FuseMoE Model</h3>", unsafe_allow_html=True)
            st.markdown("""
            <ul>
                <li>Mixture of Experts architecture</li>
                <li>3 expert models (Sleep, Weather, Stress/Diet)</li>
                <li>Simple gating network</li>
                <li>Basic fusion mechanism</li>
                <li>No optimization applied</li>
            </ul>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='model-card optimized-model'>", unsafe_allow_html=True)
            st.markdown("<h3>PyGMO-Optimized Model</h3>", unsafe_allow_html=True)
            st.markdown("""
            <ul>
                <li>Enhanced Mixture of Experts architecture</li>
                <li>4 expert models (Sleep, Weather, Stress/Diet, Physio)</li>
                <li>Optimized gating network with load balancing</li>
                <li>Advanced fusion mechanism</li>
                <li>PyGMO multi-objective optimization</li>
            </ul>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Display performance summary
        st.markdown("<h2 class='sub-header'>Performance Summary</h2>", unsafe_allow_html=True)
        
        if optimization_summary:
            # Extract performance metrics
            final_performance = optimization_summary.get('final_performance', {})
            improvement = optimization_summary.get('improvement', {})
            
            # Create metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "AUC", 
                    f"{final_performance.get('auc', 0):.4f}", 
                    f"+{improvement.get('auc_improvement', 0):.4f}"
                )
            
            with col2:
                st.metric(
                    "F1 Score", 
                    f"{final_performance.get('f1', 0):.4f}", 
                    f"+{improvement.get('f1_improvement', 0):.4f}"
                )
            
            with col3:
                st.metric(
                    "Precision", 
                    f"{final_performance.get('precision', 0):.4f}", 
                    f"+{improvement.get('precision_improvement', 0):.4f}"
                )
            
            with col4:
                st.metric(
                    "Recall", 
                    f"{final_performance.get('recall', 0):.4f}", 
                    f"+{improvement.get('recall_improvement', 0):.4f}"
                )
            
            # Display expert weights
            st.markdown("<h3>Expert Contributions</h3>", unsafe_allow_html=True)
            
            expert_weights = optimization_summary.get('optimization_phases', {}).get('gating_phase', {}).get('expert_weights', {})
            
            if expert_weights:
                # Create a pie chart of expert weights
                labels = list(expert_weights.keys())
                values = list(expert_weights.values())
                
                fig = px.pie(
                    names=labels,
                    values=values,
                    title="Expert Contribution Weights",
                    color_discrete_sequence=px.colors.sequential.Purples_r
                )
                
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(width=600, height=400)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Expert weights not available in optimization summary.")
        else:
            st.warning("Optimization summary not available. Please run the optimization process first.")
    
    elif page == "Performance Metrics":
        st.markdown("<h2 class='sub-header'>Detailed Performance Metrics</h2>", unsafe_allow_html=True)
        
        # Check if we have predictions
        if ('original_predictions' in st.session_state and 
            'optimized_predictions' in st.session_state):
            
            # Get predictions
            original_predictions = st.session_state['original_predictions']
            optimized_predictions = st.session_state['optimized_predictions']
            
            # Create metrics calculators
            metrics_original = MigrainePerformanceMetrics(y_test, original_predictions)
            metrics_optimized = MigrainePerformanceMetrics(y_test, optimized_predictions)
            
            # Calculate metrics with default threshold (0.5)
            accuracy_original, precision_original, recall_original, f1_original, specificity_original = metrics_original.calculate_metrics()
            accuracy_optimized, precision_optimized, recall_optimized, f1_optimized, specificity_optimized = metrics_optimized.calculate_metrics()
            
            # Display metrics
            st.subheader("Threshold Analysis")
            st.write("Explore how different threshold values affect model performance metrics.")
            
            # Threshold slider
            threshold = st.slider("Select Threshold", 0.0, 1.0, 0.5, 0.01)
            
            # Calculate metrics with selected threshold
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
            
            # Plot ROC curves
            st.subheader("ROC Curves")
            roc_fig = plot_roc_curves(y_test, original_predictions, optimized_predictions)
            st.plotly_chart(roc_fig, use_container_width=True)
            
            # Plot confusion matrices
            st.subheader("Confusion Matrices")
            cm_fig = plot_confusion_matrices(y_test, original_predictions, optimized_predictions, threshold)
            st.plotly_chart(cm_fig, use_container_width=True)
            
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
            expert_names_original = ['Sleep Expert', 'Weather Expert', 'Stress/Diet Expert']
            expert_names_optimized = ['Sleep Expert', 'Weather Expert', 'Stress/Diet Expert', 'Physio Expert']
            
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
                    expert_names_original[0]: [stat[0] for stat in original_stats.values()],
                    expert_names_original[1]: [stat[1] for stat in original_stats.values()],
                    expert_names_original[2]: [stat[2] for stat in original_stats.values()]
                })
                
                st.dataframe(stats_df_original, hide_index=True)
                
                # Plot distribution of expert weights
                st.subheader("Distribution of Expert Weights")
                
                # Create figure
                fig = go.Figure()
                
                # Add violin plots for each expert
                for i, expert in enumerate(expert_names_original):
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
                    expert_names_optimized[0]: [stat[0] for stat in optimized_stats.values()],
                    expert_names_optimized[1]: [stat[1] for stat in optimized_stats.values()],
                    expert_names_optimized[2]: [stat[2] for stat in optimized_stats.values()],
                    expert_names_optimized[3]: [stat[3] for stat in optimized_stats.values()] if optimized_gate_weights.shape[1] > 3 else [0, 0, 0, 0, 0]
                })
                
                st.dataframe(stats_df_optimized, hide_index=True)
                
                # Plot distribution of expert weights
                st.subheader("Distribution of Expert Weights")
                
                # Create figure
                fig = go.Figure()
                
                # Add violin plots for each expert
                for i, expert in enumerate(expert_names_optimized[:optimized_gate_weights.shape[1]]):
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
            
            # Display expert importance from optimization summary
            if optimization_summary:
                st.subheader("Expert Importance from Optimization")
                
                expert_weights = optimization_summary.get('optimization_phases', {}).get('gating_phase', {}).get('expert_weights', {})
                
                if expert_weights:
                    # Create a pie chart of expert weights
                    labels = list(expert_weights.keys())
                    values = list(expert_weights.values())
                    
                    fig = px.pie(
                        names=labels,
                        values=values,
                        title="Expert Contribution Weights from PyGMO Optimization",
                        color_discrete_sequence=px.colors.sequential.Purples_r
                    )
                    
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    fig.update_layout(width=600, height=400)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create a dataframe for expert weights
                    expert_weights_df = pd.DataFrame({
                        'Expert': labels,
                        'Weight': values
                    })
                    
                    st.dataframe(expert_weights_df, hide_index=True)
                else:
                    st.warning("Expert weights not available in optimization summary.")
            else:
                st.warning("Optimization summary not available. Please run the optimization process first.")
        else:
            st.warning("Expert contributions not available. Please check if models are loaded correctly.")
    
    elif page == "Prediction Tool":
        st.markdown("<h2 class='sub-header'>Migraine Prediction Tool</h2>", unsafe_allow_html=True)
        st.write("Use this tool to predict migraine likelihood based on input data.")
        
        # Create input form
        with st.form("prediction_form"):
            st.subheader("Sleep Data (Last 7 Days)")
            
            # Sleep quality (1-10)
            sleep_quality = []
            st.write("Sleep Quality (1-10)")
            col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
            with col1:
                sleep_quality.append(st.number_input("Day 1", 1, 10, 7))
            with col2:
                sleep_quality.append(st.number_input("Day 2", 1, 10, 6))
            with col3:
                sleep_quality.append(st.number_input("Day 3", 1, 10, 8))
            with col4:
                sleep_quality.append(st.number_input("Day 4", 1, 10, 7))
            with col5:
                sleep_quality.append(st.number_input("Day 5", 1, 10, 5))
            with col6:
                sleep_quality.append(st.number_input("Day 6", 1, 10, 6))
            with col7:
                sleep_quality.append(st.number_input("Day 7", 1, 10, 4))
            
            # Sleep duration (hours)
            sleep_duration = []
            st.write("Sleep Duration (hours)")
            col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
            with col1:
                sleep_duration.append(st.number_input("Day 1", 0.0, 12.0, 7.5, 0.5))
            with col2:
                sleep_duration.append(st.number_input("Day 2", 0.0, 12.0, 6.5, 0.5))
            with col3:
                sleep_duration.append(st.number_input("Day 3", 0.0, 12.0, 8.0, 0.5))
            with col4:
                sleep_duration.append(st.number_input("Day 4", 0.0, 12.0, 7.0, 0.5))
            with col5:
                sleep_duration.append(st.number_input("Day 5", 0.0, 12.0, 5.5, 0.5))
            with col6:
                sleep_duration.append(st.number_input("Day 6", 0.0, 12.0, 6.0, 0.5))
            with col7:
                sleep_duration.append(st.number_input("Day 7", 0.0, 12.0, 4.5, 0.5))
            
            st.subheader("Weather Data")
            col1, col2 = st.columns(2)
            with col1:
                temperature = st.slider("Temperature (°C)", -10.0, 40.0, 25.0, 0.5)
                humidity = st.slider("Humidity (%)", 0, 100, 65, 1)
            with col2:
                pressure = st.slider("Pressure (hPa)", 980.0, 1040.0, 1013.0, 0.5)
                pressure_change = st.slider("Pressure Change (hPa/day)", -20.0, 20.0, -5.0, 0.5)
            
            st.subheader("Stress/Diet Data (Last 7 Days)")
            
            # Stress level (1-10)
            stress_level = []
            st.write("Stress Level (1-10)")
            col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
            with col1:
                stress_level.append(st.number_input("Day 1 Stress", 1, 10, 3))
            with col2:
                stress_level.append(st.number_input("Day 2 Stress", 1, 10, 4))
            with col3:
                stress_level.append(st.number_input("Day 3 Stress", 1, 10, 6))
            with col4:
                stress_level.append(st.number_input("Day 4 Stress", 1, 10, 7))
            with col5:
                stress_level.append(st.number_input("Day 5 Stress", 1, 10, 8))
            with col6:
                stress_level.append(st.number_input("Day 6 Stress", 1, 10, 7))
            with col7:
                stress_level.append(st.number_input("Day 7 Stress", 1, 10, 9))
            
            # Diet quality (1-10)
            diet_quality = []
            st.write("Diet Quality (1-10)")
            col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
            with col1:
                diet_quality.append(st.number_input("Day 1 Diet", 1, 10, 8))
            with col2:
                diet_quality.append(st.number_input("Day 2 Diet", 1, 10, 7))
            with col3:
                diet_quality.append(st.number_input("Day 3 Diet", 1, 10, 6))
            with col4:
                diet_quality.append(st.number_input("Day 4 Diet", 1, 10, 5))
            with col5:
                diet_quality.append(st.number_input("Day 5 Diet", 1, 10, 4))
            with col6:
                diet_quality.append(st.number_input("Day 6 Diet", 1, 10, 5))
            with col7:
                diet_quality.append(st.number_input("Day 7 Diet", 1, 10, 3))
            
            st.subheader("Physiological Data")
            col1, col2 = st.columns(2)
            with col1:
                heart_rate = st.slider("Resting Heart Rate (bpm)", 40, 120, 75, 1)
                blood_pressure_sys = st.slider("Systolic Blood Pressure (mmHg)", 90, 200, 120, 1)
            with col2:
                blood_pressure_dia = st.slider("Diastolic Blood Pressure (mmHg)", 50, 120, 80, 1)
                body_temperature = st.slider("Body Temperature (°C)", 35.0, 40.0, 36.8, 0.1)
                hydration = st.slider("Hydration Level (1-10)", 1, 10, 6, 1)
            
            # Submit button
            submitted = st.form_submit_button("Predict Migraine Likelihood")
        
        # Process form submission
        if submitted:
            # Prepare input data
            sleep_data = np.zeros((1, 7, 6))
            for i in range(7):
                sleep_data[0, i, 0] = sleep_quality[i] / 10.0  # Normalize to 0-1
                sleep_data[0, i, 1] = sleep_duration[i] / 12.0  # Normalize to 0-1
                # Other sleep features would be calculated here
                sleep_data[0, i, 2] = 0.5  # Placeholder
                sleep_data[0, i, 3] = 0.5  # Placeholder
                sleep_data[0, i, 4] = 0.5  # Placeholder
                sleep_data[0, i, 5] = 0.5  # Placeholder
            
            weather_data = np.zeros((1, 4))
            weather_data[0, 0] = (temperature + 10) / 50.0  # Normalize to 0-1
            weather_data[0, 1] = humidity / 100.0  # Normalize to 0-1
            weather_data[0, 2] = (pressure - 980) / 60.0  # Normalize to 0-1
            weather_data[0, 3] = (pressure_change + 20) / 40.0  # Normalize to 0-1
            
            stress_diet_data = np.zeros((1, 7, 6))
            for i in range(7):
                stress_diet_data[0, i, 0] = stress_level[i] / 10.0  # Normalize to 0-1
                stress_diet_data[0, i, 1] = diet_quality[i] / 10.0  # Normalize to 0-1
                # Other stress/diet features would be calculated here
                stress_diet_data[0, i, 2] = 0.5  # Placeholder
                stress_diet_data[0, i, 3] = 0.5  # Placeholder
                stress_diet_data[0, i, 4] = 0.5  # Placeholder
                stress_diet_data[0, i, 5] = 0.5  # Placeholder
            
            physio_data = np.zeros((1, 5))
            physio_data[0, 0] = (heart_rate - 40) / 80.0  # Normalize to 0-1
            physio_data[0, 1] = (blood_pressure_sys - 90) / 110.0  # Normalize to 0-1
            physio_data[0, 2] = (blood_pressure_dia - 50) / 70.0  # Normalize to 0-1
            physio_data[0, 3] = (body_temperature - 35) / 5.0  # Normalize to 0-1
            physio_data[0, 4] = hydration / 10.0  # Normalize to 0-1
            
            # Make predictions
            input_data = [sleep_data, weather_data, stress_diet_data, physio_data]
            
            if 'original_predictions' in st.session_state and 'optimized_predictions' in st.session_state:
                # Use the models to make predictions
                if original_model is not None and optimized_model is not None:
                    # Format input for prediction - use only the first 3 arrays for original model
                    formatted_input_original = format_input_for_prediction(input_data[:3])
                    formatted_input_optimized = format_input_for_prediction(input_data)
                    
                    # Get predictions
                    original_pred = original_model.predict(formatted_input_original, verbose=0)
                    optimized_pred = optimized_model.predict(formatted_input_optimized, verbose=0)
                    
                    # Display predictions
                    st.subheader("Prediction Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("<div class='model-card original-model'>", unsafe_allow_html=True)
                        st.markdown("<h3>Original Model Prediction</h3>", unsafe_allow_html=True)
                        
                        # Get optimal threshold from optimization summary
                        optimal_threshold = 0.5
                        if optimization_summary:
                            optimal_threshold = optimization_summary.get('final_performance', {}).get('threshold', 0.5)
                        
                        # Calculate prediction
                        migraine_prob = float(original_pred[0][0])
                        migraine_pred = "Yes" if migraine_prob >= optimal_threshold else "No"
                        
                        # Display prediction
                        st.markdown(f"<h4>Migraine Probability: {migraine_prob:.2%}</h4>", unsafe_allow_html=True)
                        st.markdown(f"<h4>Prediction: {migraine_pred}</h4>", unsafe_allow_html=True)
                        
                        # Create gauge chart
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = migraine_prob * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Migraine Probability"},
                            gauge = {
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "royalblue"},
                                'steps': [
                                    {'range': [0, 25], 'color': "lightgreen"},
                                    {'range': [25, 50], 'color': "yellow"},
                                    {'range': [50, 75], 'color': "orange"},
                                    {'range': [75, 100], 'color': "red"}
                                ],
                                'threshold': {
                                    'line': {'color': "black", 'width': 4},
                                    'thickness': 0.75,
                                    'value': optimal_threshold * 100
                                }
                            }
                        ))
                        
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("<div class='model-card optimized-model'>", unsafe_allow_html=True)
                        st.markdown("<h3>Optimized Model Prediction</h3>", unsafe_allow_html=True)
                        
                        # Calculate prediction
                        migraine_prob = float(optimized_pred[0][0])
                        migraine_pred = "Yes" if migraine_prob >= optimal_threshold else "No"
                        
                        # Display prediction
                        st.markdown(f"<h4>Migraine Probability: {migraine_prob:.2%}</h4>", unsafe_allow_html=True)
                        st.markdown(f"<h4>Prediction: {migraine_pred}</h4>", unsafe_allow_html=True)
                        
                        # Create gauge chart
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = migraine_prob * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Migraine Probability"},
                            gauge = {
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "purple"},
                                'steps': [
                                    {'range': [0, 25], 'color': "lightgreen"},
                                    {'range': [25, 50], 'color': "yellow"},
                                    {'range': [50, 75], 'color': "orange"},
                                    {'range': [75, 100], 'color': "red"}
                                ],
                                'threshold': {
                                    'line': {'color': "black", 'width': 4},
                                    'thickness': 0.75,
                                    'value': optimal_threshold * 100
                                }
                            }
                        ))
                        
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Display expert contributions
                    st.subheader("Expert Contributions to Prediction")
                    
                    # Get expert contributions
                    _, original_gate_weights = get_expert_contributions(original_model, input_data, "original")
                    _, optimized_gate_weights = get_expert_contributions(optimized_model, input_data, "optimized")
                    
                    # Expert names
                    expert_names_original = ['Sleep Expert', 'Weather Expert', 'Stress/Diet Expert']
                    expert_names_optimized = ['Sleep Expert', 'Weather Expert', 'Stress/Diet Expert', 'Physio Expert']
                    
                    # Create figure
                    fig = go.Figure()
                    
                    # Add original model contributions
                    fig.add_trace(go.Bar(
                        x=expert_names_original,
                        y=original_gate_weights[0],
                        name='Original Model',
                        marker_color='royalblue'
                    ))
                    
                    # Add optimized model contributions
                    fig.add_trace(go.Bar(
                        x=expert_names_optimized[:optimized_gate_weights.shape[1]],
                        y=optimized_gate_weights[0],
                        name='Optimized Model',
                        marker_color='purple'
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title='Expert Contributions to Prediction',
                        xaxis_title='Expert',
                        yaxis_title='Contribution',
                        barmode='group',
                        legend=dict(x=0.01, y=0.99),
                        width=700,
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Models not available. Please check if models are loaded correctly.")
            else:
                st.error("Predictions not available. Please check if models are loaded correctly.")
    
    elif page == "Optimization Details":
        st.markdown("<h2 class='sub-header'>PyGMO Optimization Details</h2>", unsafe_allow_html=True)
        
        if optimization_summary:
            # Extract optimization details
            optimization_time = optimization_summary.get('optimization_time_seconds', 0) / 60  # Convert to minutes
            training_time = optimization_summary.get('training_time_seconds', 0) / 60  # Convert to minutes
            
            # Extract improvement metrics
            improvement = optimization_summary.get('improvement', {})
            auc_improvement = improvement.get('auc_improvement', 0)
            f1_improvement = improvement.get('f1_improvement', 0)
            
            # Display optimization overview
            st.subheader("Optimization Overview")
            
            # Display optimization time and training time
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Optimization Time", f"{optimization_time:.1f} minutes")
            with col2:
                st.metric("Training Time", f"{training_time:.1f} minutes")
            
            # Display performance improvements
            st.subheader("Performance Improvements")
            
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
                optimization_phases = optimization_summary.get('optimization_phases', {})
                expert_phase = optimization_phases.get('expert_phase', {})
                
                if expert_phase:
                    # Sleep expert
                    st.markdown("<h4>Sleep Expert</h4>", unsafe_allow_html=True)
                    sleep_expert = expert_phase.get('sleep', {})
                    sleep_config = sleep_expert.get('config', {})
                    sleep_fitness = sleep_expert.get('fitness', 0)
                    sleep_convergence = sleep_expert.get('convergence', {})
                    
                    if sleep_config:
                        sleep_df = pd.DataFrame({
                            'Parameter': list(sleep_config.keys()),
                            'Value': list(sleep_config.values())
                        })
                        st.dataframe(sleep_df, hide_index=True)
                        
                        # Display fitness and improvement
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Initial Fitness", f"{sleep_convergence.get('initial_fitness', 0):.3f}")
                        with col2:
                            st.metric("Final Fitness", f"{sleep_convergence.get('final_fitness', 0):.3f}")
                        with col3:
                            improvement = sleep_convergence.get('improvement', 0)
                            st.metric("Improvement", f"{improvement:.3f}", f"{improvement/max(0.001, sleep_convergence.get('initial_fitness', 0.001))*100:.1f}%")
                    
                    # Weather expert
                    st.markdown("<h4>Weather Expert</h4>", unsafe_allow_html=True)
                    weather_expert = expert_phase.get('weather', {})
                    weather_config = weather_expert.get('config', {})
                    weather_fitness = weather_expert.get('fitness', 0)
                    weather_convergence = weather_expert.get('convergence', {})
                    
                    if weather_config:
                        weather_df = pd.DataFrame({
                            'Parameter': list(weather_config.keys()),
                            'Value': list(weather_config.values())
                        })
                        st.dataframe(weather_df, hide_index=True)
                        
                        # Display fitness and improvement
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Initial Fitness", f"{weather_convergence.get('initial_fitness', 0):.3f}")
                        with col2:
                            st.metric("Final Fitness", f"{weather_convergence.get('final_fitness', 0):.3f}")
                        with col3:
                            improvement = weather_convergence.get('improvement', 0)
                            st.metric("Improvement", f"{improvement:.3f}", f"{improvement/max(0.001, weather_convergence.get('initial_fitness', 0.001))*100:.1f}%")
                    
                    # Stress/Diet expert
                    st.markdown("<h4>Stress/Diet Expert</h4>", unsafe_allow_html=True)
                    stress_diet_expert = expert_phase.get('stress_diet', {})
                    stress_diet_config = stress_diet_expert.get('config', {})
                    stress_diet_fitness = stress_diet_expert.get('fitness', 0)
                    stress_diet_convergence = stress_diet_expert.get('convergence', {})
                    
                    if stress_diet_config:
                        stress_diet_df = pd.DataFrame({
                            'Parameter': list(stress_diet_config.keys()),
                            'Value': list(stress_diet_config.values())
                        })
                        st.dataframe(stress_diet_df, hide_index=True)
                        
                        # Display fitness and improvement
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Initial Fitness", f"{stress_diet_convergence.get('initial_fitness', 0):.3f}")
                        with col2:
                            st.metric("Final Fitness", f"{stress_diet_convergence.get('final_fitness', 0):.3f}")
                        with col3:
                            improvement = stress_diet_convergence.get('improvement', 0)
                            st.metric("Improvement", f"{improvement:.3f}", f"{improvement/max(0.001, stress_diet_convergence.get('initial_fitness', 0.001))*100:.1f}%")
                    
                    # Physio expert
                    st.markdown("<h4>Physiological Expert</h4>", unsafe_allow_html=True)
                    physio_expert = expert_phase.get('physio', {})
                    physio_config = physio_expert.get('config', {})
                    physio_fitness = physio_expert.get('fitness', 0)
                    physio_convergence = physio_expert.get('convergence', {})
                    
                    if physio_config:
                        physio_df = pd.DataFrame({
                            'Parameter': list(physio_config.keys()),
                            'Value': list(physio_config.values())
                        })
                        st.dataframe(physio_df, hide_index=True)
                        
                        # Display fitness and improvement
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Initial Fitness", f"{physio_convergence.get('initial_fitness', 0):.3f}")
                        with col2:
                            st.metric("Final Fitness", f"{physio_convergence.get('final_fitness', 0):.3f}")
                        with col3:
                            improvement = physio_convergence.get('improvement', 0)
                            st.metric("Improvement", f"{improvement:.3f}", f"{improvement/max(0.001, physio_convergence.get('initial_fitness', 0.001))*100:.1f}%")
                else:
                    st.warning("Expert phase results not available.")
            
            with tab2:
                # Gating phase results
                gating_phase = optimization_phases.get('gating_phase', {})
                
                if gating_phase:
                    st.markdown("<h4>Gating Network</h4>", unsafe_allow_html=True)
                    gating_config = gating_phase.get('config', {})
                    gating_fitness = gating_phase.get('fitness', 0)
                    gating_convergence = gating_phase.get('convergence', {})
                    expert_weights = gating_phase.get('expert_weights', {})
                    
                    if gating_config:
                        gating_df = pd.DataFrame({
                            'Parameter': list(gating_config.keys()),
                            'Value': list(gating_config.values())
                        })
                        st.dataframe(gating_df, hide_index=True)
                        
                        # Display fitness and improvement
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Initial Fitness", f"{gating_convergence.get('initial_fitness', 0):.3f}")
                        with col2:
                            st.metric("Final Fitness", f"{gating_convergence.get('final_fitness', 0):.3f}")
                        with col3:
                            improvement = gating_convergence.get('improvement', 0)
                            st.metric("Improvement", f"{improvement:.3f}", f"{improvement/max(0.001, gating_convergence.get('initial_fitness', 0.001))*100:.1f}%")
                    
                    # Display expert weights
                    if expert_weights:
                        st.subheader("Expert Weights")
                        
                        # Create a pie chart of expert weights
                        labels = list(expert_weights.keys())
                        values = list(expert_weights.values())
                        
                        fig = px.pie(
                            names=labels,
                            values=values,
                            title="Expert Contribution Weights",
                            color_discrete_sequence=px.colors.sequential.Purples_r
                        )
                        
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        fig.update_layout(width=600, height=400)
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Create a dataframe for expert weights
                        expert_weights_df = pd.DataFrame({
                            'Expert': labels,
                            'Weight': values
                        })
                        
                        st.dataframe(expert_weights_df, hide_index=True)
                else:
                    st.warning("Gating phase results not available.")
            
            with tab3:
                # End-to-End phase results
                e2e_phase = optimization_phases.get('e2e_phase', {})
                
                if e2e_phase:
                    st.markdown("<h4>End-to-End Optimization</h4>", unsafe_allow_html=True)
                    e2e_config = e2e_phase.get('config', {})
                    e2e_fitness = e2e_phase.get('fitness', {})
                    e2e_convergence = e2e_phase.get('convergence', {})
                    pareto_front = e2e_phase.get('pareto_front', [])
                    
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
                                st.metric("AUC", f"{e2e_fitness.get('auc', 0):.3f}")
                            with col2:
                                st.metric("Latency", f"{e2e_fitness.get('latency', 0):.2f} ms")
                        
                        # Display convergence
                        if e2e_convergence:
                            st.subheader("Convergence")
                            
                            initial_fitness = e2e_convergence.get('initial_fitness', {})
                            final_fitness = e2e_convergence.get('final_fitness', {})
                            improvement = e2e_convergence.get('improvement', {})
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Initial AUC", f"{initial_fitness.get('auc', 0):.3f}")
                                st.metric("Initial Latency", f"{initial_fitness.get('latency', 0):.2f} ms")
                            with col2:
                                st.metric("Final AUC", f"{final_fitness.get('auc', 0):.3f}")
                                st.metric("Final Latency", f"{final_fitness.get('latency', 0):.2f} ms")
                            with col3:
                                auc_imp = improvement.get('auc', 0)
                                latency_imp = improvement.get('latency', 0)
                                st.metric("AUC Improvement", f"{auc_imp:.3f}", f"{auc_imp/max(0.001, initial_fitness.get('auc', 0.001))*100:.1f}%")
                                st.metric("Latency Improvement", f"{latency_imp:.2f} ms", f"{latency_imp/max(0.001, initial_fitness.get('latency', 0.001))*100:.1f}%")
                        
                        # Display Pareto front
                        if pareto_front:
                            st.subheader("Pareto Front")
                            
                            # Create dataframe for Pareto front
                            pareto_df = pd.DataFrame(pareto_front)
                            st.dataframe(pareto_df, hide_index=True)
                            
                            # Create scatter plot for Pareto front
                            fig = px.scatter(
                                pareto_df,
                                x='latency',
                                y='auc',
                                hover_data=['learning_rate', 'batch_size', 'l2_regularization'],
                                title="Pareto Front: AUC vs. Latency",
                                labels={'latency': 'Latency (ms)', 'auc': 'AUC'},
                                color_discrete_sequence=['purple']
                            )
                            
                            # Add line connecting Pareto points
                            fig.add_trace(go.Scatter(
                                x=pareto_df['latency'],
                                y=pareto_df['auc'],
                                mode='lines',
                                line=dict(color='rgba(128, 0, 128, 0.3)', width=2),
                                showlegend=False
                            ))
                            
                            # Update layout
                            fig.update_layout(
                                xaxis_title='Latency (ms)',
                                yaxis_title='AUC',
                                width=700,
                                height=500
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("End-to-End phase results not available.")
            
            # Display final performance metrics
            st.subheader("Final Performance Metrics")
            
            final_performance = optimization_summary.get('final_performance', {})
            
            if final_performance:
                # Create metrics dataframe
                metrics_df = pd.DataFrame({
                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC', 'Specificity', 'NPV', 'Threshold'],
                    'Value': [
                        final_performance.get('accuracy', 0),
                        final_performance.get('precision', 0),
                        final_performance.get('recall', 0),
                        final_performance.get('f1', 0),
                        final_performance.get('auc', 0),
                        final_performance.get('specificity', 0),
                        final_performance.get('npv', 0),
                        final_performance.get('threshold', 0)
                    ]
                })
                
                st.dataframe(metrics_df, hide_index=True)
                
                # Create radar chart for metrics
                fig = go.Figure()
                
                # Add radar chart
                fig.add_trace(go.Scatterpolar(
                    r=[
                        final_performance.get('accuracy', 0),
                        final_performance.get('precision', 0),
                        final_performance.get('recall', 0),
                        final_performance.get('f1', 0),
                        final_performance.get('auc', 0),
                        final_performance.get('specificity', 0),
                        final_performance.get('npv', 0)
                    ],
                    theta=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC', 'Specificity', 'NPV'],
                    fill='toself',
                    name='Optimized Model',
                    line_color='purple'
                ))
                
                # Update layout
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )
                    ),
                    title="Performance Metrics Radar Chart",
                    width=700,
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Final performance metrics not available.")
        else:
            st.warning("Optimization summary not available. Please run the optimization process first.")

if __name__ == "__main__":
    main()
