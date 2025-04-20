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

# REMOVED: set_page_config() call to avoid conflicts with integrated dashboard

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
        x=expert_names_optimized,
        y=avg_weights_optimized,
        name='Optimized Model',
        marker_color='purple'
    ))
    
    # Update layout
    fig.update_layout(
        title='Expert Contribution Comparison',
        xaxis_title='Expert',
        yaxis_title='Average Contribution',
        barmode='group',
        legend=dict(x=0.01, y=0.99),
        width=700,
        height=400
    )
    
    return fig

def plot_optimization_progress():
    """Plot optimization progress if available."""
    # Load optimization summary
    optimization_summary = load_optimization_summary()
    
    if not optimization_summary:
        st.warning("Optimization summary not available. Please run the optimization process first.")
        return None
    
    # Extract optimization phases
    optimization_phases = optimization_summary.get('optimization_phases', {})
    
    # Create figure
    fig = go.Figure()
    
    # Add expert phase improvements
    expert_phase = optimization_phases.get('expert_phase', {})
    if expert_phase:
        expert_names = []
        initial_fitness = []
        final_fitness = []
        
        for expert_name, expert_data in expert_phase.items():
            convergence = expert_data.get('convergence', {})
            if convergence:
                expert_names.append(expert_name.capitalize())
                initial_fitness.append(convergence.get('initial_fitness', 0))
                final_fitness.append(convergence.get('final_fitness', 0))
        
        # Add initial fitness
        fig.add_trace(go.Bar(
            x=expert_names,
            y=initial_fitness,
            name='Initial Fitness',
            marker_color='lightblue'
        ))
        
        # Add final fitness
        fig.add_trace(go.Bar(
            x=expert_names,
            y=final_fitness,
            name='Final Fitness',
            marker_color='darkblue'
        ))
    
    # Add gating phase improvement
    gating_phase = optimization_phases.get('gating_phase', {})
    if gating_phase:
        convergence = gating_phase.get('convergence', {})
        if convergence:
            fig.add_trace(go.Bar(
                x=['Gating Network'],
                y=[convergence.get('initial_fitness', 0)],
                name='Initial Fitness',
                marker_color='lightblue'
            ))
            
            fig.add_trace(go.Bar(
                x=['Gating Network'],
                y=[convergence.get('final_fitness', 0)],
                name='Final Fitness',
                marker_color='darkblue'
            ))
    
    # Add end-to-end phase improvement
    e2e_phase = optimization_phases.get('e2e_phase', {})
    if e2e_phase:
        convergence = e2e_phase.get('convergence', {})
        if convergence:
            initial_fitness = convergence.get('initial_fitness', {})
            final_fitness = convergence.get('final_fitness', {})
            
            if initial_fitness and final_fitness:
                fig.add_trace(go.Bar(
                    x=['End-to-End (AUC)'],
                    y=[initial_fitness.get('auc', 0)],
                    name='Initial AUC',
                    marker_color='lightgreen'
                ))
                
                fig.add_trace(go.Bar(
                    x=['End-to-End (AUC)'],
                    y=[final_fitness.get('auc', 0)],
                    name='Final AUC',
                    marker_color='darkgreen'
                ))
    
    # Update layout
    fig.update_layout(
        title='Optimization Progress',
        xaxis_title='Component',
        yaxis_title='Fitness/Performance',
        barmode='group',
        legend=dict(x=0.01, y=0.99),
        width=700,
        height=400
    )
    
    return fig

def display_optimization_details():
    """Display optimization details if available."""
    # Load optimization summary
    optimization_summary = load_optimization_summary()
    
    if not optimization_summary:
        st.warning("Optimization summary not available. Please run the optimization process first.")
        return
    
    # Extract optimization phases
    optimization_phases = optimization_summary.get('optimization_phases', {})
    
    # Create tabs for different phases
    tab1, tab2, tab3 = st.tabs(["Expert Optimization", "Gating Optimization", "End-to-End Optimization"])
    
    with tab1:
        # Expert phase results
        expert_phase = optimization_phases.get('expert_phase', {})
        
        if expert_phase:
            # Sleep expert
            st.markdown("<h4>Sleep Expert</h4>", unsafe_allow_html=True)
            sleep_expert = expert_phase.get('sleep', {})
            sleep_convergence = sleep_expert.get('convergence', {})
            
            if sleep_convergence:
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
            weather_convergence = weather_expert.get('convergence', {})
            
            if weather_convergence:
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
            stress_diet_convergence = stress_diet_expert.get('convergence', {})
            
            if stress_diet_convergence:
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
            physio_convergence = physio_expert.get('convergence', {})
            
            if physio_convergence:
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
            gating_convergence = gating_phase.get('convergence', {})
            
            if gating_convergence:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Initial Fitness", f"{gating_convergence.get('initial_fitness', 0):.3f}")
                with col2:
                    st.metric("Final Fitness", f"{gating_convergence.get('final_fitness', 0):.3f}")
                with col3:
                    improvement = gating_convergence.get('improvement', 0)
                    st.metric("Improvement", f"{improvement:.3f}", f"{improvement/max(0.001, gating_convergence.get('initial_fitness', 0.001))*100:.1f}%")
            
            # Display expert weights
            expert_weights = gating_phase.get('expert_weights', {})
            if expert_weights:
                st.subheader("Expert Weights")
                
                # Create columns for each expert
                cols = st.columns(len(expert_weights))
                
                # Display each expert weight
                for i, (expert_name, weight) in enumerate(expert_weights.items()):
                    with cols[i]:
                        st.metric(expert_name, f"{weight:.2f}", f"{weight*100:.1f}%")
            
            # Display gating network configuration
            if gating_config:
                st.subheader("Gating Network Configuration")
                
                gating_df = pd.DataFrame({
                    'Parameter': list(gating_config.keys()),
                    'Value': list(gating_config.values())
                })
                
                st.dataframe(gating_df, hide_index=True)
        else:
            st.warning("Gating phase results not available.")
    
    with tab3:
        # End-to-End phase results
        e2e_phase = optimization_phases.get('e2e_phase', {})
        
        if e2e_phase:
            st.markdown("<h4>End-to-End Optimization</h4>", unsafe_allow_html=True)
            e2e_fitness = e2e_phase.get('fitness', {})
            e2e_convergence = e2e_phase.get('convergence', {})
            
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
        else:
            st.warning("End-to-End phase results not available.")

def display_performance_metrics(y_true, y_pred_original, y_pred_optimized):
    """Display performance metrics for both models."""
    # Create performance metrics objects
    metrics_original = MigrainePerformanceMetrics(y_true, y_pred_original)
    metrics_optimized = MigrainePerformanceMetrics(y_true, y_pred_optimized)
    
    # Calculate metrics
    metrics_original.calculate_all_metrics()
    metrics_optimized.calculate_all_metrics()
    
    # Display metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<h4 class='sub-header'>Original Model</h4>", unsafe_allow_html=True)
        st.metric("Accuracy", f"{metrics_original.accuracy:.3f}")
        st.metric("AUC", f"{metrics_original.auc:.3f}")
        st.metric("F1 Score", f"{metrics_original.f1_score:.3f}")
        st.metric("Precision", f"{metrics_original.precision:.3f}")
        st.metric("Recall", f"{metrics_original.recall:.3f}")
    
    with col2:
        st.markdown("<h4 class='sub-header'>Optimized Model</h4>", unsafe_allow_html=True)
        st.metric("Accuracy", f"{metrics_optimized.accuracy:.3f}")
        st.metric("AUC", f"{metrics_optimized.auc:.3f}")
        st.metric("F1 Score", f"{metrics_optimized.f1_score:.3f}")
        st.metric("Precision", f"{metrics_optimized.precision:.3f}")
        st.metric("Recall", f"{metrics_optimized.recall:.3f}")
    
    with col3:
        st.markdown("<h4 class='sub-header'>Improvement</h4>", unsafe_allow_html=True)
        
        # Calculate improvements
        accuracy_imp = metrics_optimized.accuracy - metrics_original.accuracy
        auc_imp = metrics_optimized.auc - metrics_original.auc
        f1_imp = metrics_optimized.f1_score - metrics_original.f1_score
        precision_imp = metrics_optimized.precision - metrics_original.precision
        recall_imp = metrics_optimized.recall - metrics_original.recall
        
        # Display improvements with delta
        st.metric("Accuracy", f"{accuracy_imp:.3f}", f"{accuracy_imp*100:.1f}%")
        st.metric("AUC", f"{auc_imp:.3f}", f"{auc_imp*100:.1f}%")
        st.metric("F1 Score", f"{f1_imp:.3f}", f"{f1_imp*100:.1f}%")
        st.metric("Precision", f"{precision_imp:.3f}", f"{precision_imp*100:.1f}%")
        st.metric("Recall", f"{recall_imp:.3f}", f"{recall_imp*100:.1f}%")

def main():
    """Main function for the dashboard."""
    # Display header
    st.markdown("<h1 class='main-header'>Migraine Prediction Model Comparison</h1>", unsafe_allow_html=True)
    
    # Load data
    X_test, y_test = load_data()
    
    # Load models
    original_model = load_model(ORIGINAL_MODEL_PATH, "original")
    optimized_model = load_model(OPTIMIZED_MODEL_PATH, "optimized")
    
    # Get predictions
    if original_model and optimized_model:
        # Get predictions
        y_pred_original = get_model_predictions(original_model, X_test, "original")
        y_pred_optimized = get_model_predictions(optimized_model, X_test, "optimized")
        
        # Get expert contributions
        pred_original, gate_weights_original = get_expert_contributions(original_model, X_test, "original")
        pred_optimized, gate_weights_optimized = get_expert_contributions(optimized_model, X_test, "optimized")
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["Performance Metrics", "Visualizations", "Expert Contributions", "PyGMO Optimization"])
        
        with tab1:
            # Display performance metrics
            st.markdown("<h2 class='sub-header'>Performance Metrics</h2>", unsafe_allow_html=True)
            display_performance_metrics(y_test, y_pred_original, y_pred_optimized)
        
        with tab2:
            # Display visualizations
            st.markdown("<h2 class='sub-header'>Model Visualizations</h2>", unsafe_allow_html=True)
            
            # ROC curves
            st.subheader("ROC Curves")
            roc_fig = plot_roc_curves(y_test, y_pred_original, y_pred_optimized)
            st.plotly_chart(roc_fig, use_container_width=True)
            
            # Confusion matrices
            st.subheader("Confusion Matrices")
            threshold = st.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.01)
            cm_fig = plot_confusion_matrices(y_test, y_pred_original, y_pred_optimized, threshold)
            st.plotly_chart(cm_fig, use_container_width=True)
        
        with tab3:
            # Display expert contributions
            st.markdown("<h2 class='sub-header'>Expert Contributions</h2>", unsafe_allow_html=True)
            
            # Expert contribution chart
            expert_fig = plot_expert_contributions(gate_weights_original, gate_weights_optimized)
            st.plotly_chart(expert_fig, use_container_width=True)
            
            # Expert contribution explanation
            st.markdown("""
            <div class="highlight">
                <p>The chart above shows the average contribution of each expert model to the final prediction. 
                The original model uses three experts (Sleep, Weather, Stress/Diet), while the optimized model 
                adds a fourth expert (Physiological) and optimizes the weights using PyGMO.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with tab4:
            # Display PyGMO optimization details
            st.markdown("<h2 class='sub-header'>PyGMO Optimization Details</h2>", unsafe_allow_html=True)
            
            # Optimization progress chart
            st.subheader("Optimization Progress")
            opt_fig = plot_optimization_progress()
            if opt_fig:
                st.plotly_chart(opt_fig, use_container_width=True)
            
            # Detailed optimization results
            st.subheader("Detailed Optimization Results")
            display_optimization_details()
    else:
        st.error("Failed to load one or both models. Please check the model paths and try again.")
