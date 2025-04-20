import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import time
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

# Add the project root to the path
# Determine the absolute path of the directory containing this script
dashboard_dir = os.path.dirname(os.path.abspath(__file__))
# Determine the path of the parent directory (assumed project root relative to dashboard)
project_root = os.path.dirname(dashboard_dir)
# Add the project root directory to the Python path
sys.path.append(project_root)

# Import our modules (should now be found if they are in the project_root directory)
from model.optimized_model import OptimizedMigrainePredictionModel
# Use the dashboard-specific metrics implementation instead of the model version
from dashboard.dashboard_metrics import MigrainePerformanceMetrics

# Set page configuration
st.set_page_config(
    page_title="Migraine Prediction Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
</style>
""", unsafe_allow_html=True)

# Helper functions
def load_data(data_dir):
    """Load data from the specified directory."""
    combined_data = pd.read_csv(os.path.join(data_dir, 'combined_data.csv'))
    sleep_data = pd.read_csv(os.path.join(data_dir, 'sleep_data.csv'))
    weather_data = pd.read_csv(os.path.join(data_dir, 'weather_data.csv'))
    stress_diet_data = pd.read_csv(os.path.join(data_dir, 'stress_diet_data.csv'))
    
    return combined_data, sleep_data, weather_data, stress_diet_data

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

def get_expert_contributions(model, X_test_list):
    """Get expert contributions for test data."""
    try:
        # Import the input wrapper function
        from model.input_wrapper import format_input_for_prediction
        
        # Format the input to match the expected shape (None, 10)
        formatted_input = format_input_for_prediction(X_test_list)
        
        # Try using the model's predict method with formatted input
        try:
            predictions = model.predict(formatted_input, verbose=0)
            
            # Since we don't have gate weights from predict method,
            # we'll create mock gate weights for visualization
            num_samples = formatted_input.shape[0]
            num_experts = len(X_test_list)
            gate_outputs = tf.ones((num_samples, num_experts)) / num_experts
            
            st.info("Model prediction successful, using equal expert contributions for visualization.")
        except Exception as predict_error:
            st.warning(f"Model predict method failed: {predict_error}")
            
            # Try using the model's call method with formatted input
            try:
                result = model(formatted_input, training=False)
                
                # Handle different return formats
                if isinstance(result, tuple) and len(result) >= 2:
                    predictions, gate_outputs = result[0], result[1]
                else:
                    # If model returns only predictions, use a fallback approach
                    st.warning("Model doesn't return gate weights. Using mock weights for visualization.")
                    predictions = result
                    # Create mock gate weights (equal contribution from each expert)
                    num_samples = formatted_input.shape[0]
                    num_experts = len(X_test_list)
                    gate_outputs = tf.ones((num_samples, num_experts)) / num_experts
            except Exception as call_error:
                st.error(f"Model call method also failed: {call_error}")
                raise ValueError(f"Both prediction methods failed: {predict_error}, {call_error}")
        
        # Convert to numpy arrays
        gate_weights = gate_outputs.numpy() if hasattr(gate_outputs, 'numpy') else np.array(gate_outputs)
        predictions = predictions.numpy() if hasattr(predictions, 'numpy') else np.array(predictions)
        
        return predictions, gate_weights
    except Exception as e:
        st.error(f"Error getting expert contributions: {e}")
        # Provide fallback mock data for visualization
        num_samples = len(X_test_list[0])
        num_experts = len(X_test_list)
        mock_predictions = np.random.random((num_samples, 1))
        mock_gate_weights = np.random.random((num_samples, num_experts))
        mock_gate_weights = mock_gate_weights / mock_gate_weights.sum(axis=1, keepdims=True)
        
        st.warning("Using mock data for expert contributions due to model compatibility issues.")
        return mock_predictions, mock_gate_weights

def plot_confusion_matrix(y_true, y_pred, threshold=0.5):
    """Plot confusion matrix."""
    y_pred_binary = (y_pred >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary)
    
    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual"),
        x=['No Migraine', 'Migraine'],
        y=['No Migraine', 'Migraine'],
        color_continuous_scale='Blues'
    )
    fig.update_layout(
        title=f"Confusion Matrix (Threshold: {threshold:.2f})",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        width=500,
        height=500
    )
    
    return fig

def plot_roc_curve(y_true, y_pred):
    """Plot ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC = {roc_auc:.3f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=500,
        height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    fig.update_layout(
        xaxis_range=[0, 1],
        yaxis_range=[0, 1],
    )
    
    return fig, fpr, tpr, thresholds, roc_auc

def plot_precision_recall_curve(y_true, y_pred):
    """Plot precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    
    fig = px.area(
        x=recall, y=precision,
        title='Precision-Recall Curve',
        labels=dict(x='Recall', y='Precision'),
        width=500,
        height=500
    )
    fig.update_layout(
        xaxis_range=[0, 1],
        yaxis_range=[0, 1],
    )
    
    return fig, precision, recall, thresholds

def plot_threshold_analysis(fpr, tpr, thresholds, y_true, y_pred):
    """Plot threshold analysis."""
    # Calculate metrics at different thresholds
    thresholds_to_plot = np.linspace(0.1, 0.9, 9)
    sensitivity = []
    specificity = []
    f1_scores = []
    
    for threshold in thresholds_to_plot:
        y_pred_binary = (y_pred >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = sens
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        sensitivity.append(sens)
        specificity.append(spec)
        f1_scores.append(f1)
    
    # Create plot
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=thresholds_to_plot, y=sensitivity, name="Sensitivity", line=dict(color="blue")),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=thresholds_to_plot, y=specificity, name="Specificity", line=dict(color="red")),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=thresholds_to_plot, y=f1_scores, name="F1 Score", line=dict(color="green")),
        secondary_y=True,
    )
    
    fig.update_layout(
        title="Metrics vs. Threshold",
        xaxis_title="Threshold",
        width=700,
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_yaxes(title_text="Sensitivity/Specificity", secondary_y=False)
    fig.update_yaxes(title_text="F1 Score", secondary_y=True)
    
    return fig

def plot_expert_contributions(gate_weights, expert_names):
    """Plot expert contributions."""
    avg_weights = np.mean(gate_weights, axis=0)
    
    fig = px.bar(
        x=expert_names, 
        y=avg_weights,
        title="Average Expert Contribution",
        labels=dict(x="Expert", y="Average Weight"),
        color=expert_names,
        width=600,
        height=400
    )
    
    return fig

def plot_expert_contributions_by_outcome(gate_weights, y_true, expert_names):
    """Plot expert contributions by outcome."""
    migraine_indices = np.where(y_true == 1)[0]
    no_migraine_indices = np.where(y_true == 0)[0]
    
    migraine_weights = gate_weights[migraine_indices]
    no_migraine_weights = gate_weights[no_migraine_indices]
    
    avg_migraine_weights = np.mean(migraine_weights, axis=0) if len(migraine_weights) > 0 else np.zeros(len(expert_names))
    avg_no_migraine_weights = np.mean(no_migraine_weights, axis=0) if len(no_migraine_weights) > 0 else np.zeros(len(expert_names))
    
    df = pd.DataFrame({
        'Expert': expert_names * 2,
        'Weight': np.concatenate([avg_migraine_weights, avg_no_migraine_weights]),
        'Outcome': ['Migraine'] * len(expert_names) + ['No Migraine'] * len(expert_names)
    })
    
    fig = px.bar(
        df, 
        x='Expert', 
        y='Weight', 
        color='Outcome',
        barmode='group',
        title="Expert Contribution by Outcome",
        width=600,
        height=400
    )
    
    return fig

def plot_trigger_analysis(combined_data, predictions):
    """Plot trigger analysis."""
    # Define triggers
    pressure_drop = combined_data['pressure_change_24h'] <= -5
    sleep_disruption = (combined_data['total_sleep_hours'] < 5) | (combined_data['total_sleep_hours'] > 9)
    stress_spike = combined_data['stress_level'] >= 7
    dietary_trigger = (combined_data['alcohol_consumed'] > 0) | (combined_data['caffeine_consumed'] > 0) | (combined_data['chocolate_consumed'] > 0)
    
    # Calculate average prediction by trigger
    avg_pred_pressure = np.mean(predictions[pressure_drop])
    avg_pred_sleep = np.mean(predictions[sleep_disruption])
    avg_pred_stress = np.mean(predictions[stress_spike])
    avg_pred_dietary = np.mean(predictions[dietary_trigger])
    avg_pred_none = np.mean(predictions[~(pressure_drop | sleep_disruption | stress_spike | dietary_trigger)])
    
    # Create dataframe for plotting
    trigger_df = pd.DataFrame({
        'Trigger': ['Pressure Drop', 'Sleep Disruption', 'High Stress', 'Dietary Trigger', 'No Triggers'],
        'Average Prediction': [avg_pred_pressure, avg_pred_sleep, avg_pred_stress, avg_pred_dietary, avg_pred_none]
    })
    
    fig = px.bar(
        trigger_df,
        x='Trigger',
        y='Average Prediction',
        title="Average Prediction by Trigger",
        color='Average Prediction',
        color_continuous_scale='Viridis',
        width=700,
        height=400
    )
    
    return fig

def plot_patient_timeline(patient_data, predictions=None):
    """Plot patient timeline."""
    # Create figure
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Migraine Events", "Sleep Hours", "Barometric Pressure", "Stress Level")
    )
    
    # Add migraine events
    fig.add_trace(
        go.Scatter(
            x=patient_data['date'], 
            y=patient_data['next_day_migraine'],
            mode='markers',
            marker=dict(
                size=10,
                color=patient_data['next_day_migraine'],
                colorscale=[[0, 'rgba(0,0,255,0.3)'], [1, 'rgba(255,0,0,0.8)']],
                showscale=False
            ),
            name="Migraine"
        ),
        row=1, col=1
    )
    
    # Add predictions if available
    if predictions is not None:
        fig.add_trace(
            go.Scatter(
                x=patient_data['date'], 
                y=predictions,
                mode='lines',
                line=dict(color='purple', width=2),
                name="Prediction"
            ),
            row=1, col=1
        )
    
    # Add sleep hours
    fig.add_trace(
        go.Scatter(
            x=patient_data['date'], 
            y=patient_data['total_sleep_hours'],
            mode='lines+markers',
            line=dict(color='blue'),
            name="Sleep Hours"
        ),
        row=2, col=1
    )
    
    # Add reference lines for sleep
    fig.add_shape(
        type="line", line=dict(dash='dash', color='red'),
        x0=patient_data['date'].iloc[0], y0=5,
        x1=patient_data['date'].iloc[-1], y1=5,
        row=2, col=1
    )
    fig.add_shape(
        type="line", line=dict(dash='dash', color='red'),
        x0=patient_data['date'].iloc[0], y0=9,
        x1=patient_data['date'].iloc[-1], y1=9,
        row=2, col=1
    )
    
    # Add barometric pressure
    fig.add_trace(
        go.Scatter(
            x=patient_data['date'], 
            y=patient_data['barometric_pressure'],
            mode='lines+markers',
            line=dict(color='green'),
            name="Pressure"
        ),
        row=3, col=1
    )
    
    # Add pressure change
    fig.add_trace(
        go.Bar(
            x=patient_data['date'], 
            y=patient_data['pressure_change_24h'],
            marker_color=['red' if x <= -5 else 'lightgreen' for x in patient_data['pressure_change_24h']],
            name="Pressure Change"
        ),
        row=3, col=1
    )
    
    # Add stress level
    fig.add_trace(
        go.Scatter(
            x=patient_data['date'], 
            y=patient_data['stress_level'],
            mode='lines+markers',
            line=dict(color='orange'),
            name="Stress"
        ),
        row=4, col=1
    )
    
    # Add reference line for high stress
    fig.add_shape(
        type="line", line=dict(dash='dash', color='red'),
        x0=patient_data['date'].iloc[0], y0=7,
        x1=patient_data['date'].iloc[-1], y1=7,
        row=4, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        width=1000,
        title_text=f"Patient Timeline (ID: {patient_data['patient_id'].iloc[0]})",
        showlegend=False
    )
    
    # Update y-axes
    fig.update_yaxes(title_text="Migraine", row=1, col=1, range=[-0.1, 1.1])
    fig.update_yaxes(title_text="Sleep Hours", row=2, col=1)
    fig.update_yaxes(title_text="Pressure (hPa)", row=3, col=1)
    fig.update_yaxes(title_text="Stress (1-10)", row=4, col=1, range=[0, 10])
    
    return fig

def create_prediction_interface(model, combined_data):
    """Create an interface for making predictions."""
    st.markdown('<div class="sub-header">Make a Prediction</div>', unsafe_allow_html=True)
    
    # Create columns for input
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Sleep Data (Last 7 Days)")
        sleep_hours = []
        sleep_quality = []
        
        for i in range(7):
            day = 7 - i
            sleep_hours.append(st.slider(f"Sleep Hours (Day -{day})", 0.0, 12.0, 7.0, 0.5))
            sleep_quality.append(st.slider(f"Sleep Quality (Day -{day})", 1, 10, 7))
    
    with col2:
        st.subheader("Weather Data")
        temperature = st.slider("Temperature (°C)", -10.0, 40.0, 20.0, 0.5)
        humidity = st.slider("Humidity (%)", 0, 100, 50)
        barometric_pressure = st.slider("Barometric Pressure (hPa)", 980.0, 1040.0, 1013.0, 0.5)
        pressure_change_24h = st.slider("Pressure Change in 24h (hPa)", -15.0, 15.0, 0.0, 0.5)
    
    with col3:
        st.subheader("Stress/Diet Data (Last 7 Days)")
        stress_levels = []
        alcohol = []
        caffeine = []
        chocolate = []
        
        for i in range(7):
            day = 7 - i
            stress_levels.append(st.slider(f"Stress Level (Day -{day})", 1, 10, 5))
            
            if i == 0:  # Only ask for dietary info for the most recent day
                alcohol.append(st.checkbox(f"Alcohol Consumed (Day -{day})"))
                caffeine.append(st.checkbox(f"Caffeine Consumed (Day -{day})"))
                chocolate.append(st.checkbox(f"Chocolate Consumed (Day -{day})"))
            else:
                alcohol.append(False)
                caffeine.append(False)
                chocolate.append(False)
    
    # Create input data for model
    if st.button("Predict Migraine Risk"):
        with st.spinner("Calculating migraine risk..."):
            # Prepare input data
            sleep_data = np.array([sleep_hours + sleep_quality]).reshape(1, 7, 2)
            weather_data = np.array([[temperature, humidity, barometric_pressure, pressure_change_24h]])
            stress_diet_data = np.array([
                stress_levels + 
                [int(a) for a in alcohol] + 
                [int(c) for c in caffeine] + 
                [int(ch) for ch in chocolate]
            ]).reshape(1, 7, 3)
            
            # Normalize data (simple normalization for demo)
            sleep_data = (sleep_data - np.array([7, 5])) / np.array([2, 2])
            weather_data = (weather_data - np.array([20, 50, 1013, 0])) / np.array([10, 20, 10, 5])
            stress_diet_data = (stress_diet_data - np.array([5, 0, 0, 0])) / np.array([2, 1, 1, 1])
            
            # Make prediction
            try:
                prediction = model.predict([sleep_data, weather_data, stress_diet_data])
                risk = float(prediction[0][0])
                
                # Display result
                st.markdown(f"<h2 style='text-align: center;'>Migraine Risk: {risk:.1%}</h2>", unsafe_allow_html=True)
                
                # Create gauge chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = risk * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Migraine Risk"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "green"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 80
                        }
                    }
                ))
                
                fig.update_layout(width=400, height=300)
                st.plotly_chart(fig)
                
                # Identify triggers
                triggers = []
                if any(h < 5 or h > 9 for h in sleep_hours):
                    triggers.append("Sleep Disruption")
                if pressure_change_24h <= -5:
                    triggers.append("Barometric Pressure Drop")
                if any(s >= 7 for s in stress_levels):
                    triggers.append("High Stress")
                if any(alcohol) or any(caffeine) or any(chocolate):
                    triggers.append("Dietary Trigger")
                
                if triggers:
                    st.markdown("### Potential Triggers Identified:")
                    for trigger in triggers:
                        st.markdown(f"- {trigger}")
                else:
                    st.markdown("### No specific triggers identified")
                
                # Risk level explanation
                if risk < 0.3:
                    st.markdown('<div class="highlight">Low Risk: Continue with normal activities but maintain healthy habits.</div>', unsafe_allow_html=True)
                elif risk < 0.7:
                    st.markdown('<div class="highlight">Moderate Risk: Consider preventive measures and be prepared for a possible migraine.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="highlight">High Risk: Take preventive medication if prescribed and minimize exposure to triggers.</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")

# Main dashboard
def main():
    # Title
    st.markdown("<h1 class='main-header'>🧠 Migraine Prediction Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("Analyze model performance and explore prediction insights.")

    # --- Data Loading and Preparation ---
    with st.spinner('Loading data...'):
        try:
            # Define data and output directories using ABSOLUTE paths within the container
            DATA_DIR = '/home/ubuntu/migraine_demo/MigraineNew/migraine_prediction_project/migraine_prediction_app_complete/data'  # Absolute path inside the container
            OUTPUT_DIR = '/home/ubuntu/migraine_demo/MigraineNew/migraine_prediction_project/migraine_prediction_app_complete/output' # Absolute path inside the container
            MODEL_PATH = os.path.join(OUTPUT_DIR, 'optimized_model.keras')

            # Load data using the absolute DATA_DIR path
            combined_data, sleep_data, weather_data, stress_diet_data = load_data(DATA_DIR)
            st.session_state['combined_data'] = combined_data
            st.session_state['sleep_data'] = sleep_data
            st.session_state['weather_data'] = weather_data
            st.session_state['stress_diet_data'] = stress_diet_data
            st.success("Data loaded successfully!")
            # st.write(f"Loaded {len(combined_data)} records.") # Optional debug
            # st.dataframe(combined_data.head()) # Optional debug

        except FileNotFoundError as e:
            st.error(f"Error loading data or model: {e}")
            st.error(f"Looked in DATA_DIR: {DATA_DIR}") # Debug info
            st.info("Please ensure the data has been generated and the paths are correct.")
            st.info(f"Expected data file: {os.path.join(DATA_DIR, 'combined_data.csv')}")
            return # Stop execution if data loading fails
        except Exception as e:
            st.error(f"An unexpected error occurred during data loading: {e}")
            st.exception(e)
            return

    # --- Model Loading ---
    model = None
    if os.path.exists(MODEL_PATH):
        with st.spinner('Loading model...'):
            try:
                model = load_model(MODEL_PATH)
                if model:
                    st.session_state['model'] = model
                    st.success("Model loaded successfully!")
                else:
                    st.warning("Model loading returned None. Check model file integrity.")
            except Exception as e:
                st.error(f"An error occurred during model loading: {e}")
                st.exception(e)
    else:
        st.warning(f"Model file not found at {MODEL_PATH}. Please ensure the model has been trained and saved correctly.")
        st.info("Proceeding without model-specific features.")

    # --- Sidebar Navigation ---
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "Dashboard Overview",
        "Data Exploration",
        "Model Performance",
        "Expert Contributions",
        "Threshold Analysis",
        "Trigger Analysis",
        "Patient Timelines",
        "Live Prediction"
    ])

    # --- Page Content ---
    if page == "Dashboard Overview":
        st.markdown("<h2 class='sub-header'>Dashboard Overview</h2>", unsafe_allow_html=True)
        st.write("Welcome! This dashboard provides insights into the migraine prediction model.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Data Summary**")
            st.write(f"- Total Records: {len(combined_data)}")
            st.write(f"- Patients: {combined_data['patient_id'].nunique()}")
            st.write(f"- Start Date: {pd.to_datetime(combined_data['date']).min().date()}")
            st.write(f"- End Date: {pd.to_datetime(combined_data['date']).max().date()}")
            migraine_rate = combined_data['next_day_migraine'].mean() * 100
            st.write(f"- Overall Migraine Rate: {migraine_rate:.2f}%")
        with col2:
            st.markdown("**Model Status**")
        if model:
            st.success("Model Loaded")
            # You might want to display more model info here if available
            # st.write(f"Model Input Features: {model.input_shape}") # Example
        else:
            st.warning("Model Not Loaded")

    st.markdown("<div class='highlight'>Use the sidebar to navigate through different analysis sections.</div>", unsafe_allow_html=True)

    if page == "Data Exploration":
        st.markdown("<h2 class='sub-header'>Data Exploration</h2>", unsafe_allow_html=True)
        st.write("Explore the distributions and relationships within the synthetic data.")
        # Display sample data
        st.subheader("Combined Data Sample")
        st.dataframe(combined_data.head())

        # Migraine distribution
        st.subheader("Overall Migraine Distribution")
        fig_pie = px.pie(combined_data, names='next_day_migraine', title='Migraine vs. No Migraine (Next Day)',
                        color_discrete_map={0:'skyblue', 1:'lightcoral'},
                        labels={0:'No Migraine', 1:'Migraine'})
        st.plotly_chart(fig_pie, use_container_width=True)

        # Feature distributions
        st.subheader("Feature Distributions")
        feature_options = [col for col in combined_data.columns if col not in ['patient_id', 'date', 'next_day_migraine']]
        if feature_options: # Check if there are features to select
            feature = st.selectbox("Select a feature to plot:", feature_options)
            fig_hist = px.histogram(combined_data, x=feature, color='next_day_migraine',
                                    title=f'Distribution of {feature} by Migraine Outcome',
                                    barmode='overlay', opacity=0.7)
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.warning("No features available for distribution plot.")

    elif page == "Model Performance":
        st.markdown("<h2 class='sub-header'>Model Performance</h2>", unsafe_allow_html=True)
        if not model:
            st.warning("Model not loaded. Performance metrics cannot be displayed.")
            st.stop()

        st.write("Evaluating the performance of the optimized FuseMoE model on the test set.")

        try:
            test_predictions_path = os.path.join(OUTPUT_DIR, 'test_predictions.npz')
            if os.path.exists(test_predictions_path):
                test_data = np.load(test_predictions_path, allow_pickle=True)
                # Use the actual keys in the file ('y_true' and 'y_pred')
                y_test = test_data['y_true']
                y_pred_test = test_data['y_pred']

                st.session_state['y_test'] = y_test
                st.session_state['y_pred_test'] = y_pred_test

                st.success(f"Loaded test predictions for {len(y_test)} samples.")

                metrics = MigrainePerformanceMetrics(y_test, y_pred_test)
                threshold = st.slider("Select Prediction Threshold", 0.0, 1.0, 0.5, 0.01)
                accuracy, precision, recall, f1, specificity = metrics.calculate_metrics(threshold)
                roc_auc_score = metrics.roc_auc()

                st.markdown("**Key Performance Metrics**")
                cols = st.columns(4)
                with cols[0]:
                    st.metric("AUC", f"{roc_auc_score:.3f}")
                with cols[1]:
                    st.metric(f"Accuracy (Thr={threshold:.2f})", f"{accuracy:.3f}")
                with cols[2]:
                    st.metric(f"Precision (Thr={threshold:.2f})", f"{precision:.3f}")
                with cols[3]:
                    st.metric(f"Recall (Thr={threshold:.2f})", f"{recall:.3f}")
                # Display F1 and Specificity in a new row if needed
                cols2 = st.columns(4)
                with cols2[0]:
                    st.metric(f"F1 Score (Thr={threshold:.2f})", f"{f1:.3f}")
                with cols2[1]:
                    st.metric(f"Specificity (Thr={threshold:.2f})", f"{specificity:.3f}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Confusion Matrix")
                    fig_cm = plot_confusion_matrix(y_test, y_pred_test, threshold)
                    st.plotly_chart(fig_cm, use_container_width=True)
                with col2:
                    st.subheader("ROC Curve")
                    fig_roc, _, _, _, _ = plot_roc_curve(y_test, y_pred_test)
                    st.plotly_chart(fig_roc, use_container_width=True)

                st.subheader("Precision-Recall Curve")
                fig_pr, _, _, _ = plot_precision_recall_curve(y_test, y_pred_test)
                st.plotly_chart(fig_pr, use_container_width=True)

            else:
                st.warning(f"Test predictions file not found at {test_predictions_path}. Cannot display performance metrics.")
                st.info("Please ensure the model training script saves 'test_predictions.npz' to the output directory.")

        except Exception as e:
            st.error(f"Error loading or processing test predictions: {e}")
            st.exception(e)

    elif page == "Expert Contributions":
        st.markdown("<h2 class='sub-header'>Expert Contributions</h2>", unsafe_allow_html=True)
        if not model or 'y_pred_test' not in st.session_state:
            st.warning("Model or test predictions not loaded. Expert contribution analysis cannot be performed.")
            st.stop()

        st.write("Analyzing the contribution (gating weights) of each expert model.")

        try:
            if 'X_test_list' not in st.session_state:
                test_predictions_path = os.path.join(OUTPUT_DIR, 'test_predictions.npz')
                if os.path.exists(test_predictions_path):
                    test_data = np.load(test_predictions_path, allow_pickle=True)
                    
                    # Check if we have individual expert data arrays
                    if 'X_test_sleep' in test_data and 'X_test_weather' in test_data and 'X_test_stress_diet' in test_data:
                        # Combine individual expert arrays into X_test_list
                        X_test_sleep = test_data['X_test_sleep']
                        X_test_weather = test_data['X_test_weather']
                        X_test_stress_diet = test_data['X_test_stress_diet']
                        
                        # Create X_test_list from individual arrays
                        X_test_list = [X_test_sleep, X_test_weather, X_test_stress_diet]
                        st.session_state['X_test_list'] = X_test_list
                        st.success("Successfully loaded expert data from test predictions file")
                    elif 'X_test_list' in test_data:
                        st.session_state['X_test_list'] = test_data['X_test_list']
                    else:
                        st.error("Required expert data not found in test_predictions.npz")
                        st.stop()
            
            # Now X_test_list should be in session state if loading was successful
            X_test_list = st.session_state['X_test_list']
            y_test = st.session_state['y_test'] # Assumes y_test was loaded previously

            with st.spinner("Calculating expert contributions..."):
                _, gate_weights = get_expert_contributions(model, X_test_list)
                st.session_state['gate_weights'] = gate_weights

            expert_names = ['Sleep Expert', 'Weather Expert', 'Stress/Diet Expert']

            st.subheader("Average Expert Contribution")
            fig_avg_expert = plot_expert_contributions(gate_weights, expert_names)
            st.plotly_chart(fig_avg_expert, use_container_width=True)

            st.subheader("Expert Contribution by Outcome")
            fig_expert_outcome = plot_expert_contributions_by_outcome(gate_weights, y_test, expert_names)
            st.plotly_chart(fig_expert_outcome, use_container_width=True)

        except KeyError as e:
            st.error(f"Missing required data in session state: {e}. Ensure all necessary data (y_test, X_test_list) was loaded.")
        except Exception as e:
            st.error(f"Error during expert contribution analysis: {e}")
            st.exception(e)

    elif page == "Threshold Analysis":
        st.markdown("<h2 class='sub-header'>Threshold Analysis</h2>", unsafe_allow_html=True)
        if 'y_test' not in st.session_state or 'y_pred_test' not in st.session_state:
            st.warning("Test predictions not loaded. Threshold analysis cannot be performed.")
            st.stop()

        st.write("Analyzing how different prediction thresholds affect performance metrics.")

        y_test = st.session_state['y_test']
        y_pred_test = st.session_state['y_pred_test']

        try:
            _, fpr, tpr, thresholds_roc, _ = plot_roc_curve(y_test, y_pred_test) # Recalculate or get from state
            fig_threshold = plot_threshold_analysis(fpr, tpr, thresholds_roc, y_test, y_pred_test)
            st.plotly_chart(fig_threshold, use_container_width=True)

            st.markdown("**Interpretation:**")
            st.write(" - **Sensitivity (Recall):** True Positive Rate (correctly identified migraines)")
            st.write(" - **Specificity:** True Negative Rate (correctly identified non-migraines)")
            st.write(" - **F1 Score:** Harmonic mean of Precision and Recall, good for imbalanced datasets.")
            st.write("Adjusting the threshold changes the trade-off between these metrics. A lower threshold increases sensitivity but decreases specificity, and vice-versa.")

        except Exception as e:
            st.error(f"Error during threshold analysis: {e}")
            st.exception(e)

    elif page == "Trigger Analysis":
        st.markdown("<h2 class='sub-header'>Trigger Analysis (Conceptual)</h2>", unsafe_allow_html=True)
        if 'y_pred_test' not in st.session_state:
             st.warning("Test predictions not loaded. Trigger analysis cannot be performed.")
             st.stop()

        st.write("Exploring potential triggers by analyzing features on days preceding predicted migraines.")
        st.info("Note: This is a simplified analysis based on correlations, not causal inference.")

        try:
            # Load test data for trigger analysis
            if 'X_test_sleep' not in st.session_state or 'X_test_weather' not in st.session_state or 'X_test_stress_diet' not in st.session_state:
                test_predictions_path = os.path.join(OUTPUT_DIR, 'test_predictions.npz')
                if os.path.exists(test_predictions_path):
                    test_data = np.load(test_predictions_path, allow_pickle=True)
                    
                    # Check if we have individual expert data arrays
                    if 'X_test_sleep' in test_data and 'X_test_weather' in test_data and 'X_test_stress_diet' in test_data:
                        st.session_state['X_test_sleep'] = test_data['X_test_sleep']
                        st.session_state['X_test_weather'] = test_data['X_test_weather']
                        st.session_state['X_test_stress_diet'] = test_data['X_test_stress_diet']
                        st.success("Successfully loaded expert data for trigger analysis")
                    else:
                        st.error("Required expert data not found in test_predictions.npz")
                        st.stop()
            
            # Get predictions and true values
            y_pred_test = st.session_state['y_pred_test']
            y_test = st.session_state['y_true'] if 'y_true' in st.session_state else st.session_state['y_test']
            
            # Create threshold for predicted migraines
            threshold = 0.5
            predicted_migraine = (y_pred_test >= threshold).flatten()
            
            # Extract features from test data
            sleep_data = st.session_state['X_test_sleep']
            weather_data = st.session_state['X_test_weather']
            stress_diet_data = st.session_state['X_test_stress_diet']
            
            # Create dataframes for analysis
            sleep_df = pd.DataFrame(sleep_data[:, -1, :], columns=[f'sleep_feature_{i}' for i in range(sleep_data.shape[2])])
            weather_df = pd.DataFrame(weather_data, columns=[f'weather_feature_{i}' for i in range(weather_data.shape[1])])
            stress_diet_df = pd.DataFrame(stress_diet_data[:, -1, :], columns=[f'stress_diet_feature_{i}' for i in range(stress_diet_data.shape[2])])
            
            # Combine into a single dataframe
            features_df = pd.concat([sleep_df, weather_df, stress_diet_df], axis=1)
            features_df['predicted_migraine'] = predicted_migraine
            features_df['actual_migraine'] = y_test
            
            # Select a feature for analysis
            feature_options = features_df.columns[:-2]  # Exclude the migraine columns
            selected_feature = st.selectbox("Select a feature to analyze:", feature_options, index=0)
            
            # Create plots
            st.subheader(f"Distribution of {selected_feature} by Predicted Migraine")
            fig_pred = px.histogram(features_df, x=selected_feature, color='predicted_migraine',
                                 title=f'Distribution of {selected_feature} by Predicted Migraine',
                                 color_discrete_map={0: 'skyblue', 1: 'lightcoral'},
                                 barmode='overlay', opacity=0.7,
                                 labels={'predicted_migraine': 'Predicted Migraine', '0': 'No', '1': 'Yes'})
            st.plotly_chart(fig_pred, use_container_width=True)
            
            st.subheader(f"Distribution of {selected_feature} by Actual Migraine")
            fig_actual = px.histogram(features_df, x=selected_feature, color='actual_migraine',
                                   title=f'Distribution of {selected_feature} by Actual Migraine',
                                   color_discrete_map={0: 'skyblue', 1: 'lightcoral'},
                                   barmode='overlay', opacity=0.7,
                                   labels={'actual_migraine': 'Actual Migraine', '0': 'No', '1': 'Yes'})
            st.plotly_chart(fig_actual, use_container_width=True)
            
            # Feature correlation with migraine
            st.subheader("Feature Correlation with Migraine")
            corr_pred = features_df.drop('actual_migraine', axis=1).corr()['predicted_migraine'].sort_values(ascending=False)
            corr_actual = features_df.drop('predicted_migraine', axis=1).corr()['actual_migraine'].sort_values(ascending=False)
            
            corr_df = pd.DataFrame({
                'Correlation with Predicted': corr_pred.drop('predicted_migraine'),
                'Correlation with Actual': corr_actual.drop('actual_migraine')
            })
            
            fig_corr = px.bar(corr_df, barmode='group',
                           title='Feature Correlation with Migraine Occurrence',
                           labels={'value': 'Correlation Coefficient', 'variable': 'Migraine Type'})
            st.plotly_chart(fig_corr, use_container_width=True)
        except Exception as e:
            st.error(f"Error during trigger analysis: {e}")
            st.exception(e)

    elif page == "Patient Timelines":
        st.markdown("<h2 class='sub-header'>Patient Timelines</h2>", unsafe_allow_html=True)
        st.write("Visualize individual patient data and model predictions over time.")

        patient_ids = combined_data['patient_id'].unique()
        selected_patient = st.selectbox("Select Patient ID", patient_ids)

        patient_data = combined_data[combined_data['patient_id'] == selected_patient].copy()
        patient_data['date'] = pd.to_datetime(patient_data['date'])
        patient_data = patient_data.sort_values('date')

        # Check if predictions are available for this patient (requires careful mapping)
        # For now, just plot the actual data
        st.info("Displaying actual patient data. Prediction overlay requires mapping test set predictions back to individual patients.")

        try:
            fig_timeline = plot_patient_timeline(patient_data)
            st.plotly_chart(fig_timeline, use_container_width=True)
        except Exception as e:
            st.error(f"Error plotting patient timeline: {e}")
            st.exception(e)

    elif page == "Live Prediction":
        st.markdown("<h2 class='sub-header'>Live Prediction Interface</h2>", unsafe_allow_html=True)
        if not model:
            st.warning("Model not loaded. Live prediction is not available.")
            st.stop()

        st.write("Input current data points to get a next-day migraine prediction.")
        st.info("This uses the loaded model. Ensure the input features match the model's training data.")

        create_prediction_interface(model, combined_data) # Pass combined_data for default/range values

# --- Utility Functions for Prediction Interface (Example) ---

def prepare_data_for_prediction(data, sleep_data, weather_data, stress_diet_data):
    # This needs to replicate the exact preprocessing used during training
    # Including feature scaling, handling categorical features, etc.
    # Placeholder: Assumes data is already in the correct numerical format
    # Needs the feature names used during training

    # Example: Creating lists of DataFrames for each expert
    # This depends heavily on your model's input structure
    # You need to extract the correct features for each expert based on the input 'data' dictionary

    st.warning("Data preparation logic for prediction needs to be implemented based on training preprocessing.")

    # Placeholder - assumes direct mapping and correct feature names
    # Replace with actual feature names and structure expected by your model
    sleep_features = ['total_sleep_hours', 'sleep_quality']
    weather_features = ['temperature', 'humidity', 'barometric_pressure', 'pressure_change_24h']
    stress_diet_features = ['stress_level', 'alcohol_consumed', 'caffeine_consumed', 'chocolate_consumed']

    try:
        sleep_input = pd.DataFrame([data])[sleep_features]
        weather_input = pd.DataFrame([data])[weather_features]
        stress_diet_input = pd.DataFrame([data])[stress_diet_features]

        # Scale features using the SAME scaler used in training
        # scaler_sleep = load_scaler('sleep_scaler.pkl') # Example
        # sleep_input_scaled = scaler_sleep.transform(sleep_input)

        # Return list matching model input structure
        # return [sleep_input_scaled, weather_input_scaled, stress_diet_input_scaled]
        st.info("Returning unscaled data as placeholder.")
        return [sleep_input, weather_input, stress_diet_input]

    except KeyError as e:
        st.error(f"Missing feature in input data: {e}")
        return None
    except Exception as e:
        st.error(f"Error preparing data for prediction: {e}")
        return None


if __name__ == "__main__":
    main()
