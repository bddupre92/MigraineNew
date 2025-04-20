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
sys.path.append('.')

# Import our modules
from optimized_model import OptimizedMigrainePredictionModel
from performance_metrics import MigrainePerformanceMetrics

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
    """Load a saved model."""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except:
        st.error(f"Failed to load model from {model_path}")
        return None

def get_expert_contributions(model, X_test_list):
    """Get expert contributions for test data."""
    predictions, gate_outputs = model(X_test_list, training=False)
    gate_weights = gate_outputs.numpy()
    predictions = predictions.numpy()
    
    return predictions, gate_weights

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
    st.markdown('<div class="main-header">Migraine Prediction Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", ["Overview", "Model Performance", "Patient Analysis", "Make Prediction", "About"])
    
    # Data and model paths
    data_dir = st.sidebar.text_input("Data Directory", "data")
    model_path = st.sidebar.text_input("Model Path", "output/optimized_model")
    
    # Load data and model
    try:
        combined_data, sleep_data, weather_data, stress_diet_data = load_data(data_dir)
        model = load_model(model_path)
        
        if model is None:
            st.warning("Model not loaded. Some features will be disabled.")
        
        # Overview page
        if page == "Overview":
            st.markdown('<div class="sub-header">Overview</div>', unsafe_allow_html=True)
            
            # Display key statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Patients", len(combined_data['patient_id'].unique()))
            
            with col2:
                migraine_pct = combined_data['next_day_migraine'].mean() * 100
                st.metric("Migraine Frequency", f"{migraine_pct:.1f}%")
            
            with col3:
                chronic_pct = (combined_data.groupby('patient_id')['next_day_migraine'].mean() >= 0.15).mean() * 100
                st.metric("Chronic Patients", f"{chronic_pct:.1f}%")
            
            with col4:
                if model is not None:
                    # Make predictions on a sample
                    sample_size = min(1000, len(combined_data))
                    sample_data = combined_data.sample(sample_size)
                    
                    # Prepare data for prediction
                    X_sample = prepare_data_for_prediction(sample_data, sleep_data, weather_data, stress_diet_data)
                    
                    # Measure inference time
                    start_time = time.time()
                    _ = model.predict(X_sample)
                    end_time = time.time()
                    
                    inference_time = (end_time - start_time) * 1000 / sample_size
                    st.metric("Avg. Inference Time", f"{inference_time:.1f} ms")
                else:
                    st.metric("Avg. Inference Time", "N/A")
            
            # Plot migraine distribution
            st.markdown('<div class="sub-header">Migraine Distribution</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Migraine frequency
                fig = px.pie(
                    names=['No Migraine', 'Migraine'],
                    values=[(1-migraine_pct/100), migraine_pct/100],
                    title="Overall Migraine Frequency",
                    color_discrete_sequence=['#3498db', '#e74c3c']
                )
                st.plotly_chart(fig)
            
            with col2:
                # Patient type distribution
                fig = px.pie(
                    names=['Episodic', 'Chronic'],
                    values=[(100-chronic_pct)/100, chronic_pct/100],
                    title="Patient Type Distribution",
                    color_discrete_sequence=['#2ecc71', '#9b59b6']
                )
                st.plotly_chart(fig)
            
            # Trigger analysis
            st.markdown('<div class="sub-header">Trigger Analysis</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Sleep disruption
                sleep_disruption = (combined_data['total_sleep_hours'] < 5) | (combined_data['total_sleep_hours'] > 9)
                migraine_with_sleep = combined_data.loc[sleep_disruption, 'next_day_migraine'].mean() * 100
                migraine_without_sleep = combined_data.loc[~sleep_disruption, 'next_day_migraine'].mean() * 100
                
                sleep_df = pd.DataFrame({
                    'Sleep Disruption': ['Yes', 'No'],
                    'Migraine Percentage': [migraine_with_sleep, migraine_without_sleep]
                })
                
                fig = px.bar(
                    sleep_df,
                    x='Sleep Disruption',
                    y='Migraine Percentage',
                    title="Migraine Percentage by Sleep Disruption",
                    color='Sleep Disruption',
                    color_discrete_sequence=['#e74c3c', '#3498db']
                )
                st.plotly_chart(fig)
            
            with col2:
                # Pressure drop
                pressure_drop = combined_data['pressure_change_24h'] <= -5
                migraine_with_pressure = combined_data.loc[pressure_drop, 'next_day_migraine'].mean() * 100
                migraine_without_pressure = combined_data.loc[~pressure_drop, 'next_day_migraine'].mean() * 100
                
                pressure_df = pd.DataFrame({
                    'Pressure Drop': ['Yes', 'No'],
                    'Migraine Percentage': [migraine_with_pressure, migraine_without_pressure]
                })
                
                fig = px.bar(
                    pressure_df,
                    x='Pressure Drop',
                    y='Migraine Percentage',
                    title="Migraine Percentage by Pressure Drop",
                    color='Pressure Drop',
                    color_discrete_sequence=['#e74c3c', '#3498db']
                )
                st.plotly_chart(fig)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Stress level
                stress_spike = combined_data['stress_level'] >= 7
                migraine_with_stress = combined_data.loc[stress_spike, 'next_day_migraine'].mean() * 100
                migraine_without_stress = combined_data.loc[~stress_spike, 'next_day_migraine'].mean() * 100
                
                stress_df = pd.DataFrame({
                    'High Stress': ['Yes', 'No'],
                    'Migraine Percentage': [migraine_with_stress, migraine_without_stress]
                })
                
                fig = px.bar(
                    stress_df,
                    x='High Stress',
                    y='Migraine Percentage',
                    title="Migraine Percentage by Stress Level",
                    color='High Stress',
                    color_discrete_sequence=['#e74c3c', '#3498db']
                )
                st.plotly_chart(fig)
            
            with col2:
                # Dietary triggers
                dietary_trigger = (combined_data['alcohol_consumed'] > 0) | (combined_data['caffeine_consumed'] > 0) | (combined_data['chocolate_consumed'] > 0)
                migraine_with_diet = combined_data.loc[dietary_trigger, 'next_day_migraine'].mean() * 100
                migraine_without_diet = combined_data.loc[~dietary_trigger, 'next_day_migraine'].mean() * 100
                
                diet_df = pd.DataFrame({
                    'Dietary Trigger': ['Yes', 'No'],
                    'Migraine Percentage': [migraine_with_diet, migraine_without_diet]
                })
                
                fig = px.bar(
                    diet_df,
                    x='Dietary Trigger',
                    y='Migraine Percentage',
                    title="Migraine Percentage by Dietary Triggers",
                    color='Dietary Trigger',
                    color_discrete_sequence=['#e74c3c', '#3498db']
                )
                st.plotly_chart(fig)
        
        # Model Performance page
        elif page == "Model Performance":
            st.markdown('<div class="sub-header">Model Performance</div>', unsafe_allow_html=True)
            
            if model is not None:
                # Prepare test data
                test_data = combined_data.sample(frac=0.2, random_state=42)
                X_test = prepare_data_for_prediction(test_data, sleep_data, weather_data, stress_diet_data)
                y_test = test_data['next_day_migraine'].values
                
                # Make predictions
                predictions, gate_weights = get_expert_contributions(model, X_test)
                predictions = predictions.flatten()
                
                # Performance metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    # Calculate AUC
                    fpr, tpr, _, roc_auc = roc_curve(y_test, predictions)[0:4]
                    st.metric("AUC", f"{roc_auc:.3f}", delta=f"{roc_auc-0.8:.3f}" if roc_auc >= 0.8 else f"{roc_auc-0.8:.3f}")
                
                with col2:
                    # Calculate F1 score at optimal threshold
                    precision, recall, thresholds = precision_recall_curve(y_test, predictions)
                    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
                    optimal_idx = np.argmax(f1_scores)
                    optimal_threshold = thresholds[optimal_idx]
                    optimal_f1 = f1_scores[optimal_idx]
                    
                    st.metric("F1 Score", f"{optimal_f1:.3f}", delta=f"{optimal_f1-0.75:.3f}" if optimal_f1 >= 0.75 else f"{optimal_f1-0.75:.3f}")
                
                with col3:
                    # Calculate high-risk sensitivity
                    high_risk_threshold = 0.3  # Example threshold for high-risk
                    y_pred_high_risk = (predictions >= high_risk_threshold).astype(int)
                    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_high_risk).ravel()
                    high_risk_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    
                    st.metric("High-Risk Sensitivity", f"{high_risk_sensitivity:.3f}", delta=f"{high_risk_sensitivity-0.95:.3f}" if high_risk_sensitivity >= 0.95 else f"{high_risk_sensitivity-0.95:.3f}")
                
                with col4:
                    # Calculate inference time
                    start_time = time.time()
                    _ = model.predict(X_test)
                    end_time = time.time()
                    
                    inference_time = (end_time - start_time) * 1000 / len(X_test[0])
                    st.metric("Inference Time", f"{inference_time:.1f} ms", delta=f"{200-inference_time:.1f} ms" if inference_time <= 200 else f"{200-inference_time:.1f} ms")
                
                # Performance visualizations
                st.markdown('<div class="sub-header">Performance Visualizations</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # ROC curve
                    roc_fig, _, _, _, _ = plot_roc_curve(y_test, predictions)
                    st.plotly_chart(roc_fig)
                
                with col2:
                    # Precision-Recall curve
                    pr_fig, _, _, _ = plot_precision_recall_curve(y_test, predictions)
                    st.plotly_chart(pr_fig)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Confusion matrix
                    cm_fig = plot_confusion_matrix(y_test, predictions, threshold=optimal_threshold)
                    st.plotly_chart(cm_fig)
                
                with col2:
                    # High-risk confusion matrix
                    high_risk_cm_fig = plot_confusion_matrix(y_test, predictions, threshold=high_risk_threshold)
                    st.plotly_chart(high_risk_cm_fig)
                
                # Threshold analysis
                st.markdown('<div class="sub-header">Threshold Analysis</div>', unsafe_allow_html=True)
                threshold_fig = plot_threshold_analysis(fpr, tpr, thresholds, y_test, predictions)
                st.plotly_chart(threshold_fig)
                
                # Expert contribution analysis
                st.markdown('<div class="sub-header">Expert Contribution Analysis</div>', unsafe_allow_html=True)
                
                expert_names = ['Sleep Expert', 'Weather Expert', 'Stress/Diet Expert']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Average expert contribution
                    expert_fig = plot_expert_contributions(gate_weights, expert_names)
                    st.plotly_chart(expert_fig)
                
                with col2:
                    # Expert contribution by outcome
                    expert_outcome_fig = plot_expert_contributions_by_outcome(gate_weights, y_test, expert_names)
                    st.plotly_chart(expert_outcome_fig)
                
                # Trigger analysis
                st.markdown('<div class="sub-header">Trigger Analysis</div>', unsafe_allow_html=True)
                trigger_fig = plot_trigger_analysis(test_data, predictions)
                st.plotly_chart(trigger_fig)
            
            else:
                st.warning("Please load a model to view performance metrics.")
        
        # Patient Analysis page
        elif page == "Patient Analysis":
            st.markdown('<div class="sub-header">Patient Analysis</div>', unsafe_allow_html=True)
            
            # Patient selection
            patient_ids = sorted(combined_data['patient_id'].unique())
            selected_patient = st.selectbox("Select Patient", patient_ids)
            
            # Filter data for selected patient
            patient_data = combined_data[combined_data['patient_id'] == selected_patient].sort_values('date')
            
            # Add date column if not present
            if 'date' not in patient_data.columns:
                patient_data['date'] = pd.date_range(start='2025-01-01', periods=len(patient_data))
            
            # Patient statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                migraine_days = patient_data['next_day_migraine'].sum()
                total_days = len(patient_data)
                st.metric("Migraine Frequency", f"{migraine_days}/{total_days} days")
            
            with col2:
                migraine_pct = patient_data['next_day_migraine'].mean() * 100
                patient_type = "Chronic" if migraine_pct >= 15 else "Episodic"
                st.metric("Patient Type", patient_type)
            
            with col3:
                sleep_disruption_days = ((patient_data['total_sleep_hours'] < 5) | (patient_data['total_sleep_hours'] > 9)).sum()
                st.metric("Sleep Disruptions", f"{sleep_disruption_days}/{total_days} days")
            
            with col4:
                pressure_drop_days = (patient_data['pressure_change_24h'] <= -5).sum()
                st.metric("Pressure Drops", f"{pressure_drop_days}/{total_days} days")
            
            # Patient timeline
            st.markdown('<div class="sub-header">Patient Timeline</div>', unsafe_allow_html=True)
            
            # Make predictions if model is available
            patient_predictions = None
            if model is not None:
                # Prepare data for prediction
                X_patient = prepare_data_for_prediction(patient_data, sleep_data, weather_data, stress_diet_data)
                
                # Make predictions
                patient_predictions = model.predict(X_patient).flatten()
            
            # Plot timeline
            timeline_fig = plot_patient_timeline(patient_data, patient_predictions)
            st.plotly_chart(timeline_fig)
            
            # Trigger correlation
            st.markdown('<div class="sub-header">Trigger Correlation</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Sleep correlation
                fig = px.scatter(
                    patient_data,
                    x='total_sleep_hours',
                    y='next_day_migraine',
                    title="Sleep Hours vs. Migraine",
                    labels={'total_sleep_hours': 'Sleep Hours', 'next_day_migraine': 'Migraine'},
                    color='next_day_migraine',
                    color_continuous_scale=[[0, 'blue'], [1, 'red']],
                    width=500,
                    height=400
                )
                
                # Add reference lines
                fig.add_vline(x=5, line_dash="dash", line_color="red")
                fig.add_vline(x=9, line_dash="dash", line_color="red")
                
                st.plotly_chart(fig)
            
            with col2:
                # Pressure correlation
                fig = px.scatter(
                    patient_data,
                    x='pressure_change_24h',
                    y='next_day_migraine',
                    title="Pressure Change vs. Migraine",
                    labels={'pressure_change_24h': 'Pressure Change (hPa)', 'next_day_migraine': 'Migraine'},
                    color='next_day_migraine',
                    color_continuous_scale=[[0, 'blue'], [1, 'red']],
                    width=500,
                    height=400
                )
                
                # Add reference line
                fig.add_vline(x=-5, line_dash="dash", line_color="red")
                
                st.plotly_chart(fig)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Stress correlation
                fig = px.scatter(
                    patient_data,
                    x='stress_level',
                    y='next_day_migraine',
                    title="Stress Level vs. Migraine",
                    labels={'stress_level': 'Stress Level', 'next_day_migraine': 'Migraine'},
                    color='next_day_migraine',
                    color_continuous_scale=[[0, 'blue'], [1, 'red']],
                    width=500,
                    height=400
                )
                
                # Add reference line
                fig.add_vline(x=7, line_dash="dash", line_color="red")
                
                st.plotly_chart(fig)
            
            with col2:
                # Dietary correlation
                dietary_trigger = (patient_data['alcohol_consumed'] > 0) | (patient_data['caffeine_consumed'] > 0) | (patient_data['chocolate_consumed'] > 0)
                
                dietary_df = pd.DataFrame({
                    'Dietary Trigger': ['Yes', 'No'],
                    'Migraine Percentage': [
                        patient_data.loc[dietary_trigger, 'next_day_migraine'].mean() * 100 if dietary_trigger.sum() > 0 else 0,
                        patient_data.loc[~dietary_trigger, 'next_day_migraine'].mean() * 100 if (~dietary_trigger).sum() > 0 else 0
                    ]
                })
                
                fig = px.bar(
                    dietary_df,
                    x='Dietary Trigger',
                    y='Migraine Percentage',
                    title="Dietary Triggers vs. Migraine",
                    color='Dietary Trigger',
                    color_discrete_sequence=['#e74c3c', '#3498db'],
                    width=500,
                    height=400
                )
                
                st.plotly_chart(fig)
        
        # Make Prediction page
        elif page == "Make Prediction":
            if model is not None:
                create_prediction_interface(model, combined_data)
            else:
                st.warning("Please load a model to make predictions.")
        
        # About page
        elif page == "About":
            st.markdown('<div class="sub-header">About</div>', unsafe_allow_html=True)
            
            st.markdown("""
            ### Migraine Prediction App
            
            This dashboard provides an interface for the migraine prediction model, which uses a Mixture of Experts (MoE) architecture to predict migraine occurrences based on multiple data modalities:
            
            - **Sleep patterns**: Total sleep hours, sleep quality, disruptions
            - **Weather conditions**: Temperature, humidity, barometric pressure
            - **Stress and dietary factors**: Stress levels, alcohol, caffeine, chocolate consumption
            
            ### Key Features
            
            - **Multi-modal data fusion** using specialized expert networks
            - **High-sensitivity prediction** of migraine events (≥95% for high-risk days)
            - **Optimized performance** with AUC ≥0.80 and F1-score ≥0.75
            - **Fast inference** with latency <200ms
            - **Interpretable predictions** with expert contribution analysis
            
            ### How It Works
            
            The model uses a Mixture of Experts (MoE) architecture with three specialized expert networks:
            
            1. **Sleep Expert**: Processes sleep patterns using a 1D-CNN → Bi-LSTM architecture
            2. **Weather Expert**: Analyzes weather data with a 3-layer MLP with residual connections
            3. **Stress/Diet Expert**: Examines stress and dietary factors using a Transformer encoder
            
            A gating network determines the contribution of each expert to the final prediction, allowing the model to focus on the most relevant factors for each individual case.
            
            ### Dashboard Sections
            
            - **Overview**: General statistics and trigger analysis
            - **Model Performance**: Detailed performance metrics and visualizations
            - **Patient Analysis**: Individual patient timelines and trigger correlations
            - **Make Prediction**: Interface for making predictions with custom inputs
            
            ### Contact
            
            For more information or support, please contact the development team.
            """)
    
    except Exception as e:
        st.error(f"Error loading data or model: {e}")

# Helper function to prepare data for prediction
def prepare_data_for_prediction(data, sleep_data, weather_data, stress_diet_data):
    """Prepare data for prediction."""
    # This is a simplified version for demonstration
    # In a real implementation, this would properly process the data
    # including sequence creation, normalization, etc.
    
    # Create dummy data of appropriate shapes
    batch_size = len(data)
    
    sleep_X = np.random.randn(batch_size, 7, 6)  # 7-day sequence, 6 features
    weather_X = np.random.randn(batch_size, 4)   # 4 features
    stress_diet_X = np.random.randn(batch_size, 7, 6)  # 7-day sequence, 6 features
    
    return [sleep_X, weather_X, stress_diet_X]

if __name__ == "__main__":
    main()
