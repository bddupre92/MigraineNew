import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import expert dashboard component
try:
    from dashboard.expert_dashboard import create_expert_dashboard
except ImportError:
    # Try relative import if package import fails
    from .expert_dashboard import create_expert_dashboard

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
    .model-card {
        background-color: #F8F8FF;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def load_data():
    """Load data for the dashboard."""
    try:
        # Try to load the data from the output directory
        data_path = os.path.join('output', 'data')
        train_data = np.load(os.path.join(data_path, 'train_data.npz'))
        val_data = np.load(os.path.join(data_path, 'val_data.npz'))
        test_data = np.load(os.path.join(data_path, 'test_data.npz'))
        
        # Extract data
        X_sleep_train = train_data['X_sleep']
        X_weather_train = train_data['X_weather']
        X_stress_diet_train = train_data['X_stress_diet']
        X_physio_train = train_data['X_physio']
        y_train = train_data['y']
        
        X_sleep_val = val_data['X_sleep']
        X_weather_val = val_data['X_weather']
        X_stress_diet_val = val_data['X_stress_diet']
        X_physio_val = val_data['X_physio']
        y_val = val_data['y']
        
        X_sleep_test = test_data['X_sleep']
        X_weather_test = test_data['X_weather']
        X_stress_diet_test = test_data['X_stress_diet']
        X_physio_test = test_data['X_physio']
        y_test = test_data['y']
        
        # Combine data for visualization
        data = {
            'train': {
                'X_sleep': X_sleep_train,
                'X_weather': X_weather_train,
                'X_stress_diet': X_stress_diet_train,
                'X_physio': X_physio_train,
                'y': y_train
            },
            'val': {
                'X_sleep': X_sleep_val,
                'X_weather': X_weather_val,
                'X_stress_diet': X_stress_diet_val,
                'X_physio': X_physio_val,
                'y': y_val
            },
            'test': {
                'X_sleep': X_sleep_test,
                'X_weather': X_weather_test,
                'X_stress_diet': X_stress_diet_test,
                'X_physio': X_physio_test,
                'y': y_test
            }
        }
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Generate mock data for demonstration
        st.warning("Using generated mock data for demonstration")
        
        # Generate mock data
        n_train = 2000
        n_val = 200
        n_test = 400
        
        # Sleep data: 7 days, 6 features per day
        X_sleep_train = np.random.random((n_train, 7 * 6))
        X_sleep_val = np.random.random((n_val, 7 * 6))
        X_sleep_test = np.random.random((n_test, 7 * 6))
        
        # Weather data: 4 features
        X_weather_train = np.random.random((n_train, 4))
        X_weather_val = np.random.random((n_val, 4))
        X_weather_test = np.random.random((n_test, 4))
        
        # Stress/diet data: 7 days, 6 features per day
        X_stress_diet_train = np.random.random((n_train, 7 * 6))
        X_stress_diet_val = np.random.random((n_val, 7 * 6))
        X_stress_diet_test = np.random.random((n_test, 7 * 6))
        
        # Physiological data: 5 features
        X_physio_train = np.random.random((n_train, 5))
        X_physio_val = np.random.random((n_val, 5))
        X_physio_test = np.random.random((n_test, 5))
        
        # Labels: 10% positive rate for val/test, 50% for train (balanced)
        y_train = np.random.choice([0, 1], size=n_train, p=[0.5, 0.5])
        y_val = np.random.choice([0, 1], size=n_val, p=[0.9, 0.1])
        y_test = np.random.choice([0, 1], size=n_test, p=[0.9, 0.1])
        
        # Combine data for visualization
        data = {
            'train': {
                'X_sleep': X_sleep_train,
                'X_weather': X_weather_train,
                'X_stress_diet': X_stress_diet_train,
                'X_physio': X_physio_train,
                'y': y_train
            },
            'val': {
                'X_sleep': X_sleep_val,
                'X_weather': X_weather_val,
                'X_stress_diet': X_stress_diet_val,
                'X_physio': X_physio_val,
                'y': y_val
            },
            'test': {
                'X_sleep': X_sleep_test,
                'X_weather': X_weather_test,
                'X_stress_diet': X_stress_diet_test,
                'X_physio': X_physio_test,
                'y': y_test
            }
        }
        
        return data

def preprocess_data(data):
    """Preprocess data for model training."""
    # Ensure all arrays are 2D before stacking
    def ensure_2d(arr):
        """Convert array to 2D if it's not already."""
        if len(arr.shape) == 3:
            # Flatten 3D array to 2D (samples, features)
            return arr.reshape(arr.shape[0], -1)
        return arr
    
    # Preprocess and combine all features
    X_train = np.hstack([
        ensure_2d(data['train']['X_sleep']),
        ensure_2d(data['train']['X_weather']),
        ensure_2d(data['train']['X_stress_diet']),
        ensure_2d(data['train']['X_physio'])
    ])
    
    X_val = np.hstack([
        ensure_2d(data['val']['X_sleep']),
        ensure_2d(data['val']['X_weather']),
        ensure_2d(data['val']['X_stress_diet']),
        ensure_2d(data['val']['X_physio'])
    ])
    
    X_test = np.hstack([
        ensure_2d(data['test']['X_sleep']),
        ensure_2d(data['test']['X_weather']),
        ensure_2d(data['test']['X_stress_diet']),
        ensure_2d(data['test']['X_physio'])
    ])
    
    y_train = data['train']['y']
    y_val = data['val']['y']
    y_test = data['test']['y']
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, scaler

def train_simple_model(X_train, y_train):
    """Train a simple model for demonstration."""
    # Train a Random Forest classifier
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X, y):
    """Evaluate model performance."""
    # Get predictions
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Find optimal threshold using F1 score
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    
    thresholds = np.linspace(0.1, 0.9, 9)
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y, y_pred)
        f1_scores.append(f1)
    
    # Get optimal threshold
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Apply optimal threshold
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    # Return metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': roc_auc,
        'optimal_threshold': optimal_threshold,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'y_pred_proba': y_pred_proba
    }
    
    return metrics

def plot_class_distribution(data):
    """Plot class distribution."""
    # Calculate class distribution
    train_pos_ratio = np.mean(data['train']['y'])
    val_pos_ratio = np.mean(data['val']['y'])
    test_pos_ratio = np.mean(data['test']['y'])
    
    # Create figure
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        x=['Train', 'Validation', 'Test'],
        y=[train_pos_ratio, val_pos_ratio, test_pos_ratio],
        text=[f"{train_pos_ratio:.2%}", f"{val_pos_ratio:.2%}", f"{test_pos_ratio:.2%}"],
        textposition='auto',
        name='Positive Class Ratio',
        marker_color='purple'
    ))
    
    # Update layout
    fig.update_layout(
        title='Class Distribution (Positive Ratio)',
        xaxis_title='Dataset',
        yaxis_title='Positive Class Ratio',
        yaxis=dict(tickformat='.0%'),
        height=400
    )
    
    return fig

def plot_feature_importance(model, feature_names):
    """Plot feature importance."""
    # Get feature importance
    importances = model.feature_importances_
    
    # Sort feature importance
    indices = np.argsort(importances)[::-1]
    
    # Get top 20 features
    top_n = 20
    top_indices = indices[:top_n]
    top_importances = importances[top_indices]
    top_names = [feature_names[i] for i in top_indices]
    
    # Create figure
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        x=top_importances,
        y=top_names,
        orientation='h',
        marker_color='purple'
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Top {top_n} Feature Importance',
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=600
    )
    
    return fig

def plot_roc_curve(metrics):
    """Plot ROC curve."""
    # Create figure
    fig = go.Figure()
    
    # Add ROC curve
    fig.add_trace(go.Scatter(
        x=metrics['fpr'],
        y=metrics['tpr'],
        mode='lines',
        name=f'ROC Curve (AUC = {metrics["auc"]:.3f})',
        line=dict(color='purple', width=2)
    ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=1, dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500,
        legend=dict(x=0.01, y=0.99)
    )
    
    return fig

def plot_confusion_matrix(metrics):
    """Plot confusion matrix."""
    # Get confusion matrix
    cm = metrics['confusion_matrix']
    
    # Create figure
    fig = go.Figure()
    
    # Add heatmap
    fig.add_trace(go.Heatmap(
        z=cm,
        x=['No Migraine', 'Migraine'],
        y=['No Migraine', 'Migraine'],
        colorscale='Purples',
        showscale=False,
        text=cm,
        texttemplate="%{text}",
        textfont=dict(size=14)
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Confusion Matrix (Threshold: {metrics["optimal_threshold"]:.2f})',
        height=400
    )
    
    return fig

def plot_threshold_analysis(y_true, y_pred_proba):
    """Plot threshold analysis."""
    # Calculate precision, recall, and F1 for different thresholds
    thresholds = np.linspace(0.1, 0.9, 9)
    precision_values = []
    recall_values = []
    f1_values = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
    
    # Create figure
    fig = go.Figure()
    
    # Add lines
    fig.add_trace(go.Scatter(
        x=thresholds,
        y=precision_values,
        mode='lines+markers',
        name='Precision',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=thresholds,
        y=recall_values,
        mode='lines+markers',
        name='Recall',
        line=dict(color='red', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=thresholds,
        y=f1_values,
        mode='lines+markers',
        name='F1 Score',
        line=dict(color='green', width=2)
    ))
    
    # Update layout
    fig.update_layout(
        title='Threshold Analysis',
        xaxis_title='Threshold',
        yaxis_title='Score',
        height=500,
        legend=dict(x=0.01, y=0.99)
    )
    
    return fig

def create_prediction_tool(model, scaler):
    """Create an interactive prediction tool."""
    st.markdown("<h2 class='sub-header'>Migraine Prediction Tool</h2>", unsafe_allow_html=True)
    
    # Create columns for input
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Sleep Factors")
        sleep_duration = st.slider("Average Sleep Duration (hours)", 4.0, 10.0, 7.0, 0.1)
        sleep_quality = st.slider("Sleep Quality (0-10)", 0, 10, 7, 1)
        sleep_interruptions = st.slider("Sleep Interruptions (count)", 0, 10, 2, 1)
        deep_sleep_pct = st.slider("Deep Sleep (%)", 0, 50, 25, 1)
        rem_sleep_pct = st.slider("REM Sleep (%)", 0, 40, 20, 1)
    
    with col2:
        st.subheader("Weather Factors")
        temperature = st.slider("Temperature (°C)", -10.0, 40.0, 22.0, 0.5)
        humidity = st.slider("Humidity (%)", 0, 100, 50, 1)
        pressure = st.slider("Barometric Pressure (hPa)", 980, 1040, 1013, 1)
        weather_change = st.checkbox("Significant Weather Change")
    
    with col3:
        st.subheader("Physiological & Lifestyle")
        stress_level = st.slider("Stress Level (0-10)", 0, 10, 5, 1)
        hydration = st.slider("Hydration Level (0-10)", 0, 10, 7, 1)
        caffeine = st.slider("Caffeine Intake (cups)", 0, 10, 2, 1)
        alcohol = st.slider("Alcohol Consumption (drinks)", 0, 10, 0, 1)
        exercise = st.slider("Exercise (minutes)", 0, 120, 30, 5)
    
    # Create a button to trigger prediction
    if st.button("Predict Migraine Risk"):
        # Create feature vector
        # This is a simplified version - in a real app, you'd need to match the exact feature format
        features = np.array([
            # Sleep features (repeated for 7 days for simplicity)
            *[sleep_duration, sleep_quality, sleep_interruptions, deep_sleep_pct, rem_sleep_pct, sleep_duration/10] * 7,
            
            # Weather features
            temperature, humidity, pressure, 1 if weather_change else 0,
            
            # Stress/diet features (repeated for 7 days for simplicity)
            *[stress_level, hydration, caffeine, alcohol, exercise, stress_level/hydration] * 7,
            
            # Physiological features
            stress_level, hydration, caffeine, alcohol, exercise
        ])
        
        # Reshape and scale
        features = features.reshape(1, -1)
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction_proba = model.predict_proba(features_scaled)[0, 1]
        
        # Display result
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        
        if prediction_proba < 0.3:
            risk_level = "Low"
            color = "green"
        elif prediction_proba < 0.7:
            risk_level = "Moderate"
            color = "orange"
        else:
            risk_level = "High"
            color = "red"
        
        st.markdown(f"<h3 style='color:{color}'>Migraine Risk: {risk_level} ({prediction_proba:.1%})</h3>", unsafe_allow_html=True)
        
        # Add recommendations based on inputs
        st.markdown("<h4>Recommendations:</h4>", unsafe_allow_html=True)
        
        recommendations = []
        
        if sleep_duration < 7:
            recommendations.append("Increase sleep duration to at least 7 hours")
        
        if sleep_quality < 7:
            recommendations.append("Improve sleep quality with a consistent sleep schedule")
        
        if stress_level > 7:
            recommendations.append("Reduce stress through relaxation techniques")
        
        if hydration < 7:
            recommendations.append("Increase water intake")
        
        if caffeine > 3:
            recommendations.append("Reduce caffeine consumption")
        
        if alcohol > 1:
            recommendations.append("Limit alcohol intake")
        
        if weather_change:
            recommendations.append("Be prepared for migraine triggers during weather changes")
        
        if not recommendations:
            recommendations.append("Continue your current healthy habits")
        
        for rec in recommendations:
            st.markdown(f"- {rec}")
        
        st.markdown("</div>", unsafe_allow_html=True)

def main():
    """Main function to run the dashboard."""
    # Header
    st.markdown("<h1 class='main-header'>Migraine Prediction Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-text'>
    This dashboard demonstrates migraine prediction using machine learning techniques.
    It includes data visualization, model performance metrics, and an interactive prediction tool.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", [
        "Overview",
        "Data Analysis",
        "Model Performance",
        "Prediction Tool",
        "Implementation Details",
        "Expert Analysis"  # New page for expert dashboard
    ])
    
    # Load data
    data = load_data()
    
    # Preprocess data
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = preprocess_data(data)
    
    # Train model if not in session state
    if 'model' not in st.session_state:
        with st.spinner("Training model..."):
            model = train_simple_model(X_train, y_train)
            st.session_state['model'] = model
    else:
        model = st.session_state['model']
    
    # Evaluate model if not in session state
    if 'metrics' not in st.session_state:
        with st.spinner("Evaluating model..."):
            metrics = evaluate_model(model, X_test, y_test)
            st.session_state['metrics'] = metrics
    else:
        metrics = st.session_state['metrics']
    
    # Display selected page
    if page == "Overview":
        st.markdown("<h2 class='sub-header'>Project Overview</h2>", unsafe_allow_html=True)
        
        # Project description
        st.markdown("""
        <div class='highlight'>
        <h3>Migraine Prediction System</h3>
        <p>This project implements a machine learning system to predict migraine occurrences based on various factors including:</p>
        <ul>
            <li><strong>Sleep patterns</strong>: Duration, quality, interruptions, deep sleep percentage, REM sleep percentage</li>
            <li><strong>Weather conditions</strong>: Temperature, humidity, barometric pressure, weather changes</li>
            <li><strong>Stress and diet</strong>: Stress levels, hydration, caffeine intake, alcohol consumption</li>
            <li><strong>Physiological factors</strong>: Exercise, hormonal changes, previous migraine patterns</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Key metrics
        st.markdown("<h3>Model Performance</h3>", unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        
        with col2:
            st.metric("Precision", f"{metrics['precision']:.2%}")
        
        with col3:
            st.metric("Recall", f"{metrics['recall']:.2%}")
        
        with col4:
            st.metric("F1 Score", f"{metrics['f1']:.2%}")
        
        with col5:
            st.metric("AUC", f"{metrics['auc']:.3f}")
        
        # Class distribution
        st.markdown("<h3>Data Overview</h3>", unsafe_allow_html=True)
        st.plotly_chart(plot_class_distribution(data), use_container_width=True)
        
        # Confusion matrix
        st.plotly_chart(plot_confusion_matrix(metrics), use_container_width=True)
    
    elif page == "Data Analysis":
        st.markdown("<h2 class='sub-header'>Data Analysis</h2>", unsafe_allow_html=True)
        
        # Data statistics
        st.markdown("<h3>Dataset Statistics</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Training Samples", len(data['train']['y']))
            st.metric("Positive Ratio (Train)", f"{np.mean(data['train']['y']):.2%}")
        
        with col2:
            st.metric("Validation Samples", len(data['val']['y']))
            st.metric("Positive Ratio (Val)", f"{np.mean(data['val']['y']):.2%}")
        
        with col3:
            st.metric("Test Samples", len(data['test']['y']))
            st.metric("Positive Ratio (Test)", f"{np.mean(data['test']['y']):.2%}")
        
        # Feature importance
        st.markdown("<h3>Feature Importance</h3>", unsafe_allow_html=True)
        
        # Create feature names
        feature_names = []
        
        # Sleep features
        for day in range(7):
            for feature in ['Duration', 'Quality', 'Interruptions', 'Deep%', 'REM%', 'Efficiency']:
                feature_names.append(f"Sleep_{feature}_Day{day+1}")
        
        # Weather features
        for feature in ['Temperature', 'Humidity', 'Pressure', 'WeatherChange']:
            feature_names.append(f"Weather_{feature}")
        
        # Stress/diet features
        for day in range(7):
            for feature in ['Stress', 'Hydration', 'Caffeine', 'Alcohol', 'Exercise', 'StressRatio']:
                feature_names.append(f"StressDiet_{feature}_Day{day+1}")
        
        # Physiological features
        for feature in ['StressLevel', 'Hydration', 'Caffeine', 'Alcohol', 'Exercise']:
            feature_names.append(f"Physio_{feature}")
        
        # Plot feature importance
        st.plotly_chart(plot_feature_importance(model, feature_names), use_container_width=True)
        
        # Data correlations
        st.markdown("<h3>Feature Correlations with Migraine</h3>", unsafe_allow_html=True)
        
        # Combine all features
        X_all = np.vstack([X_train, X_val, X_test])
        y_all = np.concatenate([y_train, y_val, y_test])
        
        # Calculate correlation with target
        correlations = []
        
        for i in range(X_all.shape[1]):
            corr = np.corrcoef(X_all[:, i], y_all)[0, 1]
            correlations.append((feature_names[i], corr))
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Get top 20 correlations
        top_correlations = correlations[:20]
        
        # Create figure
        fig = go.Figure()
        
        # Add bars
        fig.add_trace(go.Bar(
            x=[c[1] for c in top_correlations],
            y=[c[0] for c in top_correlations],
            orientation='h',
            marker_color=['red' if c[1] < 0 else 'green' for c in top_correlations]
        ))
        
        # Update layout
        fig.update_layout(
            title='Top 20 Feature Correlations with Migraine',
            xaxis_title='Correlation Coefficient',
            yaxis_title='Feature',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Model Performance":
        st.markdown("<h2 class='sub-header'>Model Performance</h2>", unsafe_allow_html=True)
        
        # Performance metrics
        st.markdown("<h3>Performance Metrics</h3>", unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        
        with col2:
            st.metric("Precision", f"{metrics['precision']:.2%}")
        
        with col3:
            st.metric("Recall", f"{metrics['recall']:.2%}")
        
        with col4:
            st.metric("F1 Score", f"{metrics['f1']:.2%}")
        
        with col5:
            st.metric("AUC", f"{metrics['auc']:.3f}")
        
        # ROC curve
        st.markdown("<h3>ROC Curve</h3>", unsafe_allow_html=True)
        st.plotly_chart(plot_roc_curve(metrics), use_container_width=True)
        
        # Confusion matrix
        st.markdown("<h3>Confusion Matrix</h3>", unsafe_allow_html=True)
        st.plotly_chart(plot_confusion_matrix(metrics), use_container_width=True)
        
        # Threshold analysis
        st.markdown("<h3>Threshold Analysis</h3>", unsafe_allow_html=True)
        st.plotly_chart(plot_threshold_analysis(y_test, metrics['y_pred_proba']), use_container_width=True)
        
        # Explanation
        st.markdown("""
        <div class='info-text'>
        <p><strong>Threshold Analysis:</strong> The optimal threshold was determined to be {:.2f}, which maximizes the F1 score.
        This is particularly important for imbalanced datasets like migraine prediction, where the default threshold of 0.5 may not be optimal.</p>
        </div>
        """.format(metrics['optimal_threshold']), unsafe_allow_html=True)
    
    elif page == "Prediction Tool":
        # Create prediction tool
        create_prediction_tool(model, scaler)
    
    elif page == "Implementation Details":
        st.markdown("<h2 class='sub-header'>Implementation Details</h2>", unsafe_allow_html=True)
        
        # Implementation details
        st.markdown("""
        <div class='highlight'>
        <h3>Optimization Techniques</h3>
        <p>This migraine prediction system implements several advanced techniques to improve performance:</p>
        
        <h4>1. Threshold Optimization</h4>
        <p>Instead of using the default classification threshold of 0.5, we optimize the threshold to maximize the F1 score, 
        which is particularly important for imbalanced datasets like migraine prediction.</p>
        
        <h4>2. Class Balancing</h4>
        <p>We address the class imbalance issue (typically only 10% of samples are positive) using techniques like:</p>
        <ul>
            <li><strong>SMOTE</strong> (Synthetic Minority Over-sampling Technique): Creates synthetic samples of the minority class</li>
            <li><strong>Class weights</strong>: Assigns higher weights to the minority class during training</li>
        </ul>
        
        <h4>3. Feature Engineering</h4>
        <p>We enhance the raw features with derived metrics that capture important patterns:</p>
        <ul>
            <li><strong>Temporal patterns</strong>: Trends, variability, and changes over time</li>
            <li><strong>Interaction features</strong>: Combinations of features that may have synergistic effects</li>
            <li><strong>Domain-specific metrics</strong>: Sleep efficiency, stress-to-hydration ratio, etc.</li>
        </ul>
        
        <h4>4. Ensemble Methods</h4>
        <p>We combine multiple models to improve prediction accuracy:</p>
        <ul>
            <li><strong>Expert models</strong>: Specialized models for each data type (sleep, weather, stress/diet, physiological)</li>
            <li><strong>Random Forest</strong>: An ensemble of decision trees that reduces overfitting and improves generalization</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Architecture diagram
        st.markdown("<h3>System Architecture</h3>", unsafe_allow_html=True)
        
        # Create a simple architecture diagram
        from graphviz import Digraph
        
        # Create a graphviz diagram
        dot = Digraph()
        
        # Add nodes
        dot.node('data', 'Input Data', shape='box')
        dot.node('preprocess', 'Preprocessing', shape='box')
        dot.node('feature_eng', 'Feature Engineering', shape='box')
        dot.node('class_balance', 'Class Balancing', shape='box')
        dot.node('expert_models', 'Expert Models', shape='box')
        dot.node('ensemble', 'Ensemble Model', shape='box')
        dot.node('threshold', 'Threshold Optimization', shape='box')
        dot.node('prediction', 'Final Prediction', shape='box')
        
        # Add edges
        dot.edge('data', 'preprocess')
        dot.edge('preprocess', 'feature_eng')
        dot.edge('feature_eng', 'class_balance')
        dot.edge('class_balance', 'expert_models')
        dot.edge('expert_models', 'ensemble')
        dot.edge('ensemble', 'threshold')
        dot.edge('threshold', 'prediction')
        
        # Render the diagram
        st.graphviz_chart(dot)
        
        # Future improvements
        st.markdown("""
        <div class='highlight'>
        <h3>Future Improvements</h3>
        <p>Several enhancements could further improve the system:</p>
        <ul>
            <li><strong>Deep learning models</strong>: Using recurrent neural networks (RNNs) or transformers to better capture temporal patterns</li>
            <li><strong>Personalization</strong>: Adapting the model to individual users' patterns and triggers</li>
            <li><strong>Additional data sources</strong>: Incorporating hormonal data, medication usage, and dietary information</li>
            <li><strong>Explainable AI</strong>: Providing more detailed explanations of prediction factors</li>
            <li><strong>Real-time monitoring</strong>: Integrating with wearable devices for continuous data collection</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
    elif page == "Expert Analysis":
        # Call the expert dashboard component
        create_expert_dashboard()

if __name__ == "__main__":
    main()
