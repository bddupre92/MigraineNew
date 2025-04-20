import os
import sys
import numpy as np
import json
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add the project root to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# Create mock data for demonstration
def create_mock_data():
    """Create mock data for demonstration purposes."""
    # Performance metrics
    original_metrics = {
        'auc': 0.5625,
        'f1': 0.0741,
        'precision': 0.0667,
        'recall': 0.0833,
        'accuracy': 0.5192,
        'high_risk_sensitivity': 0.0
    }
    
    optimized_metrics = {
        'auc': 0.7845,
        'f1': 0.3571,
        'precision': 0.2500,
        'recall': 0.6250,
        'accuracy': 0.6731,
        'high_risk_sensitivity': 0.5000
    }
    
    # ROC curve data
    fpr_original = np.linspace(0, 1, 100)
    tpr_original = np.power(fpr_original, 0.7)  # Curve below diagonal
    
    fpr_optimized = np.linspace(0, 1, 100)
    tpr_optimized = np.power(fpr_optimized, 0.3)  # Curve above diagonal
    
    # Expert contributions
    expert_contributions = {
        'original': {
            'sleep': 0.35,
            'weather': 0.25,
            'stress_diet': 0.40
        },
        'optimized': {
            'sleep': 0.45,
            'weather': 0.15,
            'stress_diet': 0.40
        }
    }
    
    # Optimization results
    optimization_results = {
        'expert_phase': {
            'sleep': {
                'config': {
                    'conv_filters': 64,
                    'kernel_size': (5,),
                    'lstm_units': 128,
                    'dropout_rate': 0.3,
                    'output_dim': 64
                },
                'fitness': 0.65
            },
            'weather': {
                'config': {
                    'hidden_units': 128,
                    'activation': 'relu',
                    'dropout_rate': 0.2,
                    'output_dim': 64
                },
                'fitness': 0.58
            },
            'stress_diet': {
                'config': {
                    'embedding_dim': 64,
                    'num_heads': 4,
                    'transformer_dim': 64,
                    'dropout_rate': 0.25,
                    'output_dim': 64
                },
                'fitness': 0.70
            }
        },
        'gating_phase': {
            'config': {
                'gate_hidden_size': 128,
                'gate_top_k': 2,
                'load_balance_coef': 0.01
            },
            'fitness': 0.72
        },
        'e2e_phase': {
            'config': {
                'learning_rate': 0.001,
                'batch_size': 32,
                'dropout_rate': 0.3
            },
            'fitness': {
                'auc': 0.7845,
                'latency': 5.2
            }
        },
        'final_performance': {
            'auc_improvement': 0.222,
            'f1_improvement': 0.283,
            'inference_speedup': 1.3
        }
    }
    
    # Mock predictions
    y_true = np.random.randint(0, 2, 100)
    y_pred_original = np.random.random(100) * 0.5 + 0.25  # Random predictions around 0.5
    y_pred_optimized = np.where(y_true == 1, 
                               np.random.random(100) * 0.4 + 0.6,  # Higher values for positive cases
                               np.random.random(100) * 0.4)        # Lower values for negative cases
    
    return {
        'original_metrics': original_metrics,
        'optimized_metrics': optimized_metrics,
        'roc_curve': {
            'original': {'fpr': fpr_original, 'tpr': tpr_original},
            'optimized': {'fpr': fpr_optimized, 'tpr': tpr_optimized}
        },
        'expert_contributions': expert_contributions,
        'optimization_results': optimization_results,
        'predictions': {
            'y_true': y_true,
            'original': y_pred_original,
            'optimized': y_pred_optimized
        }
    }

def main():
    st.set_page_config(
        page_title="Migraine Prediction Model Comparison",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Create mock data
    data = create_mock_data()
    
    # Sidebar
    st.sidebar.title("Migraine Prediction App")
    st.sidebar.image("https://img.freepik.com/free-vector/headache-concept-illustration_114360-8610.jpg", width=200)
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Model Comparison", "Performance Metrics", "Expert Contributions", "Prediction Tool", "Optimization Details"]
    )
    
    # Threshold selection
    threshold = st.sidebar.slider(
        "Prediction Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Adjust the threshold for binary classification"
    )
    
    # Main content
    if page == "Model Comparison":
        display_model_comparison(data, threshold)
    elif page == "Performance Metrics":
        display_performance_metrics(data, threshold)
    elif page == "Expert Contributions":
        display_expert_contributions(data)
    elif page == "Prediction Tool":
        display_prediction_tool(data)
    elif page == "Optimization Details":
        display_optimization_details(data)

def display_model_comparison(data, threshold):
    st.title("Migraine Prediction Model Comparison")
    st.write("Compare the performance of the original FuseMoE model with the PyGMO-optimized version.")
    
    # Performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Model")
        metrics = data['original_metrics']
        st.metric("ROC AUC", f"{metrics['auc']:.4f}")
        st.metric("F1 Score", f"{metrics['f1']:.4f}")
        st.metric("Precision", f"{metrics['precision']:.4f}")
        st.metric("Recall", f"{metrics['recall']:.4f}")
        st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        st.metric("High-Risk Sensitivity", f"{metrics['high_risk_sensitivity']:.4f}")
    
    with col2:
        st.subheader("PyGMO-Optimized Model")
        metrics = data['optimized_metrics']
        st.metric("ROC AUC", f"{metrics['auc']:.4f}", f"+{metrics['auc'] - data['original_metrics']['auc']:.4f}")
        st.metric("F1 Score", f"{metrics['f1']:.4f}", f"+{metrics['f1'] - data['original_metrics']['f1']:.4f}")
        st.metric("Precision", f"{metrics['precision']:.4f}", f"+{metrics['precision'] - data['original_metrics']['precision']:.4f}")
        st.metric("Recall", f"{metrics['recall']:.4f}", f"+{metrics['recall'] - data['original_metrics']['recall']:.4f}")
        st.metric("Accuracy", f"{metrics['accuracy']:.4f}", f"+{metrics['accuracy'] - data['original_metrics']['accuracy']:.4f}")
        st.metric("High-Risk Sensitivity", f"{metrics['high_risk_sensitivity']:.4f}", f"+{metrics['high_risk_sensitivity'] - data['original_metrics']['high_risk_sensitivity']:.4f}")
    
    # ROC Curve
    st.subheader("ROC Curve Comparison")
    fig, ax = plt.figure(figsize=(10, 6)), plt.axes()
    
    # Original model
    ax.plot(
        data['roc_curve']['original']['fpr'],
        data['roc_curve']['original']['tpr'],
        label=f"Original Model (AUC = {data['original_metrics']['auc']:.4f})",
        color='blue'
    )
    
    # Optimized model
    ax.plot(
        data['roc_curve']['optimized']['fpr'],
        data['roc_curve']['optimized']['tpr'],
        label=f"Optimized Model (AUC = {data['optimized_metrics']['auc']:.4f})",
        color='red'
    )
    
    # Diagonal line
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Confusion Matrices
    st.subheader("Confusion Matrices")
    col1, col2 = st.columns(2)
    
    # Calculate confusion matrices based on threshold
    y_true = data['predictions']['y_true']
    y_pred_original = (data['predictions']['original'] >= threshold).astype(int)
    y_pred_optimized = (data['predictions']['optimized'] >= threshold).astype(int)
    
    # Original model confusion matrix
    cm_original = np.zeros((2, 2), dtype=int)
    for i in range(len(y_true)):
        cm_original[y_true[i], y_pred_original[i]] += 1
    
    # Optimized model confusion matrix
    cm_optimized = np.zeros((2, 2), dtype=int)
    for i in range(len(y_true)):
        cm_optimized[y_true[i], y_pred_optimized[i]] += 1
    
    with col1:
        st.write("Original Model")
        fig, ax = plt.figure(figsize=(8, 6)), plt.axes()
        sns.heatmap(cm_original, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Predicted No Migraine', 'Predicted Migraine'],
                   yticklabels=['Actual No Migraine', 'Actual Migraine'])
        ax.set_title(f'Confusion Matrix (Threshold = {threshold:.2f})')
        st.pyplot(fig)
    
    with col2:
        st.write("Optimized Model")
        fig, ax = plt.figure(figsize=(8, 6)), plt.axes()
        sns.heatmap(cm_optimized, annot=True, fmt='d', cmap='Reds', ax=ax,
                   xticklabels=['Predicted No Migraine', 'Predicted Migraine'],
                   yticklabels=['Actual No Migraine', 'Actual Migraine'])
        ax.set_title(f'Confusion Matrix (Threshold = {threshold:.2f})')
        st.pyplot(fig)
    
    # Summary
    st.subheader("Performance Improvement Summary")
    
    improvement = {
        'auc': data['optimized_metrics']['auc'] - data['original_metrics']['auc'],
        'f1': data['optimized_metrics']['f1'] - data['original_metrics']['f1'],
        'precision': data['optimized_metrics']['precision'] - data['original_metrics']['precision'],
        'recall': data['optimized_metrics']['recall'] - data['original_metrics']['recall'],
        'accuracy': data['optimized_metrics']['accuracy'] - data['original_metrics']['accuracy'],
        'high_risk_sensitivity': data['optimized_metrics']['high_risk_sensitivity'] - data['original_metrics']['high_risk_sensitivity']
    }
    
    # Calculate percentage improvements
    pct_improvement = {
        k: (v / data['original_metrics'][k] * 100) if data['original_metrics'][k] > 0 else float('inf')
        for k, v in improvement.items()
    }
    
    # Create a bar chart of improvements
    fig, ax = plt.figure(figsize=(12, 6)), plt.axes()
    
    metrics = list(improvement.keys())
    values = [improvement[m] for m in metrics]
    
    ax.bar(metrics, values, color='green')
    ax.set_title('Absolute Improvement in Performance Metrics')
    ax.set_ylabel('Improvement')
    ax.set_xticklabels(metrics, rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.4f}", ha='center')
    
    st.pyplot(fig)
    
    # Percentage improvement
    fig, ax = plt.figure(figsize=(12, 6)), plt.axes()
    
    metrics = [m for m in metrics if pct_improvement[m] != float('inf')]
    values = [pct_improvement[m] for m in metrics]
    
    ax.bar(metrics, values, color='purple')
    ax.set_title('Percentage Improvement in Performance Metrics')
    ax.set_ylabel('Improvement (%)')
    ax.set_xticklabels(metrics, rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(values):
        ax.text(i, v + 1, f"{v:.1f}%", ha='center')
    
    st.pyplot(fig)

def display_performance_metrics(data, threshold):
    st.title("Performance Metrics Analysis")
    st.write("Detailed analysis of model performance metrics at different thresholds.")
    
    # Generate metrics at different thresholds
    thresholds = np.linspace(0.1, 0.9, 9)
    
    # Calculate metrics for each threshold
    original_metrics = []
    optimized_metrics = []
    
    y_true = data['predictions']['y_true']
    y_pred_original = data['predictions']['original']
    y_pred_optimized = data['predictions']['optimized']
    
    for t in thresholds:
        # Original model
        y_pred_binary = (y_pred_original >= t).astype(int)
        tp = np.sum((y_true == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true == 0) & (y_pred_binary == 1))
        tn = np.sum((y_true == 0) & (y_pred_binary == 0))
        fn = np.sum((y_true == 1) & (y_pred_binary == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        original_metrics.append({
            'threshold': t,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        })
        
        # Optimized model
        y_pred_binary = (y_pred_optimized >= t).astype(int)
        tp = np.sum((y_true == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true == 0) & (y_pred_binary == 1))
        tn = np.sum((y_true == 0) & (y_pred_binary == 0))
        fn = np.sum((y_true == 1) & (y_pred_binary == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        optimized_metrics.append({
            'threshold': t,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        })
    
    # Metric selection
    metric = st.selectbox(
        "Select Metric",
        ["precision", "recall", "f1", "accuracy"],
        index=2
    )
    
    # Plot metrics vs threshold
    st.subheader(f"{metric.capitalize()} vs Threshold")
    fig, ax = plt.figure(figsize=(10, 6)), plt.axes()
    
    # Original model
    ax.plot(
        [m['threshold'] for m in original_metrics],
        [m[metric] for m in original_metrics],
        'o-',
        label="Original Model",
        color='blue'
    )
    
    # Optimized model
    ax.plot(
        [m['threshold'] for m in optimized_metrics],
        [m[metric] for m in optimized_metrics],
        'o-',
        label="Optimized Model",
        color='red'
    )
    
    # Current threshold
    ax.axvline(x=threshold, color='green', linestyle='--', label=f'Current Threshold ({threshold:.2f})')
    
    ax.set_xlabel('Threshold')
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f'{metric.capitalize()} vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Precision-Recall Curve
    st.subheader("Precision-Recall Curve")
    fig, ax = plt.figure(figsize=(10, 6)), plt.axes()
    
    # Original model
    ax.plot(
        [m['recall'] for m in original_metrics],
        [m['precision'] for m in original_metrics],
        'o-',
        label="Original Model",
        color='blue'
    )
    
    # Optimized model
    ax.plot(
        [m['recall'] for m in optimized_metrics],
        [m['precision'] for m in optimized_metrics],
        'o-',
        label="Optimized Model",
        color='red'
    )
    
    # Current threshold points
    original_current = next((m for m in original_metrics if m['threshold'] == threshold), original_metrics[4])
    optimized_current = next((m for m in optimized_metrics if m['threshold'] == threshold), optimized_metrics[4])
    
    ax.plot(original_current['recall'], original_current['precision'], 'o', color='blue', markersize=10)
    ax.plot(optimized_current['recall'], optimized_current['precision'], 'o', color='red', markersize=10)
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Metrics at current threshold
    st.subheader(f"Metrics at Threshold = {threshold:.2f}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Original Model")
        original_current = next((m for m in original_metrics if abs(m['threshold'] - threshold) < 0.01), original_metrics[4])
        st.metric("Precision", f"{original_current['precision']:.4f}")
        st.metric("Recall", f"{original_current['recall']:.4f}")
        st.metric("F1 Score", f"{original_current['f1']:.4f}")
        st.metric("Accuracy", f"{original_current['accuracy']:.4f}")
    
    with col2:
        st.write("Optimized Model")
        optimized_current = next((m for m in optimized_metrics if abs(m['threshold'] - threshold) < 0.01), optimized_metrics[4])
        st.metric("Precision", f"{optimized_current['precision']:.4f}", f"+{optimized_current['precision'] - original_current['precision']:.4f}")
        st.metric("Recall", f"{optimized_current['recall']:.4f}", f"+{optimized_current['recall'] - original_current['recall']:.4f}")
        st.metric("F1 Score", f"{optimized_current['f1']:.4f}", f"+{optimized_current['f1'] - original_current['f1']:.4f}")
        st.metric("Accuracy", f"{optimized_current['accuracy']:.4f}", f"+{optimized_current['accuracy'] - original_current['accuracy']:.4f}")

def display_expert_contributions(data):
    st.title("Expert Contributions Analysis")
    st.write("Analyze how different experts contribute to the migraine prediction.")
    
    # Expert contributions
    contributions = data['expert_contributions']
    
    # Bar chart of expert contributions
    st.subheader("Expert Contribution Weights")
    
    fig, ax = plt.figure(figsize=(10, 6)), plt.axes()
    
    experts = list(contributions['original'].keys())
    x = np.arange(len(experts))
    width = 0.35
    
    original_values = [contributions['original'][e] for e in experts]
    optimized_values = [contributions['optimized'][e] for e in experts]
    
    ax.bar(x - width/2, original_values, width, label='Original Model', color='blue')
    ax.bar(x + width/2, optimized_values, width, label='Optimized Model', color='red')
    
    ax.set_xlabel('Expert')
    ax.set_ylabel('Contribution Weight')
    ax.set_title('Expert Contribution Weights')
    ax.set_xticks(x)
    ax.set_xticklabels([e.capitalize() for e in experts])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(original_values):
        ax.text(i - width/2, v + 0.01, f"{v:.2f}", ha='center')
    
    for i, v in enumerate(optimized_values):
        ax.text(i + width/2, v + 0.01, f"{v:.2f}", ha='center')
    
    st.pyplot(fig)
    
    # Pie charts of expert contributions
    st.subheader("Expert Contribution Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Original Model")
        fig, ax = plt.figure(figsize=(8, 8)), plt.axes()
        ax.pie(
            original_values,
            labels=[e.capitalize() for e in experts],
            autopct='%1.1f%%',
            startangle=90,
            colors=['#3498db', '#2ecc71', '#e74c3c']
        )
        ax.axis('equal')
        st.pyplot(fig)
    
    with col2:
        st.write("Optimized Model")
        fig, ax = plt.figure(figsize=(8, 8)), plt.axes()
        ax.pie(
            optimized_values,
            labels=[e.capitalize() for e in experts],
            autopct='%1.1f%%',
            startangle=90,
            colors=['#3498db', '#2ecc71', '#e74c3c']
        )
        ax.axis('equal')
        st.pyplot(fig)
    
    # Expert contribution changes
    st.subheader("Expert Contribution Changes")
    
    changes = {e: contributions['optimized'][e] - contributions['original'][e] for e in experts}
    
    fig, ax = plt.figure(figsize=(10, 6)), plt.axes()
    
    ax.bar(experts, [changes[e] for e in experts], color=['green' if changes[e] > 0 else 'red' for e in experts])
    ax.set_xlabel('Expert')
    ax.set_ylabel('Change in Contribution Weight')
    ax.set_title('Changes in Expert Contributions After Optimization')
    ax.set_xticklabels([e.capitalize() for e in experts])
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, e in enumerate(experts):
        ax.text(i, changes[e] + 0.01 if changes[e] > 0 else changes[e] - 0.03, f"{changes[e]:+.2f}", ha='center')
    
    st.pyplot(fig)
    
    # Expert performance analysis
    st.subheader("Expert Performance Analysis")
    
    # Mock expert performance data
    expert_performance = {
        'sleep': {
            'original': {'auc': 0.62, 'f1': 0.15},
            'optimized': {'auc': 0.71, 'f1': 0.28}
        },
        'weather': {
            'original': {'auc': 0.55, 'f1': 0.08},
            'optimized': {'auc': 0.60, 'f1': 0.12}
        },
        'stress_diet': {
            'original': {'auc': 0.65, 'f1': 0.18},
            'optimized': {'auc': 0.73, 'f1': 0.32}
        }
    }
    
    # AUC comparison
    fig, ax = plt.figure(figsize=(10, 6)), plt.axes()
    
    x = np.arange(len(experts))
    width = 0.35
    
    original_auc = [expert_performance[e]['original']['auc'] for e in experts]
    optimized_auc = [expert_performance[e]['optimized']['auc'] for e in experts]
    
    ax.bar(x - width/2, original_auc, width, label='Original Model', color='blue')
    ax.bar(x + width/2, optimized_auc, width, label='Optimized Model', color='red')
    
    ax.set_xlabel('Expert')
    ax.set_ylabel('AUC')
    ax.set_title('Expert AUC Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([e.capitalize() for e in experts])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(original_auc):
        ax.text(i - width/2, v + 0.01, f"{v:.2f}", ha='center')
    
    for i, v in enumerate(optimized_auc):
        ax.text(i + width/2, v + 0.01, f"{v:.2f}", ha='center')
    
    st.pyplot(fig)
    
    # F1 comparison
    fig, ax = plt.figure(figsize=(10, 6)), plt.axes()
    
    original_f1 = [expert_performance[e]['original']['f1'] for e in experts]
    optimized_f1 = [expert_performance[e]['optimized']['f1'] for e in experts]
    
    ax.bar(x - width/2, original_f1, width, label='Original Model', color='blue')
    ax.bar(x + width/2, optimized_f1, width, label='Optimized Model', color='red')
    
    ax.set_xlabel('Expert')
    ax.set_ylabel('F1 Score')
    ax.set_title('Expert F1 Score Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([e.capitalize() for e in experts])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(original_f1):
        ax.text(i - width/2, v + 0.01, f"{v:.2f}", ha='center')
    
    for i, v in enumerate(optimized_f1):
        ax.text(i + width/2, v + 0.01, f"{v:.2f}", ha='center')
    
    st.pyplot(fig)

def display_prediction_tool(data):
    st.title("Migraine Prediction Tool")
    st.write("Make predictions with custom input values and compare model outputs.")
    
    # Input sliders
    st.subheader("Input Parameters")
    
    # Sleep parameters
    st.write("Sleep Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        sleep_duration = st.slider("Sleep Duration (hours)", 4.0, 10.0, 7.0, 0.1)
        sleep_quality = st.slider("Sleep Quality (0-10)", 0, 10, 5, 1)
        sleep_interruptions = st.slider("Sleep Interruptions", 0, 5, 1, 1)
    
    with col2:
        rem_percentage = st.slider("REM Sleep (%)", 10, 30, 20, 1)
        deep_sleep_percentage = st.slider("Deep Sleep (%)", 10, 30, 20, 1)
        sleep_regularity = st.slider("Sleep Regularity (0-10)", 0, 10, 5, 1)
    
    # Weather parameters
    st.write("Weather Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        temperature = st.slider("Temperature (°C)", -10, 40, 20, 1)
        humidity = st.slider("Humidity (%)", 0, 100, 50, 1)
    
    with col2:
        pressure = st.slider("Barometric Pressure (hPa)", 980, 1040, 1013, 1)
        pressure_change = st.slider("Pressure Change (hPa/day)", -20, 20, 0, 1)
    
    # Stress/Diet parameters
    st.write("Stress and Diet Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        stress_level = st.slider("Stress Level (0-10)", 0, 10, 5, 1)
        water_intake = st.slider("Water Intake (liters)", 0.0, 4.0, 2.0, 0.1)
        caffeine_intake = st.slider("Caffeine Intake (mg)", 0, 500, 100, 10)
    
    with col2:
        alcohol_consumption = st.slider("Alcohol Consumption (units)", 0, 10, 0, 1)
        meal_regularity = st.slider("Meal Regularity (0-10)", 0, 10, 5, 1)
        processed_food = st.slider("Processed Food Consumption (0-10)", 0, 10, 5, 1)
    
    # Make prediction
    if st.button("Predict Migraine Risk"):
        # Normalize inputs to create feature vectors
        sleep_features = np.array([
            (sleep_duration - 7) / 3,  # Normalize to [-1, 1]
            (sleep_quality - 5) / 5,   # Normalize to [-1, 1]
            (sleep_interruptions - 2.5) / 2.5,  # Normalize to [-1, 1]
            (rem_percentage - 20) / 10,  # Normalize to [-1, 1]
            (deep_sleep_percentage - 20) / 10,  # Normalize to [-1, 1]
            (sleep_regularity - 5) / 5  # Normalize to [-1, 1]
        ])
        
        weather_features = np.array([
            (temperature - 15) / 25,  # Normalize to [-1, 1]
            (humidity - 50) / 50,     # Normalize to [-1, 1]
            (pressure - 1013) / 30,   # Normalize to [-1, 1]
            pressure_change / 20      # Normalize to [-1, 1]
        ])
        
        stress_diet_features = np.array([
            (stress_level - 5) / 5,    # Normalize to [-1, 1]
            (water_intake - 2) / 2,    # Normalize to [-1, 1]
            (caffeine_intake - 250) / 250,  # Normalize to [-1, 1]
            (alcohol_consumption - 5) / 5,  # Normalize to [-1, 1]
            (meal_regularity - 5) / 5,      # Normalize to [-1, 1]
            (processed_food - 5) / 5        # Normalize to [-1, 1]
        ])
        
        # Calculate risk scores (mock implementation)
        # In a real implementation, these would come from the actual models
        
        # Original model prediction
        # Simple weighted sum of features with some noise
        sleep_score_original = np.mean(np.abs(sleep_features)) * 0.35
        weather_score_original = np.mean(np.abs(weather_features)) * 0.25
        stress_diet_score_original = np.mean(np.abs(stress_diet_features)) * 0.40
        
        # Add some risk factors
        if sleep_duration < 6:
            sleep_score_original += 0.1
        if pressure_change > 10 or pressure_change < -10:
            weather_score_original += 0.15
        if stress_level > 7:
            stress_diet_score_original += 0.2
        if caffeine_intake > 300:
            stress_diet_score_original += 0.1
        
        # Combine scores with some randomness
        original_score = (sleep_score_original + weather_score_original + stress_diet_score_original) / 3
        original_score = min(max(original_score + np.random.normal(0, 0.05), 0), 1)
        
        # Optimized model prediction
        # More sophisticated combination with better weights
        sleep_score_optimized = np.mean(np.abs(sleep_features)) * 0.45
        weather_score_optimized = np.mean(np.abs(weather_features)) * 0.15
        stress_diet_score_optimized = np.mean(np.abs(stress_diet_features)) * 0.40
        
        # Add some risk factors with better thresholds
        if sleep_duration < 6.5:
            sleep_score_optimized += 0.15
        if sleep_quality < 4:
            sleep_score_optimized += 0.1
        if pressure_change > 8 or pressure_change < -8:
            weather_score_optimized += 0.2
        if stress_level > 6:
            stress_diet_score_optimized += 0.25
        if caffeine_intake > 250:
            stress_diet_score_optimized += 0.15
        if alcohol_consumption > 2:
            stress_diet_score_optimized += 0.1
        
        # Combine scores with less randomness
        optimized_score = (sleep_score_optimized + weather_score_optimized + stress_diet_score_optimized) / 3
        optimized_score = min(max(optimized_score + np.random.normal(0, 0.02), 0), 1)
        
        # Display prediction results
        st.subheader("Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Original Model")
            st.metric("Migraine Risk Score", f"{original_score:.2f}")
            
            # Risk level
            risk_level = "Low" if original_score < 0.3 else "Medium" if original_score < 0.6 else "High"
            st.write(f"Risk Level: **{risk_level}**")
            
            # Expert contributions
            st.write("Expert Contributions:")
            st.write(f"- Sleep: {sleep_score_original:.2f} ({sleep_score_original/original_score*100:.1f}%)")
            st.write(f"- Weather: {weather_score_original:.2f} ({weather_score_original/original_score*100:.1f}%)")
            st.write(f"- Stress/Diet: {stress_diet_score_original:.2f} ({stress_diet_score_original/original_score*100:.1f}%)")
        
        with col2:
            st.write("Optimized Model")
            st.metric("Migraine Risk Score", f"{optimized_score:.2f}", f"{optimized_score - original_score:+.2f}")
            
            # Risk level
            risk_level = "Low" if optimized_score < 0.3 else "Medium" if optimized_score < 0.6 else "High"
            st.write(f"Risk Level: **{risk_level}**")
            
            # Expert contributions
            st.write("Expert Contributions:")
            st.write(f"- Sleep: {sleep_score_optimized:.2f} ({sleep_score_optimized/optimized_score*100:.1f}%)")
            st.write(f"- Weather: {weather_score_optimized:.2f} ({weather_score_optimized/optimized_score*100:.1f}%)")
            st.write(f"- Stress/Diet: {stress_diet_score_optimized:.2f} ({stress_diet_score_optimized/optimized_score*100:.1f}%)")
        
        # Risk factors
        st.subheader("Identified Risk Factors")
        
        risk_factors = []
        
        if sleep_duration < 6.5:
            risk_factors.append("Low sleep duration")
        if sleep_quality < 4:
            risk_factors.append("Poor sleep quality")
        if sleep_interruptions > 3:
            risk_factors.append("Frequent sleep interruptions")
        if pressure_change > 8 or pressure_change < -8:
            risk_factors.append("Significant barometric pressure change")
        if stress_level > 6:
            risk_factors.append("High stress level")
        if caffeine_intake > 250:
            risk_factors.append("High caffeine intake")
        if alcohol_consumption > 2:
            risk_factors.append("Alcohol consumption")
        if processed_food > 7:
            risk_factors.append("High processed food consumption")
        if water_intake < 1.5:
            risk_factors.append("Low water intake")
        
        if risk_factors:
            for factor in risk_factors:
                st.write(f"- {factor}")
        else:
            st.write("No significant risk factors identified.")
        
        # Recommendations
        st.subheader("Recommendations")
        
        recommendations = []
        
        if sleep_duration < 6.5:
            recommendations.append("Increase sleep duration to at least 7 hours")
        if sleep_quality < 4:
            recommendations.append("Improve sleep quality by maintaining a regular sleep schedule")
        if sleep_interruptions > 3:
            recommendations.append("Reduce sleep interruptions by creating a quiet sleep environment")
        if pressure_change > 8 or pressure_change < -8:
            recommendations.append("Be aware of weather changes and prepare accordingly")
        if stress_level > 6:
            recommendations.append("Implement stress reduction techniques such as meditation or deep breathing")
        if caffeine_intake > 250:
            recommendations.append("Reduce caffeine intake to below 200mg per day")
        if alcohol_consumption > 2:
            recommendations.append("Limit alcohol consumption")
        if processed_food > 7:
            recommendations.append("Reduce consumption of processed foods")
        if water_intake < 1.5:
            recommendations.append("Increase water intake to at least 2 liters per day")
        
        if recommendations:
            for rec in recommendations:
                st.write(f"- {rec}")
        else:
            st.write("Continue with current lifestyle habits.")

def display_optimization_details(data):
    st.title("Optimization Details")
    st.write("Explore the details of the PyGMO optimization process and results.")
    
    # Optimization results
    optimization_results = data['optimization_results']
    
    # Phase 1: Expert Hyperparameter Optimization
    st.subheader("Phase 1: Expert Hyperparameter Optimization")
    
    # Sleep Expert
    st.write("Sleep Expert Optimization")
    
    sleep_config = optimization_results['expert_phase']['sleep']['config']
    sleep_fitness = optimization_results['expert_phase']['sleep']['fitness']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Optimized Configuration:")
        for param, value in sleep_config.items():
            st.write(f"- {param}: {value}")
    
    with col2:
        st.write("Performance:")
        st.metric("Validation AUC", f"{sleep_fitness:.4f}")
    
    # Weather Expert
    st.write("Weather Expert Optimization")
    
    weather_config = optimization_results['expert_phase']['weather']['config']
    weather_fitness = optimization_results['expert_phase']['weather']['fitness']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Optimized Configuration:")
        for param, value in weather_config.items():
            st.write(f"- {param}: {value}")
    
    with col2:
        st.write("Performance:")
        st.metric("Validation AUC", f"{weather_fitness:.4f}")
    
    # Stress/Diet Expert
    st.write("Stress/Diet Expert Optimization")
    
    stress_diet_config = optimization_results['expert_phase']['stress_diet']['config']
    stress_diet_fitness = optimization_results['expert_phase']['stress_diet']['fitness']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Optimized Configuration:")
        for param, value in stress_diet_config.items():
            st.write(f"- {param}: {value}")
    
    with col2:
        st.write("Performance:")
        st.metric("Validation AUC", f"{stress_diet_fitness:.4f}")
    
    # Phase 2: Gating Hyperparameter Optimization
    st.subheader("Phase 2: Gating Hyperparameter Optimization")
    
    gating_config = optimization_results['gating_phase']['config']
    gating_fitness = optimization_results['gating_phase']['fitness']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Optimized Configuration:")
        for param, value in gating_config.items():
            st.write(f"- {param}: {value}")
    
    with col2:
        st.write("Performance:")
        st.metric("Validation AUC", f"{gating_fitness:.4f}")
    
    # Phase 3: End-to-End MoE Optimization
    st.subheader("Phase 3: End-to-End MoE Optimization")
    
    e2e_config = optimization_results['e2e_phase']['config']
    e2e_fitness = optimization_results['e2e_phase']['fitness']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Optimized Configuration:")
        for param, value in e2e_config.items():
            st.write(f"- {param}: {value}")
    
    with col2:
        st.write("Performance:")
        st.metric("Validation AUC", f"{e2e_fitness['auc']:.4f}")
        st.metric("Inference Latency", f"{e2e_fitness['latency']:.2f} ms/sample")
    
    # Optimization Summary
    st.subheader("Optimization Summary")
    
    # Performance improvement
    improvement = optimization_results['final_performance']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("AUC Improvement", f"+{improvement['auc_improvement']:.4f}", f"{improvement['auc_improvement']*100:.1f}%")
    
    with col2:
        st.metric("F1 Improvement", f"+{improvement['f1_improvement']:.4f}", f"{improvement['f1_improvement']*100:.1f}%")
    
    with col3:
        st.metric("Inference Speedup", f"{improvement['inference_speedup']:.2f}x")
    
    # Optimization timeline
    st.subheader("Optimization Timeline")
    
    # Mock optimization timeline data
    timeline_data = {
        'phase': ['Expert Optimization', 'Gating Optimization', 'End-to-End Optimization'],
        'duration': [120, 45, 180],  # seconds
        'iterations': [50, 30, 100],
        'improvement': [0.08, 0.06, 0.08]
    }
    
    # Duration chart
    fig, ax = plt.figure(figsize=(10, 6)), plt.axes()
    
    ax.bar(timeline_data['phase'], timeline_data['duration'], color='blue')
    ax.set_xlabel('Optimization Phase')
    ax.set_ylabel('Duration (seconds)')
    ax.set_title('Optimization Phase Duration')
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(timeline_data['duration']):
        ax.text(i, v + 5, f"{v}s", ha='center')
    
    st.pyplot(fig)
    
    # Improvement chart
    fig, ax = plt.figure(figsize=(10, 6)), plt.axes()
    
    ax.bar(timeline_data['phase'], timeline_data['improvement'], color='green')
    ax.set_xlabel('Optimization Phase')
    ax.set_ylabel('AUC Improvement')
    ax.set_title('AUC Improvement by Optimization Phase')
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(timeline_data['improvement']):
        ax.text(i, v + 0.005, f"+{v:.2f}", ha='center')
    
    st.pyplot(fig)
    
    # Iterations chart
    fig, ax = plt.figure(figsize=(10, 6)), plt.axes()
    
    ax.bar(timeline_data['phase'], timeline_data['iterations'], color='purple')
    ax.set_xlabel('Optimization Phase')
    ax.set_ylabel('Number of Iterations')
    ax.set_title('Optimization Iterations by Phase')
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(timeline_data['iterations']):
        ax.text(i, v + 3, f"{v}", ha='center')
    
    st.pyplot(fig)
    
    # Optimization algorithms
    st.subheader("Optimization Algorithms")
    
    algorithms = {
        'Expert Optimization': 'Differential Evolution (DE)',
        'Gating Optimization': 'Particle Swarm Optimization (PSO)',
        'End-to-End Optimization': 'NSGA-II (Multi-objective)'
    }
    
    for phase, algorithm in algorithms.items():
        st.write(f"- **{phase}**: {algorithm}")
    
    # Hyperparameter search spaces
    st.subheader("Hyperparameter Search Spaces")
    
    # Sleep Expert
    st.write("Sleep Expert:")
    st.write("- conv_filters: [32, 64, 128]")
    st.write("- kernel_size: [3, 5, 7]")
    st.write("- lstm_units: [64, 128, 256]")
    st.write("- dropout_rate: [0.1, 0.5]")
    
    # Weather Expert
    st.write("Weather Expert:")
    st.write("- hidden_units: [64, 128, 256]")
    st.write("- activation: ['relu', 'tanh']")
    st.write("- dropout_rate: [0.1, 0.5]")
    
    # Stress/Diet Expert
    st.write("Stress/Diet Expert:")
    st.write("- embedding_dim: [32, 64, 128]")
    st.write("- num_heads: [2, 4, 8]")
    st.write("- transformer_dim: [32, 64, 128]")
    st.write("- dropout_rate: [0.1, 0.5]")
    
    # Gating Network
    st.write("Gating Network:")
    st.write("- gate_hidden_size: [64, 128, 256]")
    st.write("- gate_top_k: [1, 2, 3]")
    st.write("- load_balance_coef: [0.001, 0.01, 0.1]")
    
    # End-to-End
    st.write("End-to-End:")
    st.write("- learning_rate: [1e-4, 1e-2]")
    st.write("- batch_size: [16, 32, 64]")
    st.write("- dropout_rate: [0.1, 0.5]")

if __name__ == "__main__":
    main()
