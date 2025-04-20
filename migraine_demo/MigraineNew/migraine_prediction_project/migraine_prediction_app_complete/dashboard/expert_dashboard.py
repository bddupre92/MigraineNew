import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
import os
import sys
import json

# Add parent directory to path to import from model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import model components (these may not be available in simplified mode)
try:
    from model.moe_architecture.gating_network import GatingNetwork
    from model.pygmo_optimizer import PygmoOptimizer
    MOE_COMPONENTS_AVAILABLE = True
except ImportError:
    MOE_COMPONENTS_AVAILABLE = False

def create_expert_dashboard():
    """Create the expert dashboard component."""
    st.markdown("<h2 class='sub-header'>Expert Model Analysis</h2>", unsafe_allow_html=True)
    
    # Check if MoE components are available
    if not MOE_COMPONENTS_AVAILABLE:
        st.warning("""
        The Mixture of Experts (MoE) components are not available in this simplified dashboard.
        This page shows simulated expert analysis for demonstration purposes.
        """)
    
    # Create tabs for different visualizations
    tabs = st.tabs([
        "Expert Contributions", 
        "Feature Importance", 
        "Expert Activation", 
        "Gating Network",
        "Optimization Comparison"
    ])
    
    # Expert Contributions tab
    with tabs[0]:
        st.markdown("<h3>Expert Model Contributions</h3>", unsafe_allow_html=True)
        st.markdown("""
        This visualization shows how each expert model contributes to the final predictions.
        The contribution is measured as the weighted importance of each expert's output in the final prediction.
        """)
        
        # Create expert contribution visualization
        expert_contributions = create_expert_contribution_viz()
        st.plotly_chart(expert_contributions, use_container_width=True)
        
        # Add patient profile analysis
        st.markdown("<h4>Contributions by Patient Profile</h4>", unsafe_allow_html=True)
        st.markdown("""
        Different types of patients may trigger different expert models. 
        This visualization shows how expert contributions vary across different patient profiles.
        """)
        
        profile_contributions = create_profile_contribution_viz()
        st.plotly_chart(profile_contributions, use_container_width=True)
    
    # Feature Importance tab
    with tabs[1]:
        st.markdown("<h3>Feature Importance by Expert</h3>", unsafe_allow_html=True)
        st.markdown("""
        This visualization shows which features each expert model relies on most heavily.
        The importance is calculated using the weights or feature importance scores from each expert model.
        """)
        
        # Create feature importance visualization
        feature_importance = create_feature_importance_viz()
        st.plotly_chart(feature_importance, use_container_width=True)
        
        # Add interactive feature selector
        st.markdown("<h4>Compare Feature Importance Across Experts</h4>", unsafe_allow_html=True)
        
        # Get all features
        all_features = get_all_features()
        
        # Create multiselect for features
        selected_features = st.multiselect(
            "Select features to compare:",
            options=all_features,
            default=all_features[:5]
        )
        
        if selected_features:
            # Create comparison visualization
            feature_comparison = create_feature_comparison_viz(selected_features)
            st.plotly_chart(feature_comparison, use_container_width=True)
    
    # Expert Activation tab
    with tabs[2]:
        st.markdown("<h3>Expert Activation Patterns</h3>", unsafe_allow_html=True)
        st.markdown("""
        This visualization shows when different experts are activated based on input data characteristics.
        The activation is determined by the gating network, which assigns weights to each expert model.
        """)
        
        # Create expert activation visualization
        expert_activation = create_expert_activation_viz()
        st.plotly_chart(expert_activation, use_container_width=True)
        
        # Add time-series visualization
        st.markdown("<h4>Expert Activation Over Time</h4>", unsafe_allow_html=True)
        st.markdown("""
        This visualization shows how expert activations change over time for a simulated patient.
        The time series represents a week of data, showing how different factors trigger different experts.
        """)
        
        time_series_activation = create_time_series_activation_viz()
        st.plotly_chart(time_series_activation, use_container_width=True)
    
    # Gating Network tab
    with tabs[3]:
        st.markdown("<h3>Gating Network Visualization</h3>", unsafe_allow_html=True)
        st.markdown("""
        The gating network determines how to combine the outputs of the expert models.
        These visualizations show different ways to understand how the gating network operates.
        """)
        
        # Simple weight distribution
        st.markdown("<h4>Expert Weight Distribution</h4>", unsafe_allow_html=True)
        st.markdown("""
        This simple visualization shows the average weight distribution across all experts.
        Higher weights indicate experts that have more influence on the final predictions.
        """)
        
        weight_distribution = create_weight_distribution_viz()
        st.plotly_chart(weight_distribution, use_container_width=True)
        
        # Interactive network diagram
        st.markdown("<h4>Interactive Network Diagram</h4>", unsafe_allow_html=True)
        st.markdown("""
        This interactive network diagram shows the connections between input features, 
        expert models, and the final prediction. The thickness of the lines represents 
        the strength of the connections.
        """)
        
        # Create network diagram
        network_diagram = create_network_diagram_viz()
        st.pyplot(network_diagram)
        
        # Decision tree visualization
        st.markdown("<h4>Decision Tree Visualization</h4>", unsafe_allow_html=True)
        st.markdown("""
        This decision tree visualization shows how the gating network decides which experts 
        to activate based on the input features. Each path through the tree represents a 
        different decision process.
        """)
        
        # Create decision tree visualization
        decision_tree = create_decision_tree_viz()
        st.pyplot(decision_tree)
    
    # Optimization Comparison tab
    with tabs[4]:
        st.markdown("<h3>PyGMO Optimization Comparison</h3>", unsafe_allow_html=True)
        st.markdown("""
        This comparison shows the performance difference between experts with PyGMO optimization 
        and hyperparameter tuning versus experts without optimization (baseline models).
        """)
        
        # Create optimization comparison visualization
        optimization_comparison = create_optimization_comparison_viz()
        st.plotly_chart(optimization_comparison, use_container_width=True)
        
        # Add detailed metrics comparison
        st.markdown("<h4>Detailed Performance Metrics</h4>", unsafe_allow_html=True)
        
        metrics_comparison = create_metrics_comparison_viz()
        st.plotly_chart(metrics_comparison, use_container_width=True)
        
        # Add convergence visualization
        st.markdown("<h4>Optimization Convergence</h4>", unsafe_allow_html=True)
        st.markdown("""
        This visualization shows how the PyGMO optimization algorithm converged over iterations.
        Faster convergence indicates more efficient optimization.
        """)
        
        convergence_viz = create_convergence_viz()
        st.plotly_chart(convergence_viz, use_container_width=True)

# Helper functions to create visualizations

def create_expert_contribution_viz():
    """Create visualization of expert model contributions."""
    # Simulated data for expert contributions
    experts = ['Sleep Expert', 'Weather Expert', 'Stress/Diet Expert', 'Physiological Expert']
    contributions = [0.35, 0.15, 0.30, 0.20]
    
    # Create figure
    fig = go.Figure()
    
    # Add bar chart
    fig.add_trace(go.Bar(
        x=experts,
        y=contributions,
        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
        text=[f"{c:.1%}" for c in contributions],
        textposition='auto'
    ))
    
    # Update layout
    fig.update_layout(
        title='Overall Expert Model Contributions',
        xaxis_title='Expert Model',
        yaxis_title='Contribution Weight',
        yaxis=dict(tickformat='.0%'),
        height=500
    )
    
    return fig

def create_profile_contribution_viz():
    """Create visualization of expert contributions by patient profile."""
    # Simulated data for different patient profiles
    profiles = ['High Stress', 'Weather Sensitive', 'Sleep Disrupted', 'Hormonal', 'Average']
    
    # Expert contributions for each profile
    sleep_contrib = [0.20, 0.15, 0.60, 0.25, 0.35]
    weather_contrib = [0.10, 0.55, 0.05, 0.15, 0.15]
    stress_contrib = [0.50, 0.10, 0.20, 0.20, 0.30]
    physio_contrib = [0.20, 0.20, 0.15, 0.40, 0.20]
    
    # Create figure
    fig = go.Figure()
    
    # Add stacked bar chart
    fig.add_trace(go.Bar(
        x=profiles,
        y=sleep_contrib,
        name='Sleep Expert',
        marker_color='#1f77b4'
    ))
    
    fig.add_trace(go.Bar(
        x=profiles,
        y=weather_contrib,
        name='Weather Expert',
        marker_color='#ff7f0e'
    ))
    
    fig.add_trace(go.Bar(
        x=profiles,
        y=stress_contrib,
        name='Stress/Diet Expert',
        marker_color='#2ca02c'
    ))
    
    fig.add_trace(go.Bar(
        x=profiles,
        y=physio_contrib,
        name='Physiological Expert',
        marker_color='#d62728'
    ))
    
    # Update layout
    fig.update_layout(
        title='Expert Contributions by Patient Profile',
        xaxis_title='Patient Profile',
        yaxis_title='Contribution Weight',
        yaxis=dict(tickformat='.0%'),
        barmode='stack',
        height=500
    )
    
    return fig

def create_feature_importance_viz():
    """Create visualization of feature importance by expert."""
    # Simulated data for feature importance
    sleep_features = ['Sleep Duration', 'Sleep Quality', 'Deep Sleep %', 'REM Sleep %', 'Interruptions']
    sleep_importance = [0.30, 0.25, 0.20, 0.15, 0.10]
    
    weather_features = ['Temperature', 'Pressure Change', 'Humidity', 'Weather Change']
    weather_importance = [0.40, 0.30, 0.20, 0.10]
    
    stress_features = ['Stress Level', 'Hydration', 'Caffeine', 'Alcohol', 'Diet Quality']
    stress_importance = [0.35, 0.25, 0.20, 0.15, 0.05]
    
    physio_features = ['Heart Rate Var.', 'Blood Pressure', 'Hormone Levels', 'Exercise', 'Previous Migraine']
    physio_importance = [0.25, 0.20, 0.30, 0.15, 0.10]
    
    # Create subplots
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        'Sleep Expert', 'Weather Expert', 'Stress/Diet Expert', 'Physiological Expert'
    ))
    
    # Add bar charts
    fig.add_trace(go.Bar(
        x=sleep_features,
        y=sleep_importance,
        marker_color='#1f77b4'
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=weather_features,
        y=weather_importance,
        marker_color='#ff7f0e'
    ), row=1, col=2)
    
    fig.add_trace(go.Bar(
        x=stress_features,
        y=stress_importance,
        marker_color='#2ca02c'
    ), row=2, col=1)
    
    fig.add_trace(go.Bar(
        x=physio_features,
        y=physio_importance,
        marker_color='#d62728'
    ), row=2, col=2)
    
    # Update layout
    fig.update_layout(
        title='Feature Importance by Expert Model',
        height=700,
        showlegend=False
    )
    
    # Update y-axis
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_yaxes(title_text='Importance', row=i, col=j)
    
    return fig

def get_all_features():
    """Get all features from all experts."""
    # Simulated list of all features
    features = [
        # Sleep features
        'Sleep Duration', 'Sleep Quality', 'Deep Sleep %', 'REM Sleep %', 'Interruptions',
        'Sleep Efficiency', 'Time to Fall Asleep',
        
        # Weather features
        'Temperature', 'Pressure Change', 'Humidity', 'Weather Change', 'Sunlight Hours',
        'Wind Speed', 'Air Quality',
        
        # Stress/Diet features
        'Stress Level', 'Hydration', 'Caffeine', 'Alcohol', 'Diet Quality',
        'Sugar Intake', 'Meal Regularity', 'Protein Intake',
        
        # Physiological features
        'Heart Rate Var.', 'Blood Pressure', 'Hormone Levels', 'Exercise', 'Previous Migraine',
        'Fatigue Level', 'Body Temperature', 'Medication Use'
    ]
    
    return features

def create_feature_comparison_viz(selected_features):
    """Create comparison of feature importance across experts."""
    # Simulated data for feature importance across experts
    # Each expert has a different importance for each feature
    experts = ['Sleep Expert', 'Weather Expert', 'Stress/Diet Expert', 'Physiological Expert']
    
    # Generate random importance values for demonstration
    np.random.seed(42)  # For reproducibility
    
    # Create figure
    fig = go.Figure()
    
    # Add a trace for each expert
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, expert in enumerate(experts):
        # Generate importance values between 0 and 1
        # Weighted based on expert's domain
        importances = []
        for feature in selected_features:
            if 'Sleep' in feature and expert == 'Sleep Expert':
                importance = np.random.uniform(0.5, 1.0)
            elif 'Weather' in feature and expert == 'Weather Expert':
                importance = np.random.uniform(0.5, 1.0)
            elif ('Stress' in feature or 'Diet' in feature or 'Caffeine' in feature or 'Alcohol' in feature) and expert == 'Stress/Diet Expert':
                importance = np.random.uniform(0.5, 1.0)
            elif ('Heart' in feature or 'Blood' in feature or 'Hormone' in feature or 'Exercise' in feature) and expert == 'Physiological Expert':
                importance = np.random.uniform(0.5, 1.0)
            else:
                importance = np.random.uniform(0.0, 0.5)
            
            importances.append(importance)
        
        fig.add_trace(go.Bar(
            x=selected_features,
            y=importances,
            name=expert,
            marker_color=colors[i]
        ))
    
    # Update layout
    fig.update_layout(
        title='Feature Importance Comparison Across Experts',
        xaxis_title='Feature',
        yaxis_title='Importance',
        barmode='group',
        height=500
    )
    
    return fig

def create_expert_activation_viz():
    """Create visualization of expert activation patterns."""
    # Simulated data for expert activation
    # Each row represents a sample, each column represents an expert
    np.random.seed(42)  # For reproducibility
    
    # Generate 100 samples with 4 experts
    n_samples = 100
    n_experts = 4
    
    # Generate random activations (weights) for each expert
    # Sum of weights for each sample should be 1
    activations = np.random.rand(n_samples, n_experts)
    activations = activations / activations.sum(axis=1, keepdims=True)
    
    # Generate some features that correlate with activations
    # For demonstration, we'll use 2 features: stress level and weather change
    stress_level = np.random.uniform(0, 10, n_samples)
    weather_change = np.random.uniform(0, 10, n_samples)
    
    # Create a scatter plot with color representing the dominant expert
    dominant_expert = np.argmax(activations, axis=1)
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter plot
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    expert_names = ['Sleep Expert', 'Weather Expert', 'Stress/Diet Expert', 'Physiological Expert']
    
    for i in range(n_experts):
        # Get samples where this expert is dominant
        mask = dominant_expert == i
        
        fig.add_trace(go.Scatter(
            x=stress_level[mask],
            y=weather_change[mask],
            mode='markers',
            name=expert_names[i],
            marker=dict(
                color=colors[i],
                size=10,
                opacity=0.7
            )
        ))
    
    # Update layout
    fig.update_layout(
        title='Expert Activation Patterns',
        xaxis_title='Stress Level (0-10)',
        yaxis_title='Weather Change (0-10)',
        height=600
    )
    
    return fig

def create_time_series_activation_viz():
    """Create visualization of expert activation over time."""
    # Simulated time series data for a week
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Expert activations over time
    sleep_activation = [0.4, 0.3, 0.2, 0.1, 0.2, 0.5, 0.6]
    weather_activation = [0.2, 0.4, 0.5, 0.3, 0.1, 0.1, 0.1]
    stress_activation = [0.3, 0.2, 0.2, 0.5, 0.6, 0.3, 0.2]
    physio_activation = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    
    # Create figure
    fig = go.Figure()
    
    # Add lines for each expert
    fig.add_trace(go.Scatter(
        x=days,
        y=sleep_activation,
        mode='lines+markers',
        name='Sleep Expert',
        line=dict(color='#1f77b4', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=days,
        y=weather_activation,
        mode='lines+markers',
        name='Weather Expert',
        line=dict(color='#ff7f0e', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=days,
        y=stress_activation,
        mode='lines+markers',
        name='Stress/Diet Expert',
        line=dict(color='#2ca02c', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=days,
        y=physio_activation,
        mode='lines+markers',
        name='Physiological Expert',
        line=dict(color='#d62728', width=3)
    ))
    
    # Add events annotation
    annotations = [
        dict(x='Tuesday', y=0.4, text='Weather Change', showarrow=True, arrowhead=1),
        dict(x='Thursday', y=0.5, text='High Stress', showarrow=True, arrowhead=1),
        dict(x='Saturday', y=0.5, text='Good Sleep', showarrow=True, arrowhead=1)
    ]
    
    # Update layout
    fig.update_layout(
        title='Expert Activation Over Time',
        xaxis_title='Day of Week',
        yaxis_title='Activation Weight',
        height=500,
        annotations=annotations
    )
    
    return fig

def create_weight_distribution_viz():
    """Create visualization of expert weight distribution."""
    # Simulated data for expert weights
    experts = ['Sleep Expert', 'Weather Expert', 'Stress/Diet Expert', 'Physiological Expert']
    weights = [0.35, 0.15, 0.30, 0.20]
    
    # Create figure
    fig = go.Figure()
    
    # Add pie chart
    fig.add_trace(go.Pie(
        labels=experts,
        values=weights,
        textinfo='label+percent',
        marker=dict(colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ))
    
    # Update layout
    fig.update_layout(
        title='Expert Weight Distribution',
        height=500
    )
    
    return fig

def create_network_diagram_viz():
    """Create network diagram visualization of the gating network."""
    # Create a network graph
    G = nx.DiGraph()
    
    # Add nodes
    # Input features
    input_features = [
        'Sleep Quality', 'Weather Change', 'Stress Level', 
        'Hydration', 'Previous Migraine'
    ]
    
    # Expert models
    experts = [
        'Sleep Expert', 'Weather Expert', 
        'Stress/Diet Expert', 'Physiological Expert'
    ]
    
    # Add input feature nodes
    for i, feature in enumerate(input_features):
        G.add_node(feature, pos=(0, i), node_type='input')
    
    # Add expert nodes
    for i, expert in enumerate(experts):
        G.add_node(expert, pos=(1, i), node_type='expert')
    
    # Add output node
    G.add_node('Prediction', pos=(2, 2), node_type='output')
    
    # Add edges from input features to experts
    # The weight represents the importance of the feature for the expert
    edges = [
        ('Sleep Quality', 'Sleep Expert', 0.8),
        ('Sleep Quality', 'Physiological Expert', 0.3),
        ('Weather Change', 'Weather Expert', 0.9),
        ('Weather Change', 'Stress/Diet Expert', 0.2),
        ('Stress Level', 'Stress/Diet Expert', 0.7),
        ('Stress Level', 'Sleep Expert', 0.4),
        ('Hydration', 'Physiological Expert', 0.6),
        ('Hydration', 'Stress/Diet Expert', 0.5),
        ('Previous Migraine', 'Physiological Expert', 0.8),
        ('Previous Migraine', 'Sleep Expert', 0.3)
    ]
    
    for src, dst, weight in edges:
        G.add_edge(src, dst, weight=weight)
    
    # Add edges from experts to output
    expert_weights = [0.35, 0.15, 0.30, 0.20]
    for expert, weight in zip(experts, expert_weights):
        G.add_edge(expert, 'Prediction', weight=weight)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw nodes
    node_colors = []
    for node in G.nodes():
        if G.nodes[node]['node_type'] == 'input':
            node_colors.append('#1f77b4')  # Blue for input
        elif G.nodes[node]['node_type'] == 'expert':
            node_colors.append('#ff7f0e')  # Orange for experts
        else:
            node_colors.append('#2ca02c')  # Green for output
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.8, ax=ax)
    
    # Draw edges
    edge_widths = [G[u][v]['weight'] * 5 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7, edge_color='gray', 
                          connectionstyle='arc3,rad=0.1', ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif', ax=ax)
    
    # Add edge weight labels
    edge_labels = {(u, v): f"{G[u][v]['weight']:.1f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)
    
    # Remove axis
    ax.axis('off')
    
    # Add title
    plt.title('Gating Network Diagram', fontsize=16)
    
    return fig

def create_decision_tree_viz():
    """Create decision tree visualization of the gating network."""
    # Create a simple decision tree to simulate the gating network
    # This would be replaced with the actual gating network in a real implementation
    
    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    # Features: stress level, sleep quality, weather change, previous migraine
    X = np.random.rand(n_samples, 4)
    
    # Target: dominant expert (0: Sleep, 1: Weather, 2: Stress/Diet, 3: Physiological)
    # Create a rule-based target for demonstration
    y = np.zeros(n_samples, dtype=int)
    
    # Rules:
    # If sleep quality is low (< 0.3), sleep expert dominates
    # If weather change is high (> 0.7), weather expert dominates
    # If stress level is high (> 0.7), stress/diet expert dominates
    # If previous migraine is high (> 0.7), physiological expert dominates
    # Otherwise, weighted combination based on feature values
    
    for i in range(n_samples):
        stress_level, sleep_quality, weather_change, prev_migraine = X[i]
        
        if sleep_quality < 0.3:
            y[i] = 0  # Sleep expert
        elif weather_change > 0.7:
            y[i] = 1  # Weather expert
        elif stress_level > 0.7:
            y[i] = 2  # Stress/Diet expert
        elif prev_migraine > 0.7:
            y[i] = 3  # Physiological expert
        else:
            # Weighted combination
            weights = [
                (1 - sleep_quality),  # Sleep expert weight
                weather_change,       # Weather expert weight
                stress_level,         # Stress/Diet expert weight
                prev_migraine         # Physiological expert weight
            ]
            y[i] = np.argmax(weights)
    
    # Train a decision tree
    feature_names = ['Stress Level', 'Sleep Quality', 'Weather Change', 'Previous Migraine']
    class_names = ['Sleep Expert', 'Weather Expert', 'Stress/Diet Expert', 'Physiological Expert']
    
    # Create a decision tree with limited depth for visualization
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Plot the decision tree
    plot_tree(clf, filled=True, feature_names=feature_names, class_names=class_names, 
             rounded=True, ax=ax, fontsize=10)
    
    # Add title
    plt.title('Gating Network Decision Tree', fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def create_optimization_comparison_viz():
    """Create visualization comparing optimized vs non-optimized experts."""
    # Simulated performance metrics for optimized and non-optimized models
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    
    # Performance values
    optimized = [0.96, 0.79, 0.83, 0.80, 0.99]
    non_optimized = [0.85, 0.62, 0.68, 0.65, 0.82]
    
    # Improvement percentage
    improvement = [(opt - non_opt) / non_opt * 100 for opt, non_opt in zip(optimized, non_optimized)]
    
    # Create figure
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        'Performance Metrics Comparison', 'Percentage Improvement'
    ), specs=[[{"type": "bar"}, {"type": "bar"}]])
    
    # Add bar chart for performance comparison
    fig.add_trace(go.Bar(
        x=metrics,
        y=optimized,
        name='With PyGMO Optimization',
        marker_color='#1f77b4'
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=metrics,
        y=non_optimized,
        name='Without Optimization',
        marker_color='#ff7f0e'
    ), row=1, col=1)
    
    # Add bar chart for improvement percentage
    fig.add_trace(go.Bar(
        x=metrics,
        y=improvement,
        name='Improvement',
        marker_color='#2ca02c',
        text=[f"+{imp:.1f}%" for imp in improvement],
        textposition='auto'
    ), row=1, col=2)
    
    # Update layout
    fig.update_layout(
        title='PyGMO Optimization Performance Comparison',
        height=500
    )
    
    # Update y-axis
    fig.update_yaxes(title_text='Performance', row=1, col=1)
    fig.update_yaxes(title_text='Improvement (%)', row=1, col=2)
    
    return fig

def create_metrics_comparison_viz():
    """Create detailed metrics comparison visualization."""
    # Simulated detailed metrics for each expert
    experts = ['Sleep Expert', 'Weather Expert', 'Stress/Diet Expert', 'Physiological Expert', 'Ensemble']
    
    # Accuracy for optimized and non-optimized models
    accuracy_opt = [0.92, 0.88, 0.90, 0.89, 0.96]
    accuracy_non_opt = [0.80, 0.75, 0.82, 0.78, 0.85]
    
    # F1 Score for optimized and non-optimized models
    f1_opt = [0.76, 0.70, 0.74, 0.72, 0.80]
    f1_non_opt = [0.60, 0.55, 0.62, 0.58, 0.65]
    
    # AUC for optimized and non-optimized models
    auc_opt = [0.94, 0.90, 0.92, 0.91, 0.99]
    auc_non_opt = [0.78, 0.75, 0.80, 0.76, 0.82]
    
    # Create figure
    fig = make_subplots(rows=3, cols=1, subplot_titles=(
        'Accuracy Comparison', 'F1 Score Comparison', 'AUC Comparison'
    ))
    
    # Add bar charts
    fig.add_trace(go.Bar(
        x=experts,
        y=accuracy_opt,
        name='With PyGMO Optimization',
        marker_color='#1f77b4'
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=experts,
        y=accuracy_non_opt,
        name='Without Optimization',
        marker_color='#ff7f0e'
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=experts,
        y=f1_opt,
        name='With PyGMO Optimization',
        marker_color='#1f77b4',
        showlegend=False
    ), row=2, col=1)
    
    fig.add_trace(go.Bar(
        x=experts,
        y=f1_non_opt,
        name='Without Optimization',
        marker_color='#ff7f0e',
        showlegend=False
    ), row=2, col=1)
    
    fig.add_trace(go.Bar(
        x=experts,
        y=auc_opt,
        name='With PyGMO Optimization',
        marker_color='#1f77b4',
        showlegend=False
    ), row=3, col=1)
    
    fig.add_trace(go.Bar(
        x=experts,
        y=auc_non_opt,
        name='Without Optimization',
        marker_color='#ff7f0e',
        showlegend=False
    ), row=3, col=1)
    
    # Update layout
    fig.update_layout(
        title='Detailed Performance Metrics by Expert',
        height=800,
        barmode='group'
    )
    
    # Update y-axes
    fig.update_yaxes(title_text='Accuracy', row=1, col=1)
    fig.update_yaxes(title_text='F1 Score', row=2, col=1)
    fig.update_yaxes(title_text='AUC', row=3, col=1)
    
    return fig

def create_convergence_viz():
    """Create optimization convergence visualization."""
    # Simulated convergence data
    iterations = list(range(1, 51))
    
    # Fitness values (lower is better)
    # Exponential decay with some noise
    np.random.seed(42)
    fitness = 1.0 - 0.9 * (1 - np.exp(-0.1 * np.array(iterations)))
    fitness = fitness + np.random.normal(0, 0.02, len(iterations))
    fitness = np.clip(fitness, 0, 1)
    
    # Create figure
    fig = go.Figure()
    
    # Add line chart
    fig.add_trace(go.Scatter(
        x=iterations,
        y=fitness,
        mode='lines+markers',
        name='Fitness Value',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Add annotations for key points
    annotations = [
        dict(x=10, y=fitness[9], text='Early Convergence', showarrow=True, arrowhead=1),
        dict(x=30, y=fitness[29], text='Fine-tuning', showarrow=True, arrowhead=1),
        dict(x=50, y=fitness[49], text='Final Solution', showarrow=True, arrowhead=1)
    ]
    
    # Update layout
    fig.update_layout(
        title='PyGMO Optimization Convergence',
        xaxis_title='Iteration',
        yaxis_title='Fitness Value (lower is better)',
        height=500,
        annotations=annotations
    )
    
    return fig
