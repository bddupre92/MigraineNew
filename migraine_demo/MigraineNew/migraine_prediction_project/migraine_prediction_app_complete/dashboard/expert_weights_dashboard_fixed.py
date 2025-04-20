import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_optimization_summary():
    """Load the optimization summary if available."""
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        summary_path = os.path.join(project_root, 'output', 'optimization', 'optimization_summary.json')
        
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            return summary
        else:
            st.warning(f"Optimization summary not found at {summary_path}")
            return None
    except Exception as e:
        st.error(f"Error loading optimization summary: {e}")
        return None

def expert_weights_visualization():
    """Create visualizations for expert weights from PyGMO optimization."""
    st.subheader("Expert Weights from PyGMO Optimization")
    
    # Load optimization summary
    optimization_summary = load_optimization_summary()
    
    if not optimization_summary:
        st.error("Optimization summary not available. Please run the optimization process first.")
        return
    
    # Extract expert weights from optimization summary
    expert_weights = optimization_summary.get('optimization_phases', {}).get('gating_phase', {}).get('expert_weights', {})
    
    if not expert_weights:
        st.error("Expert weights not found in optimization summary.")
        return
    
    # Create columns for different visualizations
    col1, col2 = st.columns(2)
    
    with col1:
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
    
    with col2:
        # Create a bar chart of expert weights
        fig = px.bar(
            x=list(expert_weights.keys()),
            y=list(expert_weights.values()),
            title="Expert Contribution Weights",
            color=list(expert_weights.values()),
            color_continuous_scale=px.colors.sequential.Purples
        )
        
        fig.update_layout(
            xaxis_title="Expert",
            yaxis_title="Weight",
            width=600,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Display expert weights in a table
    st.subheader("Expert Weights Table")
    
    # Create a dataframe for expert weights
    expert_weights_df = pd.DataFrame({
        'Expert': list(expert_weights.keys()),
        'Weight': list(expert_weights.values())
    })
    
    st.dataframe(expert_weights_df, hide_index=True)
    
    # Display optimization phases information
    st.subheader("Optimization Phases")
    
    # Extract optimization phases information
    optimization_phases = optimization_summary.get('optimization_phases', {})
    
    # Create tabs for each phase
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

def interactive_network_diagram():
    """Create an interactive network diagram of the expert model architecture."""
    st.subheader("Interactive Network Diagram")
    
    # Load optimization summary to get expert weights
    optimization_summary = load_optimization_summary()
    
    if not optimization_summary:
        st.error("Optimization summary not available. Please run the optimization process first.")
        return
    
    # Extract expert weights from optimization summary
    expert_weights = optimization_summary.get('optimization_phases', {}).get('gating_phase', {}).get('expert_weights', {})
    
    if not expert_weights:
        st.error("Expert weights not found in optimization summary.")
        return
    
    # Create a graph
    G = nx.DiGraph()
    
    # Add nodes
    G.add_node("Input", pos=(0, 0))
    
    # Add expert nodes
    expert_positions = {
        "Sleep Expert": (-1, -1),
        "Weather Expert": (-0.33, -1),
        "Stress/Diet Expert": (0.33, -1),
        "Physio Expert": (1, -1)
    }
    
    for expert, pos in expert_positions.items():
        G.add_node(expert, pos=pos)
    
    # Add gating network node
    G.add_node("Gating Network", pos=(0, -2))
    
    # Add fusion mechanism node
    G.add_node("Fusion Mechanism", pos=(0, -3))
    
    # Add prediction node
    G.add_node("Prediction", pos=(0, -4))
    
    # Add edges from input to experts
    for expert in expert_positions:
        G.add_edge("Input", expert)
    
    # Add edges from experts to gating network
    for expert in expert_positions:
        G.add_edge(expert, "Gating Network", weight=expert_weights.get(expert, 0.25))
    
    # Add edge from gating network to fusion mechanism
    G.add_edge("Gating Network", "Fusion Mechanism")
    
    # Add edge from fusion mechanism to prediction
    G.add_edge("Fusion Mechanism", "Prediction")
    
    # Get positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='skyblue', alpha=0.8, ax=ax)
    
    # Draw edges with varying width based on weight
    edge_widths = []
    for u, v, data in G.edges(data=True):
        if 'weight' in data:
            edge_widths.append(data['weight'] * 10)  # Scale weight for visibility
        else:
            edge_widths.append(1.0)
    
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7, edge_color='gray', arrows=True, arrowsize=20, ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)
    
    # Add edge labels for weights
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        if 'weight' in data:
            edge_labels[(u, v)] = f"{data['weight']:.2f}"
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, ax=ax)
    
    # Remove axis
    ax.axis('off')
    
    # Add title
    plt.title("Expert Model Architecture with Optimized Weights", fontsize=16)
    
    # Display the plot
    st.pyplot(fig)
    
    # Add explanation
    st.markdown("""
    ### Network Diagram Explanation
    
    This interactive diagram shows the architecture of the migraine prediction model with the optimized expert weights from PyGMO:
    
    1. **Input Data** is fed into each expert model
    2. **Expert Models** process their specific data type
    3. **Gating Network** assigns weights to each expert's prediction (shown on edges)
    4. **Fusion Mechanism** combines the weighted predictions
    5. **Prediction** outputs the final migraine likelihood
    
    The edge thickness and labels represent the weight assigned to each expert by the PyGMO optimization process.
    """)

def decision_tree_visualization():
    """Create a decision tree-like visualization showing how inputs affect expert selection."""
    st.subheader("Decision Process Visualization")
    
    # Load optimization summary to get expert weights
    optimization_summary = load_optimization_summary()
    
    if not optimization_summary:
        st.error("Optimization summary not available. Please run the optimization process first.")
        return
    
    # Extract expert weights from optimization summary
    expert_weights = optimization_summary.get('optimization_phases', {}).get('gating_phase', {}).get('expert_weights', {})
    
    if not expert_weights:
        st.error("Expert weights not found in optimization summary.")
        return
    
    # Create a Sankey diagram to visualize the decision process
    fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = ["Input Data", "Sleep Data", "Weather Data", "Stress/Diet Data", "Physio Data", 
                     "Sleep Expert", "Weather Expert", "Stress/Diet Expert", "Physio Expert", 
                     "Gating Network", "Prediction"],
            color = ["royalblue", "lightblue", "lightblue", "lightblue", "lightblue", 
                     "lightgreen", "lightgreen", "lightgreen", "lightgreen", 
                     "purple", "orange"]
        ),
        link = dict(
            source = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            target = [1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9, 10],
            value = [0.25, 0.25, 0.25, 0.25, 1, 1, 1, 1, 
                     expert_weights.get("Sleep Expert", 0.25),
                     expert_weights.get("Weather Expert", 0.25),
                     expert_weights.get("Stress/Diet Expert", 0.25),
                     expert_weights.get("Physio Expert", 0.25),
                     1],
            color = ["rgba(0,0,255,0.2)", "rgba(0,0,255,0.2)", "rgba(0,0,255,0.2)", "rgba(0,0,255,0.2)",
                     "rgba(0,255,0,0.2)", "rgba(0,255,0,0.2)", "rgba(0,255,0,0.2)", "rgba(0,255,0,0.2)",
                     "rgba(128,0,128,0.4)", "rgba(128,0,128,0.4)", "rgba(128,0,128,0.4)", "rgba(128,0,128,0.4)",
                     "rgba(255,165,0,0.4)"]
        )
    )])
    
    fig.update_layout(title_text="Decision Flow in Migraine Prediction Model", font_size=12)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add explanation
    st.markdown("""
    ### Decision Process Explanation
    
    This Sankey diagram visualizes how data flows through the migraine prediction model:
    
    1. **Input Data** is split into different data types
    2. Each data type is processed by its corresponding **Expert Model**
    3. The **Gating Network** assigns weights to each expert's prediction based on PyGMO optimization
    4. The weighted predictions are combined to produce the final **Prediction**
    
    The width of the flows represents the relative importance of each expert in the final prediction.
    """)

def expert_performance_comparison():
    """Create a comparison section for expert performance with and without optimization."""
    st.subheader("Expert Performance Comparison")
    
    # Load optimization summary
    optimization_summary = load_optimization_summary()
    
    if not optimization_summary:
        st.error("Optimization summary not available. Please run the optimization process first.")
        return
    
    # Extract expert phase information
    expert_phase = optimization_summary.get('optimization_phases', {}).get('expert_phase', {})
    
    if not expert_phase:
        st.error("Expert phase information not found in optimization summary.")
        return
    
    # Create a dataframe for expert performance comparison
    expert_data = []
    
    for expert_name, expert_info in expert_phase.items():
        convergence = expert_info.get('convergence', {})
        initial_fitness = convergence.get('initial_fitness', 0)
        final_fitness = convergence.get('final_fitness', 0)
        improvement = convergence.get('improvement', 0)
        improvement_percent = improvement / max(0.001, initial_fitness) * 100
        
        expert_data.append({
            'Expert': expert_name.capitalize(),
            'Initial Fitness': initial_fitness,
            'Final Fitness': final_fitness,
            'Improvement': improvement,
            'Improvement (%)': improvement_percent
        })
    
    expert_df = pd.DataFrame(expert_data)
    
    # Display the dataframe
    st.dataframe(expert_df, hide_index=True)
    
    # Create a bar chart for expert performance comparison
    fig = go.Figure()
    
    # Add initial fitness bars
    fig.add_trace(go.Bar(
        x=expert_df['Expert'],
        y=expert_df['Initial Fitness'],
        name='Initial Fitness (Baseline)',
        marker_color='lightblue'
    ))
    
    # Add final fitness bars
    fig.add_trace(go.Bar(
        x=expert_df['Expert'],
        y=expert_df['Final Fitness'],
        name='Final Fitness (Optimized)',
        marker_color='purple'
    ))
    
    # Update layout
    fig.update_layout(
        title='Expert Performance Before and After Optimization',
        xaxis_title='Expert',
        yaxis_title='Fitness (AUC)',
        barmode='group',
        legend=dict(x=0.01, y=0.99),
        width=700,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Create a line chart for improvement percentage
    fig = px.bar(
        expert_df,
        x='Expert',
        y='Improvement (%)',
        title='Expert Performance Improvement (%)',
        color='Improvement (%)',
        color_continuous_scale=px.colors.sequential.Viridis
    )
    
    fig.update_layout(
        xaxis_title='Expert',
        yaxis_title='Improvement (%)',
        width=700,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add explanation
    st.markdown("""
    ### Performance Comparison Explanation
    
    This section compares the performance of each expert model before and after PyGMO optimization:
    
    1. **Initial Fitness** represents the baseline performance of each expert model
    2. **Final Fitness** shows the performance after PyGMO optimization
    3. **Improvement** quantifies the absolute and percentage improvement achieved through optimization
    
    The charts visualize how much each expert model benefited from the optimization process.
    """)
