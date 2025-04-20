import streamlit as st
import sys
import os

# Set page configuration - must be first Streamlit command
st.set_page_config(
    page_title="Migraine Prediction Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add the project root to the path
dashboard_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(dashboard_dir)
sys.path.append(project_root)

# Import dashboard components
from dashboard.fixed_dashboard_final import main as main_dashboard_function
from dashboard.expert_weights_dashboard import (
    expert_weights_visualization,
    interactive_network_diagram,
    decision_tree_visualization,
    expert_performance_comparison,
    load_optimization_summary
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

# Sidebar for navigation
st.sidebar.title("Dashboard Navigation")
dashboard_type = st.sidebar.radio(
    "Select Dashboard",
    ["Main Dashboard", "Expert Weights Dashboard"]
)

# Display PyGMO flow diagram in sidebar
st.sidebar.title("PyGMO Flow Diagram")

# Get the path to the flow diagram
flow_diagram_path = os.path.join(project_root, 'static', 'pygmo_flow_diagram.svg')

if os.path.exists(flow_diagram_path):
    with open(flow_diagram_path, "r") as f:
        svg_content = f.read()
    
    st.sidebar.markdown(f"""
    <div style="text-align: center;">
        {svg_content}
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.warning("Flow diagram not found.")

# Display selected dashboard
if dashboard_type == "Main Dashboard":
    main_dashboard_function()
else:
    # Expert Weights Dashboard
    st.markdown("<h1 class='main-header'>Expert Weights Visualization</h1>", unsafe_allow_html=True)
    
    # Create tabs for different visualizations
    tabs = st.tabs(["Expert Weights", "Network Diagram", "Decision Process", "Performance Comparison"])
    
    with tabs[0]:
        expert_weights_visualization()
    
    with tabs[1]:
        interactive_network_diagram()
    
    with tabs[2]:
        decision_tree_visualization()
    
    with tabs[3]:
        expert_performance_comparison()
