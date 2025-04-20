import streamlit as st
import sys
import os

# Add the project root to the path
dashboard_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(dashboard_dir)
sys.path.append(project_root)

# Import the expert weights dashboard
from dashboard.expert_weights_dashboard import main as expert_weights_main
from dashboard.fixed_dashboard_final import main as main_dashboard

# Set page configuration
st.set_page_config(
    page_title="Migraine Prediction Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for navigation
st.sidebar.title("Dashboard Navigation")
dashboard_type = st.sidebar.radio(
    "Select Dashboard",
    ["Main Dashboard", "Expert Weights Dashboard"]
)

# Display selected dashboard
if dashboard_type == "Main Dashboard":
    main_dashboard()
else:
    expert_weights_main()
