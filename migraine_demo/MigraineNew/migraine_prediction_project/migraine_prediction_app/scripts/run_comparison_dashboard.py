"""
Run the Comparison Dashboard for Migraine Prediction Models

This script launches the Streamlit dashboard that compares the original FuseMoE model
with the PyGMO-optimized version, showing performance metrics, expert contributions,
and prediction capabilities.
"""

import os
import sys
import subprocess
import time

# Add the project root to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

def main():
    """Run the comparison dashboard."""
    print("\n=== Launching Migraine Prediction Model Comparison Dashboard ===")
    
    # Check if original model exists
    original_model_path = os.path.join(project_root, 'output', 'original_model.keras')
    if not os.path.exists(original_model_path):
        print(f"Warning: Original model not found at {original_model_path}")
        print("Creating a copy of the current model as the original model for comparison...")
        
        # Copy the current model as the original model
        current_model_path = os.path.join(project_root, 'output', 'optimized_model.keras')
        if os.path.exists(current_model_path):
            import shutil
            shutil.copy(current_model_path, original_model_path)
            print(f"Created original model at {original_model_path}")
        else:
            print(f"Error: No model found at {current_model_path}")
            print("Please run the model training first.")
            return
    
    # Check if optimized model exists
    optimized_model_path = os.path.join(project_root, 'output', 'optimized_model.keras')
    if not os.path.exists(optimized_model_path):
        print(f"Warning: Optimized model not found at {optimized_model_path}")
        print("The dashboard will show placeholder data for the optimized model.")
    
    # Launch the dashboard
    dashboard_path = os.path.join(project_root, 'dashboard', 'comparison_dashboard.py')
    
    print(f"Starting dashboard from {dashboard_path}")
    print("This may take a few moments...")
    
    # Run the dashboard
    try:
        # Use a different port to avoid conflicts with the main dashboard
        subprocess.run([
            "streamlit", "run", dashboard_path,
            "--server.port=8506",
            "--server.headless=true"
        ])
    except Exception as e:
        print(f"Error launching dashboard: {e}")
        print("Please make sure Streamlit is installed and try again.")

if __name__ == "__main__":
    main()
