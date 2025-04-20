"""
Migraine Prediction Dashboard Launcher

This script launches the Streamlit dashboard for the migraine prediction system.
"""

import os
import sys
import subprocess

def main():
    """Launch the Streamlit dashboard."""
    # Add the project root to the Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    # Launch the dashboard
    dashboard_path = os.path.join(project_root, "dashboard", "main_dashboard.py")
    subprocess.run(["streamlit", "run", dashboard_path])

if __name__ == "__main__":
    main()
