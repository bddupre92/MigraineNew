#!/bin/bash
set -e

if [ "$1" = "jupyter" ]; then
  echo "Starting Jupyter Notebook..."
  # Activate conda environment
  source /opt/conda/etc/profile.d/conda.sh
  conda activate migraine_env
  # Run Jupyter
  exec jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=""
else
  echo "Starting Streamlit dashboard..."
  # Activate conda environment
  source /opt/conda/etc/profile.d/conda.sh
  conda activate migraine_env
  cd migraine_prediction_project/migraine_prediction_app_complete/dashboard
  exec streamlit run streamlit_dashboard.py --server.port 8501 --server.address 0.0.0.0
fi 