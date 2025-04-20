FROM python:3.10-slim-bullseye

# Set the correct PATH, prioritizing system paths over potential Conda paths
ENV PATH="/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyGMO
RUN pip install --no-cache-dir pygmo

# Clone FuseMoE repository
# RUN git clone https://github.com/aaronhan223/FuseMoE.git
# WORKDIR /app/FuseMoE
# RUN pip install -e .
# WORKDIR /app

# Install Jupyter and Streamlit
RUN pip install --no-cache-dir jupyter streamlit scipy # Explicitly install scipy

# Copy application code
COPY . /app

# Make port 8501 available for Streamlit
EXPOSE 8501
# Make port 8888 available for Jupyter
EXPOSE 8888

# Create a script to run either Streamlit or Jupyter with corrected path
RUN echo '#!/bin/bash\nset -e\nif [ "$1" = "jupyter" ]; then\n  echo "Starting Jupyter Notebook..."\n  exec jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=""\nelse\n  echo "Starting Streamlit dashboard..."\n  cd /app/migraine_prediction_project/migraine_prediction_app_complete/dashboard\n  exec streamlit run streamlit_dashboard.py --server.port 8501 --server.address 0.0.0.0\nfi' > /app/run.sh
RUN chmod +x /app/run.sh

# Set the entrypoint
ENTRYPOINT ["/app/run.sh"]
