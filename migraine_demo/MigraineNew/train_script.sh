#!/bin/sh
set -e

echo 'Clearing *.pyc files...'
find /app/migraine_prediction_project/migraine_prediction_app_complete/model -name '*.pyc' -print -delete
find /app/migraine_prediction_project/migraine_prediction_app_complete/model/moe_architecture -name '*.pyc' -print -delete

echo '--- Verifying code in migraine_prediction_model.py (lines 120-150) ---'
sed -n '120,150p' /app/migraine_prediction_project/migraine_prediction_app_complete/model/migraine_prediction_model.py
echo '--- End of code verification ---'

echo 'Executing Python training script...'
python3 /app/run_training.py # Execute the dedicated Python script

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "Python training script failed with exit code $EXIT_CODE"
  exit $EXIT_CODE
fi

echo 'Training script finished.'
exit 0 