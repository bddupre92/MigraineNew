import unittest
import pandas as pd
import numpy as np
import tensorflow as tf
import tempfile
import shutil
import os
import sys
from unittest.mock import patch, MagicMock

# Mock the imports that are causing issues
sys.modules['experts.sleep_expert'] = MagicMock()
sys.modules['experts.weather_expert'] = MagicMock()
sys.modules['experts.stress_diet_expert'] = MagicMock()
sys.modules['migraine_prediction_model'] = MagicMock()
sys.modules['optimized_model'] = MagicMock()
sys.modules['performance_metrics'] = MagicMock()

# Now we can import the dashboard module
import streamlit_dashboard

class TestDashboardFunctions(unittest.TestCase):
    """Test cases for the Streamlit dashboard functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests."""
        # Create temporary directories for test data and output
        cls.temp_dir = tempfile.mkdtemp()
        cls.data_dir = os.path.join(cls.temp_dir, 'data')
        cls.output_dir = os.path.join(cls.temp_dir, 'output')
        
        os.makedirs(cls.data_dir, exist_ok=True)
        os.makedirs(cls.output_dir, exist_ok=True)
        
        # Create synthetic test data
        cls._create_test_data()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Remove temporary directory
        shutil.rmtree(cls.temp_dir)
    
    @classmethod
    def _create_test_data(cls):
        """Create synthetic test data for dashboard testing."""
        # Create patient IDs
        patient_ids = np.repeat(np.arange(1, 11), 180)  # 10 patients, 180 days each
        
        # Create dates
        start_date = pd.Timestamp('2025-01-01')
        dates = []
        for patient_id in range(1, 11):
            patient_dates = [start_date + pd.Timedelta(days=i) for i in range(180)]
            dates.extend(patient_dates)
        
        # Create migraine events (approximately 15% of days)
        np.random.seed(42)
        next_day_migraine = np.random.binomial(1, 0.15, size=len(patient_ids))
        
        # Create sleep data
        total_sleep_hours = np.random.normal(7, 1.5, size=len(patient_ids))
        total_sleep_hours = np.clip(total_sleep_hours, 0, 12)
        sleep_quality = np.random.randint(1, 11, size=len(patient_ids))
        
        # Create weather data
        temperature = np.random.normal(20, 8, size=len(patient_ids))
        humidity = np.random.normal(50, 15, size=len(patient_ids))
        humidity = np.clip(humidity, 0, 100)
        barometric_pressure = np.random.normal(1013, 5, size=len(patient_ids))
        
        # Create pressure changes (some days with significant drops)
        pressure_change_24h = np.random.normal(0, 3, size=len(patient_ids))
        # Add some significant drops (about 10% of days)
        significant_drops = np.random.choice(len(patient_ids), size=int(len(patient_ids)*0.1), replace=False)
        pressure_change_24h[significant_drops] = np.random.uniform(-15, -5, size=len(significant_drops))
        
        # Create stress and dietary data
        stress_level = np.random.randint(1, 11, size=len(patient_ids))
        alcohol_consumed = np.random.binomial(1, 0.2, size=len(patient_ids))
        caffeine_consumed = np.random.binomial(1, 0.5, size=len(patient_ids))
        chocolate_consumed = np.random.binomial(1, 0.3, size=len(patient_ids))
        
        # Create combined data DataFrame
        combined_data = pd.DataFrame({
            'patient_id': patient_ids,
            'date': dates,
            'next_day_migraine': next_day_migraine,
            'total_sleep_hours': total_sleep_hours,
            'sleep_quality': sleep_quality,
            'temperature': temperature,
            'humidity': humidity,
            'barometric_pressure': barometric_pressure,
            'pressure_change_24h': pressure_change_24h,
            'stress_level': stress_level,
            'alcohol_consumed': alcohol_consumed,
            'caffeine_consumed': caffeine_consumed,
            'chocolate_consumed': chocolate_consumed
        })
        
        # Create individual modality DataFrames
        sleep_data = combined_data[['patient_id', 'date', 'total_sleep_hours', 'sleep_quality']].copy()
        weather_data = combined_data[['patient_id', 'date', 'temperature', 'humidity', 'barometric_pressure', 'pressure_change_24h']].copy()
        stress_diet_data = combined_data[['patient_id', 'date', 'stress_level', 'alcohol_consumed', 'caffeine_consumed', 'chocolate_consumed']].copy()
        
        # Save data to CSV files
        combined_data.to_csv(os.path.join(cls.data_dir, 'combined_data.csv'), index=False)
        sleep_data.to_csv(os.path.join(cls.data_dir, 'sleep_data.csv'), index=False)
        weather_data.to_csv(os.path.join(cls.data_dir, 'weather_data.csv'), index=False)
        stress_diet_data.to_csv(os.path.join(cls.data_dir, 'stress_diet_data.csv'), index=False)
    
    def test_load_data(self):
        """Test the load_data function."""
        combined_data, sleep_data, weather_data, stress_diet_data = streamlit_dashboard.load_data(self.data_dir)
        
        # Check that data was loaded correctly
        self.assertEqual(len(combined_data), 1800)  # 10 patients * 180 days
        self.assertEqual(len(sleep_data), 1800)
        self.assertEqual(len(weather_data), 1800)
        self.assertEqual(len(stress_diet_data), 1800)
        
        # Check that all required columns are present
        required_columns = [
            'patient_id', 'date', 'next_day_migraine', 'total_sleep_hours',
            'barometric_pressure', 'pressure_change_24h', 'stress_level'
        ]
        for column in required_columns:
            self.assertIn(column, combined_data.columns)
    
    def test_plot_confusion_matrix(self):
        """Test the plot_confusion_matrix function."""
        # Create sample data
        y_true = np.random.binomial(1, 0.2, size=100)
        y_pred = np.random.random(size=100)
        
        # Create confusion matrix plot
        fig = streamlit_dashboard.plot_confusion_matrix(y_true, y_pred, threshold=0.5)
        
        # Check that figure was created
        self.assertIsNotNone(fig)
    
    def test_plot_roc_curve(self):
        """Test the plot_roc_curve function."""
        # Create sample data
        y_true = np.random.binomial(1, 0.2, size=100)
        y_pred = np.random.random(size=100)
        
        # Create ROC curve plot
        fig, fpr, tpr, thresholds, roc_auc = streamlit_dashboard.plot_roc_curve(y_true, y_pred)
        
        # Check that figure and data were created
        self.assertIsNotNone(fig)
        self.assertIsNotNone(fpr)
        self.assertIsNotNone(tpr)
        self.assertIsNotNone(thresholds)
        self.assertIsNotNone(roc_auc)
    
    def test_plot_precision_recall_curve(self):
        """Test the plot_precision_recall_curve function."""
        # Create sample data
        y_true = np.random.binomial(1, 0.2, size=100)
        y_pred = np.random.random(size=100)
        
        # Create precision-recall curve plot
        fig, precision, recall, thresholds = streamlit_dashboard.plot_precision_recall_curve(y_true, y_pred)
        
        # Check that figure and data were created
        self.assertIsNotNone(fig)
        self.assertIsNotNone(precision)
        self.assertIsNotNone(recall)
        self.assertIsNotNone(thresholds)
    
    def test_plot_threshold_analysis(self):
        """Test the plot_threshold_analysis function."""
        # Create sample data
        y_true = np.random.binomial(1, 0.2, size=100)
        y_pred = np.random.random(size=100)
        
        # Get ROC curve data
        _, fpr, tpr, thresholds, _ = streamlit_dashboard.plot_roc_curve(y_true, y_pred)
        
        # Create threshold analysis plot
        fig = streamlit_dashboard.plot_threshold_analysis(fpr, tpr, thresholds, y_true, y_pred)
        
        # Check that figure was created
        self.assertIsNotNone(fig)
    
    def test_plot_expert_contributions(self):
        """Test the plot_expert_contributions function."""
        # Create sample data
        batch_size = 100
        gate_weights = np.random.random(size=(batch_size, 3))
        gate_weights = gate_weights / np.sum(gate_weights, axis=1, keepdims=True)  # Normalize to sum to 1
        expert_names = ['Sleep Expert', 'Weather Expert', 'Stress/Diet Expert']
        
        # Create expert contributions plot
        fig = streamlit_dashboard.plot_expert_contributions(gate_weights, expert_names)
        
        # Check that figure was created
        self.assertIsNotNone(fig)
    
    def test_plot_expert_contributions_by_outcome(self):
        """Test the plot_expert_contributions_by_outcome function."""
        # Create sample data
        batch_size = 100
        gate_weights = np.random.random(size=(batch_size, 3))
        gate_weights = gate_weights / np.sum(gate_weights, axis=1, keepdims=True)  # Normalize to sum to 1
        y_true = np.random.binomial(1, 0.2, size=batch_size)
        expert_names = ['Sleep Expert', 'Weather Expert', 'Stress/Diet Expert']
        
        # Create expert contributions by outcome plot
        fig = streamlit_dashboard.plot_expert_contributions_by_outcome(gate_weights, y_true, expert_names)
        
        # Check that figure was created
        self.assertIsNotNone(fig)
    
    def test_plot_trigger_analysis(self):
        """Test the plot_trigger_analysis function."""
        # Load test data
        combined_data, _, _, _ = streamlit_dashboard.load_data(self.data_dir)
        
        # Create sample predictions
        predictions = np.random.random(size=len(combined_data))
        
        # Create trigger analysis plot
        fig = streamlit_dashboard.plot_trigger_analysis(combined_data, predictions)
        
        # Check that figure was created
        self.assertIsNotNone(fig)
    
    def test_plot_patient_timeline(self):
        """Test the plot_patient_timeline function."""
        # Load test data
        combined_data, _, _, _ = streamlit_dashboard.load_data(self.data_dir)
        
        # Get data for one patient
        patient_data = combined_data[combined_data['patient_id'] == 1].copy()
        
        # Create sample predictions
        predictions = np.random.random(size=len(patient_data))
        
        # Create patient timeline plot
        fig = streamlit_dashboard.plot_patient_timeline(patient_data, predictions)
        
        # Check that figure was created
        self.assertIsNotNone(fig)
    
    def test_prepare_data_for_prediction(self):
        """Test the prepare_data_for_prediction function."""
        # Load test data
        combined_data, sleep_data, weather_data, stress_diet_data = streamlit_dashboard.load_data(self.data_dir)
        
        # Prepare data for prediction
        X_list = streamlit_dashboard.prepare_data_for_prediction(
            combined_data, sleep_data, weather_data, stress_diet_data
        )
        
        # Check that the output has the right format
        self.assertEqual(len(X_list), 3)
        self.assertEqual(X_list[0].shape[0], len(combined_data))
        self.assertEqual(X_list[0].shape[1], 7)  # 7-day sequence
        self.assertEqual(X_list[0].shape[2], 6)  # 6 features
        self.assertEqual(X_list[1].shape[0], len(combined_data))
        self.assertEqual(X_list[1].shape[1], 4)  # 4 features
        self.assertEqual(X_list[2].shape[0], len(combined_data))
        self.assertEqual(X_list[2].shape[1], 7)  # 7-day sequence
        self.assertEqual(X_list[2].shape[2], 6)  # 6 features

if __name__ == '__main__':
    unittest.main()
