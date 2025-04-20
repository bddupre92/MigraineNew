"""
Sleep Data Generator for Migraine Prediction App

This module generates synthetic sleep data with realistic patterns
and sleep disruptions that can trigger migraines.
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

class SleepDataGenerator:
    """
    Generates synthetic sleep data for patients with realistic patterns and disruptions.
    
    Attributes:
        patient_profiles (list): List of patient profile dictionaries
        days (int): Number of days to generate data for
        seed (int): Random seed for reproducibility
        rng (RandomState): NumPy random number generator
    """
    
    def __init__(self, patient_profiles, days=180, seed=None):
        """
        Initialize the SleepDataGenerator.
        
        Args:
            patient_profiles (list): List of patient profile dictionaries
            days (int): Number of days to generate data for
            seed (int): Random seed for reproducibility
        """
        self.patient_profiles = patient_profiles
        self.days = days
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
    def generate_sleep_data(self):
        """
        Generate synthetic sleep data for all patients.
        
        Returns:
            tuple: (sleep_data, sleep_triggers) where:
                - sleep_data is a dictionary with patient_ids as keys and arrays of daily sleep data as values
                - sleep_triggers is a dictionary with patient_ids as keys and boolean arrays indicating trigger days
        """
        sleep_data = {}
        sleep_triggers = {}
        
        for profile in self.patient_profiles:
            patient_id = profile['patient_id']
            
            # Initialize sleep data array
            data = np.zeros(self.days, dtype=[
                ('date', 'datetime64[D]'),
                ('total_sleep_hours', 'float32'),
                ('deep_sleep_pct', 'float32'),
                ('rem_sleep_pct', 'float32'),
                ('light_sleep_pct', 'float32'),
                ('awake_time_mins', 'float32'),
                ('sleep_quality', 'float32')
            ])
            
            # Initialize triggers array
            triggers = np.zeros(self.days, dtype=bool)
            
            # Set dates
            start_date = np.datetime64('2024-01-01')
            data['date'] = [start_date + np.timedelta64(i, 'D') for i in range(self.days)]
            
            # Generate sleep data with realistic patterns
            for day in range(self.days):
                # Base sleep hours (normal distribution around 7 hours)
                base_sleep = self.rng.normal(7, 1.5)
                
                # Add weekly patterns (e.g., more sleep on weekends)
                weekday = day % 7
                if weekday >= 5:  # Weekend
                    base_sleep += self.rng.uniform(0, 1.5)
                
                # Add some autocorrelation (sleep patterns tend to persist)
                if day > 0:
                    base_sleep = 0.7 * base_sleep + 0.3 * data[day-1]['total_sleep_hours']
                
                # Occasionally introduce sleep disruptions (trigger events)
                if self.rng.random() < 0.15:  # 15% chance of sleep disruption
                    if self.rng.random() < 0.5:
                        # Short sleep
                        base_sleep = self.rng.uniform(3, 5)
                    else:
                        # Long sleep
                        base_sleep = self.rng.uniform(9, 12)
                    
                    # Mark as trigger
                    triggers[day] = True
                
                # Ensure sleep hours are within realistic bounds
                total_sleep = max(min(base_sleep, 14), 2)
                data[day]['total_sleep_hours'] = total_sleep
                
                # Generate sleep stage percentages
                deep_pct = self.rng.uniform(0.1, 0.3)
                rem_pct = self.rng.uniform(0.15, 0.25)
                light_pct = 1 - deep_pct - rem_pct
                
                data[day]['deep_sleep_pct'] = deep_pct
                data[day]['rem_sleep_pct'] = rem_pct
                data[day]['light_sleep_pct'] = light_pct
                
                # Generate awake time
                data[day]['awake_time_mins'] = self.rng.uniform(10, 60)
                
                # Calculate sleep quality (1-100 scale)
                # Higher deep sleep and REM, lower awake time = better quality
                quality = 50 + 100 * (deep_pct * 2 + rem_pct - data[day]['awake_time_mins']/120)
                data[day]['sleep_quality'] = max(min(quality, 100), 0)
            
            sleep_data[patient_id] = data
            sleep_triggers[patient_id] = triggers
            
        return sleep_data, sleep_triggers
    
    def save_sleep_data(self, sleep_data, output_dir='./data'):
        """
        Save sleep data to a CSV file.
        
        Args:
            sleep_data (dict): Dictionary with patient_ids as keys and arrays of daily sleep data as values
            output_dir (str): Directory to save the CSV file
        
        Returns:
            str: Path to the saved CSV file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Flatten the sleep data for CSV format
        all_sleep = []
        for patient_id, data in sleep_data.items():
            for day in range(len(data)):
                all_sleep.append({
                    'patient_id': patient_id,
                    'date': data[day]['date'],
                    'total_sleep_hours': data[day]['total_sleep_hours'],
                    'deep_sleep_pct': data[day]['deep_sleep_pct'],
                    'rem_sleep_pct': data[day]['rem_sleep_pct'],
                    'light_sleep_pct': data[day]['light_sleep_pct'],
                    'awake_time_mins': data[day]['awake_time_mins'],
                    'sleep_quality': data[day]['sleep_quality']
                })
        
        # Convert to DataFrame and save
        df = pd.DataFrame(all_sleep)
        file_path = os.path.join(output_dir, 'sleep_data.csv')
        df.to_csv(file_path, index=False)
        
        return file_path

if __name__ == "__main__":
    # This would be used in conjunction with other generators
    # Example usage would be demonstrated in the main synthetic data generator script
    pass
