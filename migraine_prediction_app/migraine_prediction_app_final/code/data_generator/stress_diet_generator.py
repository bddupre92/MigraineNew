"""
Stress and Dietary Data Generator for Migraine Prediction App

This module generates synthetic stress levels and dietary data
with patterns that can trigger migraines.
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

class StressDietGenerator:
    """
    Generates synthetic stress and dietary data for patients.
    
    Attributes:
        patient_profiles (list): List of patient profile dictionaries
        days (int): Number of days to generate data for
        seed (int): Random seed for reproducibility
        rng (RandomState): NumPy random number generator
    """
    
    def __init__(self, patient_profiles, days=180, seed=None):
        """
        Initialize the StressDietGenerator.
        
        Args:
            patient_profiles (list): List of patient profile dictionaries
            days (int): Number of days to generate data for
            seed (int): Random seed for reproducibility
        """
        self.patient_profiles = patient_profiles
        self.days = days
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
    def generate_stress_diet_data(self):
        """
        Generate synthetic stress and dietary data for all patients.
        
        Returns:
            tuple: (stress_diet_data, stress_diet_triggers) where:
                - stress_diet_data is a dictionary with patient_ids as keys and arrays of daily stress/diet data
                - stress_diet_triggers is a dictionary with patient_ids as keys and boolean arrays indicating trigger days
        """
        stress_diet_data = {}
        stress_diet_triggers = {}
        
        for profile in self.patient_profiles:
            patient_id = profile['patient_id']
            
            # Initialize stress/diet data array
            data = np.zeros(self.days, dtype=[
                ('date', 'datetime64[D]'),
                ('stress_level', 'float32'),
                ('alcohol_consumed', bool),
                ('caffeine_consumed', bool),
                ('chocolate_consumed', bool),
                ('processed_food_consumed', bool),
                ('water_consumed_liters', 'float32')
            ])
            
            # Initialize triggers array
            triggers = np.zeros(self.days, dtype=bool)
            
            # Set dates
            start_date = np.datetime64('2024-01-01')
            data['date'] = [start_date + np.timedelta64(i, 'D') for i in range(self.days)]
            
            # Generate baseline stress pattern (1-10 scale)
            # Some people have higher baseline stress
            baseline_stress = self.rng.uniform(2, 5)
            
            # Weekly stress pattern (higher on weekdays)
            weekly_stress = np.zeros(self.days)
            for day in range(self.days):
                weekday = day % 7
                if weekday < 5:  # Weekday
                    weekly_stress[day] = self.rng.uniform(0, 2)
                else:  # Weekend
                    weekly_stress[day] = self.rng.uniform(-1, 0.5)
            
            # Generate stress with occasional spikes
            stress = np.zeros(self.days)
            for day in range(self.days):
                # Base stress with some autocorrelation
                if day > 0:
                    stress[day] = 0.7 * (baseline_stress + weekly_stress[day]) + 0.3 * stress[day-1]
                else:
                    stress[day] = baseline_stress + weekly_stress[day]
                
                # Occasional stress spikes
                if self.rng.random() < 0.1:  # 10% chance of stress spike
                    spike = self.rng.uniform(3, 5)
                    stress[day] += spike
                    
                    # Mark as trigger if spike is significant
                    if spike >= 3:
                        triggers[day] = True
                
                # Ensure stress is within 1-10 range
                stress[day] = max(min(stress[day], 10), 1)
            
            data['stress_level'] = stress
            
            # Generate dietary flags
            # Baseline consumption probabilities
            p_alcohol = self.rng.uniform(0.05, 0.3)  # 5-30% of days
            p_caffeine = self.rng.uniform(0.3, 0.9)  # 30-90% of days
            p_chocolate = self.rng.uniform(0.1, 0.4)  # 10-40% of days
            p_processed = self.rng.uniform(0.3, 0.7)  # 30-70% of days
            
            for day in range(self.days):
                # Weekly patterns
                weekday = day % 7
                alcohol_adj = 0.2 if weekday >= 5 else 0  # More alcohol on weekends
                
                # Generate consumption flags
                data[day]['alcohol_consumed'] = self.rng.random() < (p_alcohol + alcohol_adj)
                data[day]['caffeine_consumed'] = self.rng.random() < p_caffeine
                data[day]['chocolate_consumed'] = self.rng.random() < p_chocolate
                data[day]['processed_food_consumed'] = self.rng.random() < p_processed
                
                # Water consumption (liters)
                data[day]['water_consumed_liters'] = self.rng.uniform(0.5, 3.0)
                
                # Mark dietary triggers
                if (data[day]['alcohol_consumed'] or 
                    data[day]['caffeine_consumed'] or 
                    data[day]['chocolate_consumed']):
                    # Only mark as trigger if multiple items consumed or individual susceptibility is high
                    trigger_count = (data[day]['alcohol_consumed'] + 
                                    data[day]['caffeine_consumed'] + 
                                    data[day]['chocolate_consumed'])
                    
                    if trigger_count >= 2 or self.rng.random() < profile['trigger_susceptibility']['diet']:
                        triggers[day] = True
            
            stress_diet_data[patient_id] = data
            stress_diet_triggers[patient_id] = triggers
        
        return stress_diet_data, stress_diet_triggers
    
    def save_stress_diet_data(self, stress_diet_data, output_dir='./data'):
        """
        Save stress and dietary data to a CSV file.
        
        Args:
            stress_diet_data (dict): Dictionary with patient_ids as keys and arrays of daily stress/diet data
            output_dir (str): Directory to save the CSV file
        
        Returns:
            str: Path to the saved CSV file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Flatten the stress/diet data for CSV format
        all_stress_diet = []
        for patient_id, data in stress_diet_data.items():
            for day in range(len(data)):
                all_stress_diet.append({
                    'patient_id': patient_id,
                    'date': data[day]['date'],
                    'stress_level': data[day]['stress_level'],
                    'alcohol_consumed': data[day]['alcohol_consumed'],
                    'caffeine_consumed': data[day]['caffeine_consumed'],
                    'chocolate_consumed': data[day]['chocolate_consumed'],
                    'processed_food_consumed': data[day]['processed_food_consumed'],
                    'water_consumed_liters': data[day]['water_consumed_liters']
                })
        
        # Convert to DataFrame and save
        df = pd.DataFrame(all_stress_diet)
        file_path = os.path.join(output_dir, 'stress_diet_data.csv')
        df.to_csv(file_path, index=False)
        
        return file_path

if __name__ == "__main__":
    # This would be used in conjunction with other generators
    # Example usage would be demonstrated in the main synthetic data generator script
    pass
