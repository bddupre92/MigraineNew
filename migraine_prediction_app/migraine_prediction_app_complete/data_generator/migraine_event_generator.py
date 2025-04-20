"""
Migraine Event Generator for Migraine Prediction App

This module generates synthetic migraine events based on patient profiles
and trigger data from various modalities.
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

class MigraineEventGenerator:
    """
    Generates synthetic migraine events based on patient profiles and trigger data.
    
    Attributes:
        patient_profiles (list): List of patient profile dictionaries
        days (int): Number of days to generate data for
        seed (int): Random seed for reproducibility
        rng (RandomState): NumPy random number generator
    """
    
    def __init__(self, patient_profiles, days=180, seed=None):
        """
        Initialize the MigraineEventGenerator.
        
        Args:
            patient_profiles (list): List of patient profile dictionaries
            days (int): Number of days to generate data for
            seed (int): Random seed for reproducibility
        """
        self.patient_profiles = patient_profiles
        self.days = days
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
    def generate_events(self, triggers_data):
        """
        Generate migraine events based on patient profiles and trigger data.
        
        Args:
            triggers_data: Dictionary with keys as patient_ids and values as 
                           arrays of daily trigger information
        
        Returns:
            dict: Dictionary with patient_ids as keys and arrays of daily migraine events as values
        """
        migraine_events = {}
        
        for profile in self.patient_profiles:
            patient_id = profile['patient_id']
            patient_triggers = triggers_data[patient_id]
            
            # Initialize migraine events array
            events = np.zeros(self.days, dtype=[
                ('date', 'datetime64[D]'),
                ('has_migraine', bool),
                ('with_aura', bool),
                ('severity', 'float32'),
                ('duration', 'float32')
            ])
            
            # Set dates
            start_date = np.datetime64('2024-01-01')
            events['date'] = [start_date + np.timedelta64(i, 'D') for i in range(self.days)]
            
            # Generate migraine events
            for day in range(self.days):
                # Base probability from patient profile
                prob = profile['base_frequency']
                
                # Adjust probability based on triggers from previous day
                if day > 0:
                    weather_trigger = patient_triggers[day-1]['weather_trigger']
                    sleep_trigger = patient_triggers[day-1]['sleep_trigger']
                    stress_trigger = patient_triggers[day-1]['stress_trigger']
                    diet_trigger = patient_triggers[day-1]['diet_trigger']
                    
                    # Increase probability based on triggers and patient susceptibility
                    if weather_trigger:
                        prob += 0.15 * profile['trigger_susceptibility']['weather']
                    if sleep_trigger:
                        prob += 0.15 * profile['trigger_susceptibility']['sleep']
                    if stress_trigger:
                        prob += 0.15 * profile['trigger_susceptibility']['stress']
                    if diet_trigger:
                        prob += 0.15 * profile['trigger_susceptibility']['diet']
                
                # Cap probability at 0.95
                prob = min(prob, 0.95)
                
                # Determine if migraine occurs
                has_migraine = self.rng.random() < prob
                events[day]['has_migraine'] = has_migraine
                
                if has_migraine:
                    # Determine if with aura
                    events[day]['with_aura'] = profile['with_aura_ratio']
                    
                    # Determine severity (1-10 scale)
                    events[day]['severity'] = self.rng.uniform(3, 10)
                    
                    # Determine duration (hours)
                    events[day]['duration'] = self.rng.uniform(4, 72)
            
            migraine_events[patient_id] = events
            
        return migraine_events
    
    def save_events(self, migraine_events, output_dir='./data'):
        """
        Save migraine events to a CSV file.
        
        Args:
            migraine_events (dict): Dictionary with patient_ids as keys and arrays of daily migraine events as values
            output_dir (str): Directory to save the CSV file
        
        Returns:
            str: Path to the saved CSV file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Flatten the events data for CSV format
        all_events = []
        for patient_id, events in migraine_events.items():
            for day in range(len(events)):
                if events[day]['has_migraine']:
                    all_events.append({
                        'patient_id': patient_id,
                        'date': events[day]['date'],
                        'with_aura': events[day]['with_aura'],
                        'severity': events[day]['severity'],
                        'duration': events[day]['duration']
                    })
        
        # Convert to DataFrame and save
        df = pd.DataFrame(all_events)
        file_path = os.path.join(output_dir, 'migraine_events.csv')
        df.to_csv(file_path, index=False)
        
        return file_path

if __name__ == "__main__":
    # This would be used in conjunction with other generators
    # Example usage would be demonstrated in the main synthetic data generator script
    pass
