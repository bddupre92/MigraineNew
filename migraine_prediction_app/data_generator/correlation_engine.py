"""
Correlation Engine for Migraine Prediction App

This module ensures realistic correlations between different data modalities
and migraine events.
"""

import numpy as np
import pandas as pd
import os

class CorrelationEngine:
    """
    Ensures realistic correlations between different data modalities and migraine events.
    
    Attributes:
        patient_profiles (list): List of patient profile dictionaries
        seed (int): Random seed for reproducibility
        rng (RandomState): NumPy random number generator
    """
    
    def __init__(self, patient_profiles, seed=None):
        """
        Initialize the CorrelationEngine.
        
        Args:
            patient_profiles (list): List of patient profile dictionaries
            seed (int): Random seed for reproducibility
        """
        self.patient_profiles = patient_profiles
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def combine_triggers(self, sleep_triggers, weather_triggers, stress_diet_triggers):
        """
        Combine triggers from different modalities into a single triggers dataset.
        
        Args:
            sleep_triggers (dict): Dictionary with patient_ids as keys and boolean arrays for sleep triggers
            weather_triggers (dict): Dictionary with patient_ids as keys and boolean arrays for weather triggers
            stress_diet_triggers (dict): Dictionary with patient_ids as keys and boolean arrays for stress/diet triggers
        
        Returns:
            dict: Dictionary with patient_ids as keys and arrays of daily trigger information
        """
        combined_triggers = {}
        
        for profile in self.patient_profiles:
            patient_id = profile['patient_id']
            days = len(sleep_triggers[patient_id])
            
            # Create combined triggers array
            triggers = np.zeros(days, dtype=[
                ('sleep_trigger', bool),
                ('weather_trigger', bool),
                ('stress_trigger', bool),
                ('diet_trigger', bool)
            ])
            
            # Fill in triggers from each modality
            triggers['sleep_trigger'] = sleep_triggers[patient_id]
            triggers['weather_trigger'] = weather_triggers[patient_id]
            
            # For stress and diet, we need to separate them from the combined stress_diet_triggers
            # For simplicity, we'll use the same array for both stress and diet in this implementation
            triggers['stress_trigger'] = stress_diet_triggers[patient_id]
            triggers['diet_trigger'] = stress_diet_triggers[patient_id]
            
            combined_triggers[patient_id] = triggers
        
        return combined_triggers
    
    def ensure_correlations(self, migraine_events, sleep_data, weather_data, stress_diet_data):
        """
        Ensure realistic correlations between modalities and migraine events.
        This may adjust some data points to strengthen the correlations.
        
        Args:
            migraine_events (dict): Dictionary with patient_ids as keys and arrays of migraine events
            sleep_data (dict): Dictionary with patient_ids as keys and arrays of sleep data
            weather_data (dict): Dictionary with patient_ids as keys and arrays of weather data
            stress_diet_data (dict): Dictionary with patient_ids as keys and arrays of stress/diet data
        
        Returns:
            tuple: (sleep_data, weather_data, stress_diet_data) with enhanced correlations
        """
        # For each patient, enhance correlations between modalities
        for profile in self.patient_profiles:
            patient_id = profile['patient_id']
            
            # Get data for this patient
            patient_migraines = migraine_events[patient_id]
            patient_sleep = sleep_data[patient_id]
            patient_weather = weather_data[patient_id]
            patient_stress_diet = stress_diet_data[patient_id]
            
            # Enhance correlations for each day
            for day in range(1, len(patient_migraines)):
                # If migraine occurs, potentially adjust previous day's data to strengthen triggers
                if patient_migraines[day]['has_migraine']:
                    # With some probability, ensure at least one trigger is present
                    if self.rng.random() < 0.7:  # 70% chance of having a clear trigger
                        # Randomly select which modality to adjust
                        modality = self.rng.choice(['sleep', 'weather', 'stress', 'diet'])
                        
                        if modality == 'sleep' and not patient_sleep[day-1]['total_sleep_hours'] < 5 and not patient_sleep[day-1]['total_sleep_hours'] > 9:
                            # Adjust sleep to be a trigger (either too little or too much)
                            if self.rng.random() < 0.5:
                                patient_sleep[day-1]['total_sleep_hours'] = self.rng.uniform(2, 4.9)
                            else:
                                patient_sleep[day-1]['total_sleep_hours'] = self.rng.uniform(9.1, 12)
                            
                            # Adjust sleep quality accordingly
                            if patient_sleep[day-1]['total_sleep_hours'] < 5:
                                patient_sleep[day-1]['sleep_quality'] = self.rng.uniform(20, 50)
                            
                        elif modality == 'weather' and patient_weather[day-1]['pressure_change_24h'] > -5:
                            # Adjust weather to have a significant pressure drop
                            patient_weather[day-1]['pressure_change_24h'] = self.rng.uniform(-10, -5)
                            patient_weather[day-1]['pressure'] = patient_weather[day-2]['pressure'] + patient_weather[day-1]['pressure_change_24h']
                            
                        elif modality == 'stress' and patient_stress_diet[day-1]['stress_level'] < 8:
                            # Adjust stress to be high
                            patient_stress_diet[day-1]['stress_level'] = self.rng.uniform(8, 10)
                            
                        elif modality == 'diet':
                            # Ensure at least two dietary triggers
                            trigger_count = 0
                            if not patient_stress_diet[day-1]['alcohol_consumed']:
                                patient_stress_diet[day-1]['alcohol_consumed'] = True
                                trigger_count += 1
                            
                            if trigger_count < 2 and not patient_stress_diet[day-1]['caffeine_consumed']:
                                patient_stress_diet[day-1]['caffeine_consumed'] = True
                                trigger_count += 1
                            
                            if trigger_count < 2 and not patient_stress_diet[day-1]['chocolate_consumed']:
                                patient_stress_diet[day-1]['chocolate_consumed'] = True
                                trigger_count += 1
        
        return sleep_data, weather_data, stress_diet_data

if __name__ == "__main__":
    # This would be used in conjunction with other generators
    # Example usage would be demonstrated in the main synthetic data generator script
    pass
