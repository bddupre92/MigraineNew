"""
Data Export Module for Migraine Prediction App

This module handles exporting synthetic data to CSV files and creating
combined datasets for model training.
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime

class DataExportModule:
    """
    Exports synthetic data to CSV files and creates combined datasets.
    
    Attributes:
        output_dir (str): Directory to save the exported data
    """
    
    def __init__(self, output_dir='./data'):
        """
        Initialize the DataExportModule.
        
        Args:
            output_dir (str): Directory to save the exported data
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def export_data(self, patient_profiles, migraine_events, sleep_data, weather_data, stress_diet_data):
        """
        Export all data to CSV files.
        
        Args:
            patient_profiles (list): List of patient profile dictionaries
            migraine_events (dict): Dictionary with patient_ids as keys and arrays of migraine events
            sleep_data (dict): Dictionary with patient_ids as keys and arrays of sleep data
            weather_data (dict): Dictionary with patient_ids as keys and arrays of weather data
            stress_diet_data (dict): Dictionary with patient_ids as keys and arrays of stress/diet data
        
        Returns:
            dict: Dictionary with paths to all exported files
        """
        # Export patient profiles
        profiles_path = self._export_patient_profiles(patient_profiles)
        
        # Export migraine events
        events_path = self._export_migraine_events(migraine_events)
        
        # Export sleep data
        sleep_path = self._export_sleep_data(sleep_data)
        
        # Export weather data
        weather_path = self._export_weather_data(weather_data)
        
        # Export stress/diet data
        stress_diet_path = self._export_stress_diet_data(stress_diet_data)
        
        # Export combined dataset for easy model training
        combined_path = self._export_combined_dataset(migraine_events, sleep_data, weather_data, stress_diet_data)
        
        return {
            'patient_profiles': profiles_path,
            'migraine_events': events_path,
            'sleep_data': sleep_path,
            'weather_data': weather_path,
            'stress_diet_data': stress_diet_path,
            'combined_data': combined_path
        }
    
    def _export_patient_profiles(self, patient_profiles):
        """Export patient profiles to CSV."""
        # Flatten the trigger_susceptibility dictionary for CSV format
        flattened_profiles = []
        for profile in patient_profiles:
            flat_profile = profile.copy()
            for trigger, value in profile['trigger_susceptibility'].items():
                flat_profile[f'susceptibility_{trigger}'] = value
            del flat_profile['trigger_susceptibility']
            flattened_profiles.append(flat_profile)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(flattened_profiles)
        file_path = os.path.join(self.output_dir, 'patient_profiles.csv')
        df.to_csv(file_path, index=False)
        
        return file_path
    
    def _export_migraine_events(self, migraine_events):
        """Export migraine events to CSV."""
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
        file_path = os.path.join(self.output_dir, 'migraine_events.csv')
        df.to_csv(file_path, index=False)
        
        return file_path
    
    def _export_sleep_data(self, sleep_data):
        """Export sleep data to CSV."""
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
        file_path = os.path.join(self.output_dir, 'sleep_data.csv')
        df.to_csv(file_path, index=False)
        
        return file_path
    
    def _export_weather_data(self, weather_data):
        """Export weather data to CSV."""
        all_weather = []
        for patient_id, data in weather_data.items():
            for day in range(len(data)):
                all_weather.append({
                    'patient_id': patient_id,
                    'date': data[day]['date'],
                    'temperature': data[day]['temperature'],
                    'humidity': data[day]['humidity'],
                    'pressure': data[day]['pressure'],
                    'pressure_change_24h': data[day]['pressure_change_24h']
                })
        
        # Convert to DataFrame and save
        df = pd.DataFrame(all_weather)
        file_path = os.path.join(self.output_dir, 'weather_data.csv')
        df.to_csv(file_path, index=False)
        
        return file_path
    
    def _export_stress_diet_data(self, stress_diet_data):
        """Export stress/diet data to CSV."""
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
        file_path = os.path.join(self.output_dir, 'stress_diet_data.csv')
        df.to_csv(file_path, index=False)
        
        return file_path
    
    def _export_combined_dataset(self, migraine_events, sleep_data, weather_data, stress_diet_data):
        """
        Create and export a combined dataset with all modalities and migraine labels.
        
        This joins all modalities into a single dataframe with migraine labels for the next day,
        making it easier to use for model training.
        """
        combined_data = []
        
        # For each patient
        for patient_id in migraine_events.keys():
            patient_migraines = migraine_events[patient_id]
            patient_sleep = sleep_data[patient_id]
            patient_weather = weather_data[patient_id]
            patient_stress_diet = stress_diet_data[patient_id]
            
            # For each day (except the last one)
            for day in range(len(patient_migraines) - 1):
                # Get data from current day
                current_date = patient_migraines[day]['date']
                
                # Get migraine status for next day (target variable)
                next_day_migraine = patient_migraines[day + 1]['has_migraine']
                next_day_with_aura = patient_migraines[day + 1]['with_aura'] if next_day_migraine else False
                next_day_severity = patient_migraines[day + 1]['severity'] if next_day_migraine else 0
                
                # Combine all features
                record = {
                    'patient_id': patient_id,
                    'date': current_date,
                    
                    # Sleep features
                    'total_sleep_hours': patient_sleep[day]['total_sleep_hours'],
                    'deep_sleep_pct': patient_sleep[day]['deep_sleep_pct'],
                    'rem_sleep_pct': patient_sleep[day]['rem_sleep_pct'],
                    'light_sleep_pct': patient_sleep[day]['light_sleep_pct'],
                    'awake_time_mins': patient_sleep[day]['awake_time_mins'],
                    'sleep_quality': patient_sleep[day]['sleep_quality'],
                    
                    # Weather features
                    'temperature': patient_weather[day]['temperature'],
                    'humidity': patient_weather[day]['humidity'],
                    'pressure': patient_weather[day]['pressure'],
                    'pressure_change_24h': patient_weather[day]['pressure_change_24h'],
                    
                    # Stress/Diet features
                    'stress_level': patient_stress_diet[day]['stress_level'],
                    'alcohol_consumed': patient_stress_diet[day]['alcohol_consumed'],
                    'caffeine_consumed': patient_stress_diet[day]['caffeine_consumed'],
                    'chocolate_consumed': patient_stress_diet[day]['chocolate_consumed'],
                    'processed_food_consumed': patient_stress_diet[day]['processed_food_consumed'],
                    'water_consumed_liters': patient_stress_diet[day]['water_consumed_liters'],
                    
                    # Target variables (next day)
                    'next_day_migraine': next_day_migraine,
                    'next_day_with_aura': next_day_with_aura,
                    'next_day_severity': next_day_severity
                }
                
                combined_data.append(record)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(combined_data)
        file_path = os.path.join(self.output_dir, 'combined_data.csv')
        df.to_csv(file_path, index=False)
        
        return file_path

if __name__ == "__main__":
    # This would be used in conjunction with other generators
    # Example usage would be demonstrated in the main synthetic data generator script
    pass
