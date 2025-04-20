"""
Weather Data Generator for Migraine Prediction App

This module generates synthetic weather data with realistic patterns
and barometric pressure changes that can trigger migraines.
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

class WeatherDataGenerator:
    """
    Generates synthetic weather data with realistic patterns and pressure changes.
    
    Attributes:
        patient_profiles (list): List of patient profile dictionaries
        days (int): Number of days to generate data for
        seed (int): Random seed for reproducibility
        rng (RandomState): NumPy random number generator
    """
    
    def __init__(self, patient_profiles, days=180, seed=None):
        """
        Initialize the WeatherDataGenerator.
        
        Args:
            patient_profiles (list): List of patient profile dictionaries
            days (int): Number of days to generate data for
            seed (int): Random seed for reproducibility
        """
        self.patient_profiles = patient_profiles
        self.days = days
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
    def generate_weather_data(self):
        """
        Generate synthetic weather data for all patients.
        
        Returns:
            tuple: (weather_data, weather_triggers) where:
                - weather_data is a dictionary with patient_ids as keys and arrays of daily weather data as values
                - weather_triggers is a dictionary with patient_ids as keys and boolean arrays indicating trigger days
        """
        weather_data = {}
        weather_triggers = {}
        
        # Generate a set of base weather patterns (assuming patients are in different locations)
        num_locations = min(20, len(self.patient_profiles))  # Up to 20 different weather patterns
        base_weather_patterns = []
        
        for _ in range(num_locations):
            # Initialize weather data array for this location
            data = np.zeros(self.days, dtype=[
                ('date', 'datetime64[D]'),
                ('temperature', 'float32'),
                ('humidity', 'float32'),
                ('pressure', 'float32'),
                ('pressure_change_24h', 'float32')
            ])
            
            # Initialize triggers array
            triggers = np.zeros(self.days, dtype=bool)
            
            # Set dates
            start_date = np.datetime64('2024-01-01')
            data['date'] = [start_date + np.timedelta64(i, 'D') for i in range(self.days)]
            
            # Generate base temperature with seasonal pattern (assuming Northern Hemisphere)
            seasonal_temp = 15 + 15 * np.sin(np.linspace(0, 2*np.pi, self.days) - np.pi/2)
            
            # Add random variations to temperature
            temperature = seasonal_temp + self.rng.normal(0, 5, self.days)
            
            # Generate humidity (partially correlated with temperature)
            humidity = 60 + self.rng.normal(0, 15, self.days) - 0.5 * (temperature - seasonal_temp)
            humidity = np.clip(humidity, 20, 100)
            
            # Generate pressure with realistic patterns (around 1013 hPa)
            # Start with a base pressure
            pressure = 1013 + self.rng.normal(0, 2, self.days)
            
            # Add some autocorrelation and occasional pressure systems
            for day in range(1, self.days):
                # Autocorrelation
                pressure[day] = 0.9 * pressure[day] + 0.1 * pressure[day-1]
                
                # Occasionally introduce pressure systems
                if self.rng.random() < 0.1:  # 10% chance of pressure system
                    # Pressure change over next 3-5 days
                    change_duration = self.rng.randint(3, 6)
                    change_magnitude = self.rng.uniform(-15, 15)
                    
                    # Apply gradual change
                    for i in range(min(change_duration, self.days - day)):
                        pressure[day + i] += change_magnitude * (i + 1) / change_duration
            
            # Calculate 24-hour pressure changes
            pressure_change = np.zeros(self.days)
            for day in range(1, self.days):
                pressure_change[day] = pressure[day] - pressure[day-1]
            
            # Mark significant pressure drops as triggers
            for day in range(1, self.days):
                if pressure_change[day] <= -5:  # Drop of 5 hPa or more
                    triggers[day] = True
            
            # Store data in arrays
            data['temperature'] = temperature
            data['humidity'] = humidity
            data['pressure'] = pressure
            data['pressure_change_24h'] = pressure_change
            
            base_weather_patterns.append((data, triggers))
        
        # Assign weather patterns to patients
        for profile in self.patient_profiles:
            patient_id = profile['patient_id']
            
            # Randomly select a weather pattern for this patient
            pattern_idx = self.rng.randint(0, len(base_weather_patterns))
            pattern_data, pattern_triggers = base_weather_patterns[pattern_idx]
            
            # Add some patient-specific variations
            data = pattern_data.copy()
            data['temperature'] += self.rng.uniform(-2, 2)
            data['humidity'] += self.rng.uniform(-5, 5)
            data['pressure'] += self.rng.uniform(-1, 1)
            
            weather_data[patient_id] = data
            weather_triggers[patient_id] = pattern_triggers
        
        return weather_data, weather_triggers
    
    def save_weather_data(self, weather_data, output_dir='./data'):
        """
        Save weather data to a CSV file.
        
        Args:
            weather_data (dict): Dictionary with patient_ids as keys and arrays of daily weather data as values
            output_dir (str): Directory to save the CSV file
        
        Returns:
            str: Path to the saved CSV file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Flatten the weather data for CSV format
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
        file_path = os.path.join(output_dir, 'weather_data.csv')
        df.to_csv(file_path, index=False)
        
        return file_path

if __name__ == "__main__":
    # This would be used in conjunction with other generators
    # Example usage would be demonstrated in the main synthetic data generator script
    pass
