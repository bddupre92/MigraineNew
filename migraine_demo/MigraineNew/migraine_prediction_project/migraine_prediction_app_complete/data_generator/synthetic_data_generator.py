"""
Synthetic Data Generator for Migraine Prediction App

This is the main module that orchestrates the generation of synthetic data
for the migraine prediction app, integrating all the individual generators.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime

# Use relative imports for modules in the same directory
from .patient_profile_generator import PatientProfileGenerator
from .weather_data_generator import WeatherDataGenerator
from .sleep_data_generator import SleepDataGenerator
from .stress_diet_generator import StressDietGenerator
from .migraine_event_generator import MigraineEventGenerator
from .correlation_engine import CorrelationEngine
from .data_export_module import DataExportModule

class SyntheticDataGenerator:
    """
    Main class that orchestrates the generation of synthetic data for migraine prediction.
    
    Attributes:
        num_patients (int): Number of patients to generate data for
        days (int): Number of days to generate data for
        output_dir (str): Directory to save the generated data
        seed (int): Random seed for reproducibility
    """
    
    def __init__(self, num_patients=1000, days=180, output_dir='./data', seed=None):
        """
        Initialize the SyntheticDataGenerator.
        
        Args:
            num_patients (int): Number of patients to generate data for
            days (int): Number of days to generate data for (default: 180 days = 6 months)
            output_dir (str): Directory to save the generated data
            seed (int): Random seed for reproducibility
        """
        self.num_patients = num_patients
        self.days = days
        self.output_dir = output_dir
        self.seed = seed
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_data(self):
        """
        Generate all synthetic data for migraine prediction.
        
        Returns:
            dict: Dictionary with paths to all exported files
        """
        print(f"Generating synthetic data for {self.num_patients} patients over {self.days} days...")
        
        # 1. Generate patient profiles
        print("Generating patient profiles...")
        profile_gen = PatientProfileGenerator(
            num_patients=self.num_patients, 
            chronic_ratio=0.15, 
            with_aura_ratio=0.3,
            seed=self.seed
        )
        patient_profiles = profile_gen.generate_profiles()
        
        # 2. Generate modality-specific data
        # Sleep data
        print("Generating sleep data...")
        sleep_gen = SleepDataGenerator(
            patient_profiles=patient_profiles,
            days=self.days,
            seed=self.seed
        )
        sleep_data, sleep_triggers = sleep_gen.generate_sleep_data()
        
        # Weather data
        print("Generating weather data...")
        weather_gen = WeatherDataGenerator(
            patient_profiles=patient_profiles,
            days=self.days,
            seed=self.seed
        )
        weather_data, weather_triggers = weather_gen.generate_weather_data()
        
        # Stress/Diet data
        print("Generating stress and dietary data...")
        stress_diet_gen = StressDietGenerator(
            patient_profiles=patient_profiles,
            days=self.days,
            seed=self.seed
        )
        stress_diet_data, stress_diet_triggers = stress_diet_gen.generate_stress_diet_data()
        
        # 3. Combine triggers and ensure correlations
        print("Combining triggers and ensuring correlations...")
        correlation_engine = CorrelationEngine(
            patient_profiles=patient_profiles,
            seed=self.seed
        )
        
        combined_triggers = correlation_engine.combine_triggers(
            sleep_triggers, weather_triggers, stress_diet_triggers
        )
        
        # 4. Generate migraine events based on triggers
        print("Generating migraine events based on triggers...")
        migraine_gen = MigraineEventGenerator(
            patient_profiles=patient_profiles,
            days=self.days,
            seed=self.seed
        )
        migraine_events = migraine_gen.generate_events(combined_triggers)
        
        # 5. Ensure correlations between modalities
        print("Enhancing correlations between modalities...")
        sleep_data, weather_data, stress_diet_data = correlation_engine.ensure_correlations(
            migraine_events, sleep_data, weather_data, stress_diet_data
        )
        
        # 6. Export data
        print("Exporting data to CSV files...")
        exporter = DataExportModule(output_dir=self.output_dir)
        export_paths = exporter.export_data(
            patient_profiles, migraine_events, sleep_data, weather_data, stress_diet_data
        )
        
        print(f"Data generation complete. Files saved to {self.output_dir}/")
        
        # 7. Generate and print summary statistics
        self._print_summary_statistics(patient_profiles, migraine_events)
        
        return export_paths
    
    def _print_summary_statistics(self, patient_profiles, migraine_events):
        """
        Generate and print summary statistics about the generated data.
        
        Args:
            patient_profiles (list): List of patient profile dictionaries
            migraine_events (dict): Dictionary with patient_ids as keys and arrays of migraine events
        """
        # Patient statistics
        chronic_count = sum(1 for p in patient_profiles if p['is_chronic'])
        with_aura_count = sum(1 for p in patient_profiles if p['with_aura_ratio'])
        
        print("\nSummary Statistics:")
        print(f"Total patients: {len(patient_profiles)}")
        print(f"- Chronic patients: {chronic_count} ({chronic_count/len(patient_profiles)*100:.1f}%)")
        print(f"- Patients with aura: {with_aura_count} ({with_aura_count/len(patient_profiles)*100:.1f}%)")
        
        # Migraine event statistics
        total_migraine_days = 0
        total_with_aura = 0
        
        for patient_id, events in migraine_events.items():
            migraine_days = sum(1 for day in range(len(events)) if events[day]['has_migraine'])
            with_aura_days = sum(1 for day in range(len(events)) if events[day]['has_migraine'] and events[day]['with_aura'])
            
            total_migraine_days += migraine_days
            total_with_aura += with_aura_days
        
        avg_migraine_days = total_migraine_days / len(patient_profiles)
        avg_migraine_days_per_month = avg_migraine_days / (self.days / 30)
        
        print(f"Total migraine days: {total_migraine_days}")
        print(f"- Average migraine days per patient: {avg_migraine_days:.1f}")
        print(f"- Average migraine days per patient per month: {avg_migraine_days_per_month:.1f}")
        
        if total_migraine_days > 0:
            print(f"- Migraines with aura: {total_with_aura} ({total_with_aura/total_migraine_days*100:.1f}%)")

if __name__ == "__main__":
    # Example usage
    generator = SyntheticDataGenerator(
        num_patients=100,  # Smaller number for testing
        days=180,          # 6 months of data
        output_dir='./data',
        seed=42            # For reproducibility
    )
    
    export_paths = generator.generate_data()
    
    # Print paths to exported files
    print("\nExported files:")
    for name, path in export_paths.items():
        print(f"- {name}: {path}")
