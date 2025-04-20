"""
Patient Profile Generator for Migraine Prediction App

This module generates synthetic patient profiles with demographic information
and migraine susceptibility parameters.
"""

import numpy as np
import pandas as pd
import os

class PatientProfileGenerator:
    """
    Generates synthetic patient profiles with demographic and susceptibility information.
    
    Attributes:
        num_patients (int): Number of patients to generate
        chronic_ratio (float): Ratio of chronic migraine patients (≥15 days/month)
        with_aura_ratio (float): Ratio of patients who experience migraines with aura
        seed (int): Random seed for reproducibility
        rng (RandomState): NumPy random number generator
    """
    
    def __init__(self, num_patients=1000, chronic_ratio=0.15, with_aura_ratio=0.3, seed=None):
        """
        Initialize the PatientProfileGenerator.
        
        Args:
            num_patients (int): Number of patients to generate
            chronic_ratio (float): Ratio of chronic migraine patients (≥15 days/month)
            with_aura_ratio (float): Ratio of patients who experience migraines with aura
            seed (int): Random seed for reproducibility
        """
        self.num_patients = num_patients
        self.chronic_ratio = chronic_ratio
        self.with_aura_ratio = with_aura_ratio
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
    def generate_profiles(self):
        """
        Generate patient profiles with demographic and susceptibility information.
        
        Returns:
            list: List of dictionaries containing patient profiles
        """
        profiles = []
        for i in range(self.num_patients):
            is_chronic = self.rng.random() < self.chronic_ratio
            
            # Base migraine frequency (days per month)
            if is_chronic:
                base_frequency = self.rng.uniform(15, 25)  # Chronic: 15-25 days/month
            else:
                base_frequency = self.rng.uniform(1, 14)   # Episodic: 1-14 days/month
            
            # Susceptibility to different triggers (0-1 scale)
            profile = {
                'patient_id': f'P{i:04d}',
                'age': self.rng.randint(18, 65),
                'sex': self.rng.choice(['M', 'F'], p=[0.3, 0.7]),  # Migraines more common in females
                'is_chronic': is_chronic,
                'base_frequency': base_frequency / 30.0,  # Convert to daily probability
                'with_aura_ratio': self.rng.random() < self.with_aura_ratio,
                'trigger_susceptibility': {
                    'weather': self.rng.uniform(0.1, 1.0),
                    'sleep': self.rng.uniform(0.1, 1.0),
                    'stress': self.rng.uniform(0.1, 1.0),
                    'diet': self.rng.uniform(0.1, 1.0)
                }
            }
            profiles.append(profile)
        return profiles
    
    def save_profiles(self, profiles, output_dir='./data'):
        """
        Save patient profiles to a CSV file.
        
        Args:
            profiles (list): List of patient profile dictionaries
            output_dir (str): Directory to save the CSV file
        
        Returns:
            str: Path to the saved CSV file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Flatten the trigger_susceptibility dictionary for CSV format
        flattened_profiles = []
        for profile in profiles:
            flat_profile = profile.copy()
            for trigger, value in profile['trigger_susceptibility'].items():
                flat_profile[f'susceptibility_{trigger}'] = value
            del flat_profile['trigger_susceptibility']
            flattened_profiles.append(flat_profile)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(flattened_profiles)
        file_path = os.path.join(output_dir, 'patient_profiles.csv')
        df.to_csv(file_path, index=False)
        
        return file_path

if __name__ == "__main__":
    # Example usage
    generator = PatientProfileGenerator(num_patients=100, seed=42)
    profiles = generator.generate_profiles()
    file_path = generator.save_profiles(profiles, output_dir='./data')
    print(f"Saved patient profiles to {file_path}")
    
    # Print summary statistics
    chronic_count = sum(1 for p in profiles if p['is_chronic'])
    with_aura_count = sum(1 for p in profiles if p['with_aura_ratio'])
    
    print(f"Generated {len(profiles)} patient profiles:")
    print(f"- Chronic patients: {chronic_count} ({chronic_count/len(profiles)*100:.1f}%)")
    print(f"- Patients with aura: {with_aura_count} ({with_aura_count/len(profiles)*100:.1f}%)")
