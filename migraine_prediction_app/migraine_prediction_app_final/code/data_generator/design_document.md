# Synthetic Data Generator Design for Migraine Prediction

## Overview

This document outlines the design for synthetic data generators that will produce realistic time-series data for the migraine prediction app. The data will cover three modalities in Phase 1 (MVP):

1. Sleep data
2. Weather data
3. Stress/Dietary logs

Each modality will be generated with specific patterns that correlate with migraine events, following the trigger specifications provided by the user.

## Data Specifications

### Time Period
- 6-month (180 days) time series per patient
- Daily timestamps with appropriate granularity per modality

### Patient Population
- Total patients: 1000 (configurable)
- Migraine distribution:
  - 15% chronic patients (≥15 days/month with migraines)
  - 85% episodic patients (<15 days/month with migraines)
- Migraine type distribution:
  - 30% with aura
  - 70% without aura

### Migraine Triggers
- Barometric pressure drops ≥ 5 hPa within 24h (increases next-day migraine probability by ~15-20%)
- Sleep disruptions: nights with < 5h or > 9h sleep (increases next-day migraine probability by ~15-20%)
- Stress spikes: sudden +3-5 points on a 1-10 daily stress scale (increases next-day migraine probability by ~15-20%)
- Dietary flags: alcohol, caffeine, chocolate intake days (increases next-day migraine probability by ~15-20%)

## Architecture

The data generator system will consist of the following components:

1. **Patient Profile Generator**: Creates baseline patient profiles with demographic information and migraine susceptibility parameters
2. **Migraine Event Generator**: Determines when migraine events occur based on triggers and patient susceptibility
3. **Modality-Specific Generators**:
   - Sleep data generator
   - Weather data generator
   - Stress/Dietary logs generator
4. **Correlation Engine**: Ensures realistic correlations between modalities and migraine events
5. **Data Export Module**: Formats and exports data for use in the FuseMoE model

## Detailed Component Design

### 1. Patient Profile Generator

```python
class PatientProfileGenerator:
    def __init__(self, num_patients=1000, chronic_ratio=0.15, with_aura_ratio=0.3, seed=None):
        self.num_patients = num_patients
        self.chronic_ratio = chronic_ratio
        self.with_aura_ratio = with_aura_ratio
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
    def generate_profiles(self):
        """Generate patient profiles with demographic and susceptibility information."""
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
```

### 2. Migraine Event Generator

```python
class MigraineEventGenerator:
    def __init__(self, patient_profiles, days=180, seed=None):
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
            Dictionary with patient_ids as keys and arrays of daily migraine events as values
        """
        migraine_events = {}
        
        for profile in self.patient_profiles:
            patient_id = profile['patient_id']
            patient_triggers = triggers_data[patient_id]
            
            # Initialize migraine events array
            events = np.zeros(self.days, dtype={
                'names': ['date', 'has_migraine', 'with_aura', 'severity', 'duration'],
                'formats': ['datetime64[D]', 'bool', 'bool', 'float32', 'float32']
            })
            
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
                    events[day]['with_aura'] = self.rng.random() < profile['with_aura_ratio']
                    
                    # Determine severity (1-10 scale)
                    events[day]['severity'] = self.rng.uniform(3, 10)
                    
                    # Determine duration (hours)
                    events[day]['duration'] = self.rng.uniform(4, 72)
            
            migraine_events[patient_id] = events
            
        return migraine_events
```

### 3. Modality-Specific Generators

#### 3.1 Sleep Data Generator

```python
class SleepDataGenerator:
    def __init__(self, patient_profiles, days=180, seed=None):
        self.patient_profiles = patient_profiles
        self.days = days
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
    def generate_sleep_data(self):
        """Generate synthetic sleep data for all patients."""
        sleep_data = {}
        sleep_triggers = {}
        
        for profile in self.patient_profiles:
            patient_id = profile['patient_id']
            
            # Initialize sleep data array
            data = np.zeros(self.days, dtype={
                'names': ['date', 'total_sleep_hours', 'deep_sleep_pct', 'rem_sleep_pct', 
                          'light_sleep_pct', 'awake_time_mins', 'sleep_quality'],
                'formats': ['datetime64[D]', 'float32', 'float32', 'float32', 
                            'float32', 'float32', 'float32']
            })
            
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
```

#### 3.2 Weather Data Generator

```python
class WeatherDataGenerator:
    def __init__(self, patient_profiles, days=180, seed=None):
        self.patient_profiles = patient_profiles
        self.days = days
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
    def generate_weather_data(self):
        """Generate synthetic weather data for all patients."""
        weather_data = {}
        weather_triggers = {}
        
        # Generate a set of base weather patterns (assuming patients are in different locations)
        num_locations = min(20, len(self.patient_profiles))  # Up to 20 different weather patterns
        base_weather_patterns = []
        
        for _ in range(num_locations):
            # Initialize weather data array for this location
            data = np.zeros(self.days, dtype={
                'names': ['date', 'temperature', 'humidity', 'pressure', 'pressure_change_24h'],
                'formats': ['datetime64[D]', 'float32', 'float32', 'float32', 'float32']
            })
            
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
```

#### 3.3 Stress/Dietary Logs Generator

```python
class StressDietGenerator:
    def __init__(self, patient_profiles, days=180, seed=None):
        self.patient_profiles = patient_profiles
        self.days = days
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
    def generate_stress_diet_data(self):
        """Generate synthetic stress and dietary data for all patients."""
        stress_diet_data = {}
        stress_diet_triggers = {}
        
        for profile in self.patient_profiles:
            patient_id = profile['patient_id']
            
            # Initialize stress/diet data array
            data = np.zeros(self.days, dtype={
                'names': ['date', 'stress_level', 'alcohol_consumed', 'caffeine_consumed', 
                          'chocolate_consumed', 'processed_food_consumed', 'water_consumed_liters'],
                'formats': ['datetime64[D]', 'float32', 'bool', 'bool', 'bool', 'bool', 'float32']
            })
            
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
```

### 4. Correlation Engine

```python
class CorrelationEngine:
    def __init__(self, patient_profiles, seed=None):
        self.patient_profiles = patient_profiles
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def combine_triggers(self, sleep_triggers, weather_triggers, stress_diet_triggers):
        """
        Combine triggers from different modalities into a single triggers dataset.
        
        Returns:
            Dictionary with patient_ids as keys and arrays of daily trigger information as values
        """
        combined_triggers = {}
        
        for profile in self.patient_profiles:
            patient_id = profile['patient_id']
            days = len(sleep_triggers[patient_id])
            
            # Create combined triggers array
            triggers = np.zeros(days, dtype={
                'names': ['sleep_trigger', 'weather_trigger', 'stress_trigger', 'diet_trigger'],
                'formats': ['bool', 'bool', 'bool', 'bool']
            })
            
            # Fill in triggers from each modality
            triggers['sleep_trigger'] = sleep_triggers[patient_id]
            triggers['weather_trigger'] = weather_triggers[patient_id]
            triggers['stress_trigger'] = stress_diet_triggers[patient_id]
            triggers['diet_trigger'] = stress_diet_triggers[patient_id]  # Using same array for both stress and diet
            
            combined_triggers[patient_id] = triggers
        
        return combined_triggers
    
    def ensure_correlations(self, migraine_events, sleep_data, weather_data, stress_diet_data):
        """
        Ensure realistic correlations between modalities and migraine events.
        This may adjust some data points to strengthen the correlations.
        """
        # Implementation would enhance correlations between modalities
        # For example, stress might affect sleep quality, or migraine events might affect next-day stress
        
        # This is a placeholder for the actual implementation
        return sleep_data, weather_data, stress_diet_data
```

### 5. Data Export Module

```python
class DataExportModule:
    def __init__(self, output_dir='./data'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def export_data(self, patient_profiles, migraine_events, sleep_data, weather_data, stress_diet_data):
        """Export all data to CSV files."""
        # Export patient profiles
        profiles_df = pd.DataFrame(patient_profiles)
        profiles_df.to_csv(f'{self.output_dir}/patient_profiles.csv', index=False)
        
        # Export migraine events
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
        
        events_df = pd.DataFrame(all_events)
        events_df.to_csv(f'{self.output_dir}/migraine_events.csv', index=False)
        
        # Export sleep data
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
        
        sleep_df = pd.DataFrame(all_sleep)
        sleep_df.to_csv(f'{self.output_dir}/sleep_data.csv', index=False)
        
        # Export weather data
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
        
        weather_df = pd.DataFrame(all_weather)
        weather_df.to_csv(f'{self.output_dir}/weather_data.csv', index=False)
        
        # Export stress/diet data
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
        
        stress_diet_df = pd.DataFrame(all_stress_diet)
        stress_diet_df.to_csv(f'{self.output_dir}/stress_diet_data.csv', index=False)
        
        # Export combined dataset for easy model training
        # This joins all modalities into a single dataframe with migraine labels
        self._export_combined_dataset(migraine_events, sleep_data, weather_data, stress_diet_data)
    
    def _export_combined_dataset(self, migraine_events, sleep_data, weather_data, stress_diet_data):
        """Create and export a combined dataset with all modalities and migraine labels."""
        # Implementation would join all datasets on patient_id and date
        # Then export as a single CSV file for easy model training
        pass
```

## Main Generator Class

```python
class SyntheticDataGenerator:
    def __init__(self, num_patients=1000, days=180, output_dir='./data', seed=None):
        self.num_patients = num_patients
        self.days = days
        self.output_dir = output_dir
        self.seed = seed
        
    def generate_data(self):
        """Generate all synthetic data for migraine prediction."""
        # 1. Generate patient profiles
        profile_gen = PatientProfileGenerator(
            num_patients=self.num_patients, 
            chronic_ratio=0.15, 
            with_aura_ratio=0.3,
            seed=self.seed
        )
        patient_profiles = profile_gen.generate_profiles()
        
        # 2. Generate modality-specific data
        # Sleep data
        sleep_gen = SleepDataGenerator(
            patient_profiles=patient_profiles,
            days=self.days,
            seed=self.seed
        )
        sleep_data, sleep_triggers = sleep_gen.generate_sleep_data()
        
        # Weather data
        weather_gen = WeatherDataGenerator(
            patient_profiles=patient_profiles,
            days=self.days,
            seed=self.seed
        )
        weather_data, weather_triggers = weather_gen.generate_weather_data()
        
        # Stress/Diet data
        stress_diet_gen = StressDietGenerator(
            patient_profiles=patient_profiles,
            days=self.days,
            seed=self.seed
        )
        stress_diet_data, stress_diet_triggers = stress_diet_gen.generate_stress_diet_data()
        
        # 3. Combine triggers and ensure correlations
        correlation_engine = CorrelationEngine(
            patient_profiles=patient_profiles,
            seed=self.seed
        )
        
        combined_triggers = correlation_engine.combine_triggers(
            sleep_triggers, weather_triggers, stress_diet_triggers
        )
        
        # 4. Generate migraine events based on triggers
        migraine_gen = MigraineEventGenerator(
            patient_profiles=patient_profiles,
            days=self.days,
            seed=self.seed
        )
        migraine_events = migraine_gen.generate_events(combined_triggers)
        
        # 5. Ensure correlations between modalities
        sleep_data, weather_data, stress_diet_data = correlation_engine.ensure_correlations(
            migraine_events, sleep_data, weather_data, stress_diet_data
        )
        
        # 6. Export data
        exporter = DataExportModule(output_dir=self.output_dir)
        exporter.export_data(
            patient_profiles, migraine_events, sleep_data, weather_data, stress_diet_data
        )
        
        return {
            'patient_profiles': patient_profiles,
            'migraine_events': migraine_events,
            'sleep_data': sleep_data,
            'weather_data': weather_data,
            'stress_diet_data': stress_diet_data
        }
```

## Usage Example

```python
# Generate synthetic data
generator = SyntheticDataGenerator(
    num_patients=1000,  # Number of patients
    days=180,           # 6 months of data
    output_dir='./data',
    seed=42             # For reproducibility
)

data = generator.generate_data()
```

## Data Validation

The data generator will include validation steps to ensure:

1. Realistic distributions of values for each modality
2. Proper correlations between triggers and migraine events
3. Appropriate patient distributions (chronic vs. episodic, with/without aura)
4. Temporal consistency across all modalities

## Next Steps

After designing the synthetic data generators, the next steps will be:

1. Implement the generators according to this design
2. Validate the generated data for realism and proper correlations
3. Use the data to train and evaluate the FuseMoE model
