import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Create output directories
os.makedirs('output/data', exist_ok=True)
os.makedirs('output/models', exist_ok=True)
os.makedirs('output/results', exist_ok=True)
os.makedirs('output/evaluation', exist_ok=True)

def generate_sleep_data(n_samples=1000, n_days=7, n_features=6, positive_ratio=0.1):
    """
    Generate synthetic sleep data.
    
    Args:
        n_samples: Number of samples
        n_days: Number of days of data per sample
        n_features: Number of features per day
        positive_ratio: Ratio of positive samples
        
    Returns:
        tuple: Sleep data and labels
    """
    print(f"Generating sleep data: {n_samples} samples, {n_days} days, {n_features} features")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate random sleep data
    X_sleep = np.random.randn(n_samples, n_days, n_features)
    
    # Generate labels
    y = np.zeros(n_samples)
    n_positive = int(n_samples * positive_ratio)
    
    # For positive samples, introduce sleep patterns associated with migraines
    for i in range(n_positive):
        # Sleep duration reduction (feature 0)
        X_sleep[i, -2:, 0] -= 1.5  # Reduced sleep before migraine
        
        # Sleep quality reduction (feature 1)
        X_sleep[i, -3:, 1] -= 1.2  # Reduced quality before migraine
        
        # Increased interruptions (feature 2)
        X_sleep[i, -2:, 2] += 1.0  # More interruptions before migraine
        
        # REM sleep reduction (feature 3)
        X_sleep[i, -2:, 3] -= 0.8  # Less REM sleep before migraine
        
        # Set label to positive
        y[i] = 1
    
    # Shuffle data
    indices = np.random.permutation(n_samples)
    X_sleep = X_sleep[indices]
    y = y[indices]
    
    print(f"Generated sleep data: {X_sleep.shape}, labels: {y.shape}, positive ratio: {np.mean(y):.2f}")
    
    return X_sleep, y

def generate_weather_data(n_samples=1000, n_features=4, positive_ratio=0.1):
    """
    Generate synthetic weather data.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        positive_ratio: Ratio of positive samples
        
    Returns:
        tuple: Weather data and labels
    """
    print(f"Generating weather data: {n_samples} samples, {n_features} features")
    
    # Set random seed for reproducibility
    np.random.seed(43)
    
    # Generate random weather data
    X_weather = np.random.randn(n_samples, n_features)
    
    # Generate labels
    y = np.zeros(n_samples)
    n_positive = int(n_samples * positive_ratio)
    
    # For positive samples, introduce weather patterns associated with migraines
    for i in range(n_positive):
        # Barometric pressure changes (feature 0)
        X_weather[i, 0] += 1.8  # Significant pressure change
        
        # Humidity (feature 1)
        X_weather[i, 1] += 1.2  # Higher humidity
        
        # Set label to positive
        y[i] = 1
    
    # Shuffle data
    indices = np.random.permutation(n_samples)
    X_weather = X_weather[indices]
    y = y[indices]
    
    print(f"Generated weather data: {X_weather.shape}, labels: {y.shape}, positive ratio: {np.mean(y):.2f}")
    
    return X_weather, y

def generate_stress_diet_data(n_samples=1000, n_days=7, n_features=6, positive_ratio=0.1):
    """
    Generate synthetic stress and diet data.
    
    Args:
        n_samples: Number of samples
        n_days: Number of days of data per sample
        n_features: Number of features per day
        positive_ratio: Ratio of positive samples
        
    Returns:
        tuple: Stress/diet data and labels
    """
    print(f"Generating stress/diet data: {n_samples} samples, {n_days} days, {n_features} features")
    
    # Set random seed for reproducibility
    np.random.seed(44)
    
    # Generate random stress/diet data
    X_stress_diet = np.random.randn(n_samples, n_days, n_features)
    
    # Generate labels
    y = np.zeros(n_samples)
    n_positive = int(n_samples * positive_ratio)
    
    # For positive samples, introduce stress/diet patterns associated with migraines
    for i in range(n_positive):
        # Stress level increase (feature 0)
        X_stress_diet[i, -3:, 0] += 1.5  # Increased stress before migraine
        
        # Caffeine intake increase (feature 1)
        X_stress_diet[i, -2:, 1] += 1.2  # Increased caffeine before migraine
        
        # Water intake decrease (feature 2)
        X_stress_diet[i, -2:, 2] -= 1.0  # Decreased hydration before migraine
        
        # Set label to positive
        y[i] = 1
    
    # Shuffle data
    indices = np.random.permutation(n_samples)
    X_stress_diet = X_stress_diet[indices]
    y = y[indices]
    
    print(f"Generated stress/diet data: {X_stress_diet.shape}, labels: {y.shape}, positive ratio: {np.mean(y):.2f}")
    
    return X_stress_diet, y

def generate_physio_data(n_samples=1000, n_features=5, positive_ratio=0.1):
    """
    Generate synthetic physiological data.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        positive_ratio: Ratio of positive samples
        
    Returns:
        tuple: Physiological data and labels
    """
    print(f"Generating physiological data: {n_samples} samples, {n_features} features")
    
    # Set random seed for reproducibility
    np.random.seed(45)
    
    # Generate random physiological data
    X_physio = np.random.randn(n_samples, n_features)
    
    # Generate labels
    y = np.zeros(n_samples)
    n_positive = int(n_samples * positive_ratio)
    
    # For positive samples, introduce physiological patterns associated with migraines
    for i in range(n_positive):
        # Heart rate variability decrease (feature 0)
        X_physio[i, 0] -= 1.2  # Decreased HRV before migraine
        
        # Cortisol level increase (feature 1)
        X_physio[i, 1] += 1.5  # Increased cortisol before migraine
        
        # Set label to positive
        y[i] = 1
    
    # Shuffle data
    indices = np.random.permutation(n_samples)
    X_physio = X_physio[indices]
    y = y[indices]
    
    print(f"Generated physiological data: {X_physio.shape}, labels: {y.shape}, positive ratio: {np.mean(y):.2f}")
    
    return X_physio, y

def combine_data_and_split(X_sleep, X_weather, X_stress_diet, X_physio, y, test_size=0.2, val_size=0.1):
    """
    Combine data from different sources and split into train, validation, and test sets.
    
    Args:
        X_sleep: Sleep data
        X_weather: Weather data
        X_stress_diet: Stress/diet data
        X_physio: Physiological data
        y: Labels
        test_size: Proportion of data for testing
        val_size: Proportion of data for validation
        
    Returns:
        tuple: Train, validation, and test data
    """
    print("Combining data and splitting into train, validation, and test sets")
    
    # Verify all data has the same number of samples
    n_samples = len(y)
    assert X_sleep.shape[0] == n_samples
    assert X_weather.shape[0] == n_samples
    assert X_stress_diet.shape[0] == n_samples
    assert X_physio.shape[0] == n_samples
    
    # First split: train+val vs test
    X_sleep_train_val, X_sleep_test, X_weather_train_val, X_weather_test, X_stress_diet_train_val, X_stress_diet_test, X_physio_train_val, X_physio_test, y_train_val, y_test = train_test_split(
        X_sleep, X_weather, X_stress_diet, X_physio, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    X_sleep_train, X_sleep_val, X_weather_train, X_weather_val, X_stress_diet_train, X_stress_diet_val, X_physio_train, X_physio_val, y_train, y_val = train_test_split(
        X_sleep_train_val, X_weather_train_val, X_stress_diet_train_val, X_physio_train_val, y_train_val, test_size=val_ratio, random_state=42, stratify=y_train_val
    )
    
    # Print split sizes
    print(f"Train set: {len(y_train)} samples, positive ratio: {np.mean(y_train):.2f}")
    print(f"Validation set: {len(y_val)} samples, positive ratio: {np.mean(y_val):.2f}")
    print(f"Test set: {len(y_test)} samples, positive ratio: {np.mean(y_test):.2f}")
    
    # Create data dictionaries
    train_data = {
        'X_sleep': X_sleep_train,
        'X_weather': X_weather_train,
        'X_stress_diet': X_stress_diet_train,
        'X_physio': X_physio_train,
        'y': y_train
    }
    
    val_data = {
        'X_sleep': X_sleep_val,
        'X_weather': X_weather_val,
        'X_stress_diet': X_stress_diet_val,
        'X_physio': X_physio_val,
        'y': y_val
    }
    
    test_data = {
        'X_sleep': X_sleep_test,
        'X_weather': X_weather_test,
        'X_stress_diet': X_stress_diet_test,
        'X_physio': X_physio_test,
        'y': y_test
    }
    
    return train_data, val_data, test_data

def apply_smote(train_data):
    """
    Apply SMOTE to balance the training data.
    
    Args:
        train_data: Training data dictionary
        
    Returns:
        dict: Balanced training data
    """
    print("Applying SMOTE to balance training data")
    
    # Extract data
    X_sleep = train_data['X_sleep']
    X_weather = train_data['X_weather']
    X_stress_diet = train_data['X_stress_diet']
    X_physio = train_data['X_physio']
    y = train_data['y']
    
    # Flatten sleep data
    X_sleep_flat = X_sleep.reshape(X_sleep.shape[0], -1)
    
    # Flatten stress/diet data
    X_stress_diet_flat = X_stress_diet.reshape(X_stress_diet.shape[0], -1)
    
    # Combine all features
    X_combined = np.hstack([X_sleep_flat, X_weather, X_stress_diet_flat, X_physio])
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_combined_balanced, y_balanced = smote.fit_resample(X_combined, y)
    
    # Split back into separate data types
    sleep_size = X_sleep_flat.shape[1]
    weather_size = X_weather.shape[1]
    stress_diet_size = X_stress_diet_flat.shape[1]
    physio_size = X_physio.shape[1]
    
    X_sleep_balanced = X_combined_balanced[:, :sleep_size].reshape(-1, X_sleep.shape[1], X_sleep.shape[2])
    X_weather_balanced = X_combined_balanced[:, sleep_size:sleep_size+weather_size]
    X_stress_diet_balanced = X_combined_balanced[:, sleep_size+weather_size:sleep_size+weather_size+stress_diet_size].reshape(-1, X_stress_diet.shape[1], X_stress_diet.shape[2])
    X_physio_balanced = X_combined_balanced[:, sleep_size+weather_size+stress_diet_size:]
    
    # Create balanced data dictionary
    balanced_train_data = {
        'X_sleep': X_sleep_balanced,
        'X_weather': X_weather_balanced,
        'X_stress_diet': X_stress_diet_balanced,
        'X_physio': X_physio_balanced,
        'y': y_balanced
    }
    
    print(f"Original training data: {len(y)} samples, positive ratio: {np.mean(y):.2f}")
    print(f"Balanced training data: {len(y_balanced)} samples, positive ratio: {np.mean(y_balanced):.2f}")
    
    return balanced_train_data

def save_data(train_data, val_data, test_data, output_dir='output/data'):
    """
    Save data to files.
    
    Args:
        train_data: Training data dictionary
        val_data: Validation data dictionary
        test_data: Test data dictionary
        output_dir: Output directory
    """
    print(f"Saving data to {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training data
    np.savez(
        os.path.join(output_dir, 'train_data.npz'),
        X_sleep=train_data['X_sleep'],
        X_weather=train_data['X_weather'],
        X_stress_diet=train_data['X_stress_diet'],
        X_physio=train_data['X_physio'],
        y=train_data['y']
    )
    
    # Save validation data
    np.savez(
        os.path.join(output_dir, 'val_data.npz'),
        X_sleep=val_data['X_sleep'],
        X_weather=val_data['X_weather'],
        X_stress_diet=val_data['X_stress_diet'],
        X_physio=val_data['X_physio'],
        y=val_data['y']
    )
    
    # Save test data
    np.savez(
        os.path.join(output_dir, 'test_data.npz'),
        X_sleep=test_data['X_sleep'],
        X_weather=test_data['X_weather'],
        X_stress_diet=test_data['X_stress_diet'],
        X_physio=test_data['X_physio'],
        y=test_data['y']
    )
    
    print("Data saved successfully")

def plot_data_distribution(train_data, val_data, test_data, output_dir='output/data'):
    """
    Plot data distribution.
    
    Args:
        train_data: Training data dictionary
        val_data: Validation data dictionary
        test_data: Test data dictionary
        output_dir: Output directory
    """
    print("Plotting data distribution")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract labels
    y_train = train_data['y']
    y_val = val_data['y']
    y_test = test_data['y']
    
    # Calculate positive ratios
    train_pos_ratio = np.mean(y_train)
    val_pos_ratio = np.mean(y_val)
    test_pos_ratio = np.mean(y_test)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot positive ratios
    plt.bar(['Train', 'Validation', 'Test'], [train_pos_ratio, val_pos_ratio, test_pos_ratio])
    plt.ylabel('Positive Ratio')
    plt.title('Migraine Event Ratio in Data Splits')
    plt.ylim(0, 1)
    
    # Add value labels
    for i, v in enumerate([train_pos_ratio, val_pos_ratio, test_pos_ratio]):
        plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'data_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Data distribution plot saved")

def main():
    """
    Main function to generate synthetic data.
    """
    print("Starting synthetic data generation")
    start_time = time.time()
    
    # Set parameters
    n_samples = 2000
    positive_ratio = 0.1
    
    # Generate data
    X_sleep, y_sleep = generate_sleep_data(n_samples=n_samples, positive_ratio=positive_ratio)
    X_weather, y_weather = generate_weather_data(n_samples=n_samples, positive_ratio=positive_ratio)
    X_stress_diet, y_stress_diet = generate_stress_diet_data(n_samples=n_samples, positive_ratio=positive_ratio)
    X_physio, y_physio = generate_physio_data(n_samples=n_samples, positive_ratio=positive_ratio)
    
    # Combine labels (use sleep labels as reference)
    y = y_sleep
    
    # Split data
    train_data, val_data, test_data = combine_data_and_split(X_sleep, X_weather, X_stress_diet, X_physio, y)
    
    # Apply SMOTE to balance training data
    balanced_train_data = apply_smote(train_data)
    
    # Save data
    save_data(balanced_train_data, val_data, test_data)
    
    # Plot data distribution
    plot_data_distribution(balanced_train_data, val_data, test_data)
    
    # Print completion message
    elapsed_time = time.time() - start_time
    print(f"Synthetic data generation completed in {elapsed_time:.2f} seconds")
    print(f"Data saved to output/data/")

if __name__ == "__main__":
    main()
