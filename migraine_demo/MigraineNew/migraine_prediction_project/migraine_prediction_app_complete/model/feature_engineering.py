import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
from scipy import signal, stats
import os
import pywt

class FeatureEngineer:
    """
    A class for enhancing features for migraine prediction expert models.
    Implements advanced feature engineering techniques for sleep, weather, 
    physiological, and stress/diet data.
    """
    
    def __init__(self, output_dir='output'):
        """
        Initialize the FeatureEngineer.
        
        Args:
            output_dir (str): Directory to save results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def enhance_sleep_features(self, sleep_data):
        """
        Enhance sleep data features with temporal patterns and derived metrics.
        
        Args:
            sleep_data (np.ndarray): Raw sleep data with shape (samples, time_steps, features)
            
        Returns:
            np.ndarray: Enhanced sleep features
        """
        print("Enhancing sleep features...")
        
        # Extract basic shape information
        n_samples = sleep_data.shape[0]
        n_timesteps = sleep_data.shape[1]
        n_features = sleep_data.shape[2]
        
        # Initialize enhanced features array with a safe size
        # We'll add several new features based on the input data
        n_enhanced_features = n_features * 5  # Estimate of features we'll create
        enhanced_features = np.zeros((n_samples, n_enhanced_features))
        
        for i in range(n_samples):
            sample = sleep_data[i]
            feature_idx = 0
            
            # 1. Extract statistical features across time
            for j in range(n_features):
                feature_series = sample[:, j]
                
                # Basic statistics
                enhanced_features[i, feature_idx] = np.mean(feature_series)
                feature_idx += 1
                enhanced_features[i, feature_idx] = np.std(feature_series)
                feature_idx += 1
                enhanced_features[i, feature_idx] = np.max(feature_series) - np.min(feature_series)
                feature_idx += 1
                
                # Trend features (linear regression slope)
                x = np.arange(len(feature_series))
                slope, _, _, _, _ = stats.linregress(x, feature_series)
                enhanced_features[i, feature_idx] = slope
                feature_idx += 1
                
                # Variability features
                if len(feature_series) > 1:
                    # Day-to-day changes
                    daily_changes = np.diff(feature_series)
                    enhanced_features[i, feature_idx] = np.mean(np.abs(daily_changes))
                    feature_idx += 1
                    enhanced_features[i, feature_idx] = np.std(daily_changes)
                    feature_idx += 1
                else:
                    # Placeholder if not enough data points
                    enhanced_features[i, feature_idx:feature_idx+2] = 0
                    feature_idx += 2
                
                # Frequency domain features - FIX: Check array bounds before accessing
                if len(feature_series) >= 4:  # Need at least a few points for FFT
                    fft_features = np.abs(np.fft.rfft(feature_series))
                    if len(fft_features) > 1:  # Make sure we have more than just DC component
                        enhanced_features[i, feature_idx] = np.sum(fft_features[1:]) / len(fft_features[1:])  # Average power
                    else:
                        enhanced_features[i, feature_idx] = 0  # No frequency components
                    feature_idx += 1
                    
                    # Dominant frequency
                    if len(fft_features) > 1:
                        enhanced_features[i, feature_idx] = np.argmax(fft_features[1:]) + 1
                    else:
                        enhanced_features[i, feature_idx] = 0
                    feature_idx += 1
                else:
                    # Placeholder if not enough data points
                    enhanced_features[i, feature_idx:feature_idx+2] = 0
                    feature_idx += 2
            
            # 2. Sleep quality composite score (weighted combination of relevant features)
            # Assuming first few features are duration, efficiency, deep sleep %, REM %
            if n_features >= 4:
                sleep_quality = (
                    0.3 * np.mean(sample[:, 0]) +  # Duration
                    0.3 * np.mean(sample[:, 1]) +  # Efficiency
                    0.2 * np.mean(sample[:, 2]) +  # Deep sleep
                    0.2 * np.mean(sample[:, 3])    # REM sleep
                )
                enhanced_features[i, feature_idx] = sleep_quality
                feature_idx += 1
            else:
                enhanced_features[i, feature_idx] = np.mean(sample)
                feature_idx += 1
            
            # 3. Sleep consistency (inverse of standard deviation across days)
            sleep_consistency = 1.0 / (np.std(np.mean(sample, axis=1)) + 1e-6)
            enhanced_features[i, feature_idx] = sleep_consistency
            feature_idx += 1
            
            # 4. Sleep debt estimation (cumulative deviation from ideal 8 hours)
            if n_features >= 1:  # Assuming first feature is duration
                ideal_sleep = 8.0  # 8 hours as ideal
                sleep_debt = np.cumsum(ideal_sleep - sample[:, 0])
                enhanced_features[i, feature_idx] = sleep_debt[-1]  # Final accumulated debt
                feature_idx += 1
                enhanced_features[i, feature_idx] = np.max(sleep_debt)  # Maximum debt
                feature_idx += 1
            else:
                enhanced_features[i, feature_idx:feature_idx+2] = 0
                feature_idx += 2
            
            # 5. Circadian disruption (variation in sleep onset time)
            # Assuming we have sleep onset times in the last feature
            if n_features >= 1:
                # Simulate sleep onset time variation (in hours)
                sleep_onset_variation = np.std(sample[:, -1])
                enhanced_features[i, feature_idx] = sleep_onset_variation
                feature_idx += 1
            else:
                enhanced_features[i, feature_idx] = 0
                feature_idx += 1
        
        # Trim any unused feature columns
        if feature_idx < enhanced_features.shape[1]:
            enhanced_features = enhanced_features[:, :feature_idx]
        
        print(f"Enhanced sleep features shape: {enhanced_features.shape}")
        return enhanced_features
    
    def enhance_weather_features(self, weather_data):
        """
        Enhance weather data features with derived metrics and temporal patterns.
        
        Args:
            weather_data (np.ndarray): Raw weather data with shape (samples, time_steps, features)
            
        Returns:
            np.ndarray: Enhanced weather features
        """
        print("Enhancing weather features...")
        
        # Extract basic shape information
        n_samples = weather_data.shape[0]
        n_timesteps = weather_data.shape[1]
        n_features = weather_data.shape[2]
        
        # Initialize enhanced features array with a safe size
        n_enhanced_features = n_features * 6  # Estimate of features we'll create
        enhanced_features = np.zeros((n_samples, n_enhanced_features))
        
        for i in range(n_samples):
            sample = weather_data[i]
            feature_idx = 0
            
            # 1. Extract statistical features across time
            for j in range(n_features):
                feature_series = sample[:, j]
                
                # Basic statistics
                enhanced_features[i, feature_idx] = np.mean(feature_series)
                feature_idx += 1
                enhanced_features[i, feature_idx] = np.std(feature_series)
                feature_idx += 1
                enhanced_features[i, feature_idx] = np.max(feature_series)
                feature_idx += 1
                enhanced_features[i, feature_idx] = np.min(feature_series)
                feature_idx += 1
                enhanced_features[i, feature_idx] = np.max(feature_series) - np.min(feature_series)  # Range
                feature_idx += 1
                
                # Rate of change features
                if len(feature_series) > 1:
                    # Calculate changes
                    changes = np.diff(feature_series)
                    enhanced_features[i, feature_idx] = np.mean(changes)  # Average change
                    feature_idx += 1
                    enhanced_features[i, feature_idx] = np.max(np.abs(changes))  # Maximum absolute change
                    feature_idx += 1
                    enhanced_features[i, feature_idx] = np.sum(np.abs(changes))  # Total change magnitude
                    feature_idx += 1
                else:
                    # Placeholder if not enough data points
                    enhanced_features[i, feature_idx:feature_idx+3] = 0
                    feature_idx += 3
                
                # Frequency domain features - FIX: Check array bounds before accessing
                if len(feature_series) >= 4:  # Need at least a few points for FFT
                    fft_features = np.abs(np.fft.rfft(feature_series))
                    if len(fft_features) > 1:  # Make sure we have more than just DC component
                        enhanced_features[i, feature_idx] = np.sum(fft_features[1:]) / len(fft_features[1:])  # Average power
                    else:
                        enhanced_features[i, feature_idx] = 0  # No frequency components
                    feature_idx += 1
                    
                    # Dominant frequency
                    if len(fft_features) > 1:
                        enhanced_features[i, feature_idx] = np.argmax(fft_features[1:]) + 1
                    else:
                        enhanced_features[i, feature_idx] = 0
                    feature_idx += 1
                else:
                    # Placeholder if not enough data points
                    enhanced_features[i, feature_idx:feature_idx+2] = 0
                    feature_idx += 2
            
            # 2. Weather pattern change indicators
            # Assuming first few features are temperature, humidity, pressure, etc.
            if n_features >= 3 and n_timesteps > 1:
                # Temperature changes
                temp_changes = np.diff(sample[:, 0])
                enhanced_features[i, feature_idx] = np.sum(np.abs(temp_changes) > 5)  # Count significant temp changes
                feature_idx += 1
                
                # Pressure changes (important for migraine)
                pressure_changes = np.diff(sample[:, 2])
                enhanced_features[i, feature_idx] = np.sum(np.abs(pressure_changes) > 3)  # Count significant pressure changes
                feature_idx += 1
                enhanced_features[i, feature_idx] = np.max(np.abs(pressure_changes))  # Maximum pressure change
                feature_idx += 1
            else:
                enhanced_features[i, feature_idx:feature_idx+3] = 0
                feature_idx += 3
            
            # 3. Derived comfort indices
            if n_features >= 2:  # Assuming temp and humidity are first two features
                # Heat index (simplified)
                temp = np.mean(sample[:, 0])
                humidity = np.mean(sample[:, 1])
                heat_index = 0.5 * (temp + 61.0 + ((temp - 68.0) * 1.2) + (humidity * 0.094))
                enhanced_features[i, feature_idx] = heat_index
                feature_idx += 1
                
                # Discomfort index
                discomfort = temp - 0.55 * (1 - 0.01 * humidity) * (temp - 14.5)
                enhanced_features[i, feature_idx] = discomfort
                feature_idx += 1
            else:
                enhanced_features[i, feature_idx:feature_idx+2] = 0
                feature_idx += 2
            
            # 4. Weather stability metric
            if n_features >= 1:
                # Calculate overall weather stability (inverse of average normalized std across features)
                feature_stds = np.zeros(n_features)
                feature_ranges = np.zeros(n_features)
                
                for j in range(n_features):
                    feature_series = sample[:, j]
                    feature_range = np.max(feature_series) - np.min(feature_series)
                    if feature_range > 0:
                        feature_stds[j] = np.std(feature_series) / feature_range
                    else:
                        feature_stds[j] = 0
                    feature_ranges[j] = feature_range
                
                weather_stability = 1.0 / (np.mean(feature_stds) + 1e-6)
                enhanced_features[i, feature_idx] = weather_stability
                feature_idx += 1
                
                # Weather variability (sum of normalized ranges)
                weather_variability = np.sum(feature_ranges)
                enhanced_features[i, feature_idx] = weather_variability
                feature_idx += 1
            else:
                enhanced_features[i, feature_idx:feature_idx+2] = 0
                feature_idx += 2
        
        # Trim any unused feature columns
        if feature_idx < enhanced_features.shape[1]:
            enhanced_features = enhanced_features[:, :feature_idx]
        
        print(f"Enhanced weather features shape: {enhanced_features.shape}")
        return enhanced_features
    
    def enhance_physio_features(self, physio_data):
        """
        Enhance physiological data features with derived metrics and patterns.
        
        Args:
            physio_data (np.ndarray): Raw physiological data with shape (samples, time_steps, features)
            
        Returns:
            np.ndarray: Enhanced physiological features
        """
        print("Enhancing physiological features...")
        
        # Extract basic shape information
        n_samples = physio_data.shape[0]
        n_timesteps = physio_data.shape[1]
        n_features = physio_data.shape[2]
        
        # Initialize enhanced features array with a safe size
        n_enhanced_features = n_features * 7  # Estimate of features we'll create
        enhanced_features = np.zeros((n_samples, n_enhanced_features))
        
        for i in range(n_samples):
            sample = physio_data[i]
            feature_idx = 0
            
            # 1. Extract statistical features across time
            for j in range(n_features):
                feature_series = sample[:, j]
                
                # Basic statistics
                enhanced_features[i, feature_idx] = np.mean(feature_series)
                feature_idx += 1
                enhanced_features[i, feature_idx] = np.std(feature_series)
                feature_idx += 1
                enhanced_features[i, feature_idx] = np.median(feature_series)
                feature_idx += 1
                enhanced_features[i, feature_idx] = stats.skew(feature_series)  # Skewness
                feature_idx += 1
                enhanced_features[i, feature_idx] = stats.kurtosis(feature_series)  # Kurtosis
                feature_idx += 1
                
                # Rate of change features
                if len(feature_series) > 1:
                    # Calculate changes
                    changes = np.diff(feature_series)
                    enhanced_features[i, feature_idx] = np.mean(changes)  # Average change
                    feature_idx += 1
                    enhanced_features[i, feature_idx] = np.max(np.abs(changes))  # Maximum absolute change
                    feature_idx += 1
                else:
                    # Placeholder if not enough data points
                    enhanced_features[i, feature_idx:feature_idx+2] = 0
                    feature_idx += 2
                
                # Frequency domain features - FIX: Check array bounds before accessing
                if len(feature_series) >= 4:  # Need at least a few points for FFT
                    fft_features = np.abs(np.fft.rfft(feature_series))
                    if len(fft_features) > 1:  # Make sure we have more than just DC component
                        enhanced_features[i, feature_idx] = np.sum(fft_features[1:]) / len(fft_features[1:])  # Average power
                    else:
                        enhanced_features[i, feature_idx] = 0  # No frequency components
                    feature_idx += 1
                    
                    # Dominant frequency
                    if len(fft_features) > 1:
                        enhanced_features[i, feature_idx] = np.argmax(fft_features[1:]) + 1
                    else:
                        enhanced_features[i, feature_idx] = 0
                    feature_idx += 1
                else:
                    # Placeholder if not enough data points
                    enhanced_features[i, feature_idx:feature_idx+2] = 0
                    feature_idx += 2
            
            # 2. Heart rate variability metrics (if applicable)
            if n_features >= 1:  # Assuming first feature might be heart rate
                # Simulate HRV metrics
                hr_series = sample[:, 0]
                if len(hr_series) > 1:
                    # RMSSD (Root Mean Square of Successive Differences)
                    rmssd = np.sqrt(np.mean(np.diff(hr_series) ** 2))
                    enhanced_features[i, feature_idx] = rmssd
                    feature_idx += 1
                    
                    # pNN50 (percentage of successive RR intervals that differ by more than 50 ms)
                    pnn50 = np.sum(np.abs(np.diff(hr_series)) > 50) / len(np.diff(hr_series))
                    enhanced_features[i, feature_idx] = pnn50
                    feature_idx += 1
                else:
                    enhanced_features[i, feature_idx:feature_idx+2] = 0
                    feature_idx += 2
            else:
                enhanced_features[i, feature_idx:feature_idx+2] = 0
                feature_idx += 2
            
            # 3. Signal complexity measures
            if n_features >= 1 and n_timesteps >= 4:
                # Sample Entropy (simplified approximation)
                for j in range(min(n_features, 2)):  # Apply to first two features if available
                    feature_series = sample[:, j]
                    # Normalize the series
                    normalized_series = (feature_series - np.mean(feature_series)) / (np.std(feature_series) + 1e-6)
                    # Count similar patterns (simplified)
                    patterns = np.abs(normalized_series[:-1] - normalized_series[1:])
                    similar_patterns = np.sum(patterns < 0.2)
                    entropy = -np.log(similar_patterns / len(patterns) + 1e-6)
                    enhanced_features[i, feature_idx] = entropy
                    feature_idx += 1
            else:
                enhanced_features[i, feature_idx:feature_idx+2] = 0
                feature_idx += 2
            
            # 4. Physiological stress indicators
            if n_features >= 2:  # Assuming we have HR and another stress indicator
                # Simplified stress score
                stress_score = 0.7 * np.mean(sample[:, 0]) + 0.3 * np.mean(sample[:, 1])
                enhanced_features[i, feature_idx] = stress_score
                feature_idx += 1
            else:
                enhanced_features[i, feature_idx] = 0
                feature_idx += 1
        
        # Trim any unused feature columns
        if feature_idx < enhanced_features.shape[1]:
            enhanced_features = enhanced_features[:, :feature_idx]
        
        print(f"Enhanced physiological features shape: {enhanced_features.shape}")
        return enhanced_features
    
    def enhance_stress_diet_features(self, stress_diet_data):
        """
        Enhance stress and diet data features with derived metrics and patterns.
        
        Args:
            stress_diet_data (np.ndarray): Raw stress/diet data with shape (samples, time_steps, features)
            
        Returns:
            np.ndarray: Enhanced stress/diet features
        """
        print("Enhancing stress/diet features...")
        
        # Extract basic shape information
        n_samples = stress_diet_data.shape[0]
        n_timesteps = stress_diet_data.shape[1]
        n_features = stress_diet_data.shape[2]
        
        # Initialize enhanced features array with a safe size
        n_enhanced_features = n_features * 6  # Estimate of features we'll create
        enhanced_features = np.zeros((n_samples, n_enhanced_features))
        
        for i in range(n_samples):
            sample = stress_diet_data[i]
            feature_idx = 0
            
            # 1. Extract statistical features across time
            for j in range(n_features):
                feature_series = sample[:, j]
                
                # Basic statistics
                enhanced_features[i, feature_idx] = np.mean(feature_series)
                feature_idx += 1
                enhanced_features[i, feature_idx] = np.std(feature_series)
                feature_idx += 1
                enhanced_features[i, feature_idx] = np.max(feature_series)
                feature_idx += 1
                enhanced_features[i, feature_idx] = np.min(feature_series)
                feature_idx += 1
                
                # Trend features (linear regression slope)
                x = np.arange(len(feature_series))
                slope, _, _, _, _ = stats.linregress(x, feature_series)
                enhanced_features[i, feature_idx] = slope
                feature_idx += 1
                
                # Variability features
                if len(feature_series) > 1:
                    # Day-to-day changes
                    daily_changes = np.diff(feature_series)
                    enhanced_features[i, feature_idx] = np.mean(np.abs(daily_changes))
                    feature_idx += 1
                    enhanced_features[i, feature_idx] = np.max(np.abs(daily_changes))
                    feature_idx += 1
                else:
                    # Placeholder if not enough data points
                    enhanced_features[i, feature_idx:feature_idx+2] = 0
                    feature_idx += 2
                
                # Frequency domain features - FIX: Check array bounds before accessing
                if len(feature_series) >= 4:  # Need at least a few points for FFT
                    fft_features = np.abs(np.fft.rfft(feature_series))
                    if len(fft_features) > 1:  # Make sure we have more than just DC component
                        enhanced_features[i, feature_idx] = np.sum(fft_features[1:]) / len(fft_features[1:])  # Average power
                    else:
                        enhanced_features[i, feature_idx] = 0  # No frequency components
                    feature_idx += 1
                    
                    # Dominant frequency
                    if len(fft_features) > 1:
                        enhanced_features[i, feature_idx] = np.argmax(fft_features[1:]) + 1
                    else:
                        enhanced_features[i, feature_idx] = 0
                    feature_idx += 1
                else:
                    # Placeholder if not enough data points
                    enhanced_features[i, feature_idx:feature_idx+2] = 0
                    feature_idx += 2
            
            # 2. Stress accumulation features
            if n_features >= 1:  # Assuming first feature is stress level
                stress_series = sample[:, 0]
                # Cumulative stress
                cumulative_stress = np.cumsum(stress_series)
                enhanced_features[i, feature_idx] = cumulative_stress[-1]  # Final accumulated stress
                feature_idx += 1
                enhanced_features[i, feature_idx] = np.max(cumulative_stress)  # Maximum accumulated stress
                feature_idx += 1
                
                # Stress volatility
                if len(stress_series) > 1:
                    stress_changes = np.diff(stress_series)
                    enhanced_features[i, feature_idx] = np.sum(np.abs(stress_changes))  # Total stress fluctuation
                    feature_idx += 1
                else:
                    enhanced_features[i, feature_idx] = 0
                    feature_idx += 1
            else:
                enhanced_features[i, feature_idx:feature_idx+3] = 0
                feature_idx += 3
            
            # 3. Diet quality features
            if n_features >= 3:  # Assuming we have some diet-related features
                # Diet consistency (inverse of standard deviation)
                diet_consistency = 1.0 / (np.std(np.mean(sample[:, 1:3], axis=1)) + 1e-6)
                enhanced_features[i, feature_idx] = diet_consistency
                feature_idx += 1
                
                # Diet quality score (simplified)
                diet_quality = 0.5 * np.mean(sample[:, 1]) + 0.5 * np.mean(sample[:, 2])
                enhanced_features[i, feature_idx] = diet_quality
                feature_idx += 1
            else:
                enhanced_features[i, feature_idx:feature_idx+2] = 0
                feature_idx += 2
            
            # 4. Combined stress-diet interaction features
            if n_features >= 3:  # Assuming we have stress and diet features
                # Stress-diet interaction (simplified)
                stress_diet_interaction = np.mean(sample[:, 0]) * np.mean(sample[:, 1:3])
                enhanced_features[i, feature_idx] = np.mean(stress_diet_interaction)
                feature_idx += 1
            else:
                enhanced_features[i, feature_idx] = 0
                feature_idx += 1
        
        # Trim any unused feature columns
        if feature_idx < enhanced_features.shape[1]:
            enhanced_features = enhanced_features[:, :feature_idx]
        
        print(f"Enhanced stress/diet features shape: {enhanced_features.shape}")
        return enhanced_features
    
    def select_best_features(self, X, y, k=10):
        """
        Select the best k features using univariate statistical tests.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            k (int): Number of features to select
            
        Returns:
            np.ndarray: Selected features
        """
        print(f"Selecting best {k} features...")
        
        # Ensure k is not larger than the number of features
        k = min(k, X.shape[1])
        
        # Apply SelectKBest
        selector = SelectKBest(f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature indices
        selected_indices = selector.get_support(indices=True)
        
        # Get feature scores
        scores = selector.scores_
        
        # Plot feature importance
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(scores)), scores)
        plt.xlabel('Feature Index')
        plt.ylabel('F-Score')
        plt.title('Feature Importance')
        plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Selected {k} best features with indices: {selected_indices}")
        return X_selected, selected_indices
    
    def apply_pca(self, X, n_components=0.95):
        """
        Apply PCA for dimensionality reduction.
        
        Args:
            X (np.ndarray): Feature matrix
            n_components (float or int): Number of components or variance ratio to preserve
            
        Returns:
            np.ndarray: PCA-transformed features
        """
        print(f"Applying PCA with n_components={n_components}...")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Plot explained variance
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Explained Variance')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'pca_explained_variance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"PCA reduced dimensions from {X.shape[1]} to {X_pca.shape[1]}")
        return X_pca, pca, scaler
