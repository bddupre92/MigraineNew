# Advanced Feature Engineering for Migraine Prediction

This document outlines advanced feature engineering techniques to enhance the migraine prediction model's ability to capture complex patterns and relationships in the data.

## Overview

While the current feature engineering implementation provides a solid foundation, we can further enhance the model's performance by implementing more sophisticated feature extraction and transformation techniques that better capture the complex nature of migraine triggers and their interactions.

## Proposed Feature Engineering Techniques

### 1. Causal Feature Discovery

Implement causal discovery algorithms to identify true migraine triggers rather than mere correlations.

```python
import networkx as nx
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils

class CausalFeatureDiscovery:
    def __init__(self, data_dict, output_dir='output/causal_features'):
        self.data_dict = data_dict
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def discover_causal_features(self):
        """Discover causal relationships in the data."""
        # Combine all features
        X_combined = self._combine_features()
        y = self.data_dict['y']
        
        # Add target variable to data
        data = np.hstack([X_combined, y.reshape(-1, 1)])
        
        # Run PC algorithm
        skeleton, separating_sets = pc(data, alpha=0.05)
        
        # Convert to graph
        G = nx.DiGraph()
        n_features = X_combined.shape[1]
        
        # Add nodes
        for i in range(n_features):
            G.add_node(i, name=f"Feature_{i}")
        G.add_node(n_features, name="Migraine")
        
        # Add edges
        for i, j in skeleton:
            if j == n_features:  # Edge to migraine
                G.add_edge(i, j, weight=1.0)
        
        # Identify direct causes of migraine
        direct_causes = [i for i, j in G.edges() if j == n_features]
        
        # Create causal features
        causal_features = X_combined[:, direct_causes]
        
        # Plot causal graph
        self._plot_causal_graph(G)
        
        return causal_features, direct_causes
    
    def _combine_features(self):
        """Combine all features into a single matrix."""
        X_sleep = self.data_dict['X_sleep']
        X_weather = self.data_dict['X_weather']
        X_stress_diet = self.data_dict['X_stress_diet']
        X_physio = self.data_dict['X_physio']
        
        # Flatten 3D features if needed
        if len(X_sleep.shape) > 2:
            X_sleep = X_sleep.reshape(X_sleep.shape[0], -1)
        if len(X_stress_diet.shape) > 2:
            X_stress_diet = X_stress_diet.reshape(X_stress_diet.shape[0], -1)
        
        # Combine all features
        X_combined = np.hstack([X_sleep, X_weather, X_stress_diet, X_physio])
        return X_combined
    
    def _plot_causal_graph(self, G):
        """Plot the causal graph."""
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                              node_color=['red' if n == max(G.nodes()) else 'skyblue' for n in G.nodes()],
                              node_size=500)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, arrows=True)
        
        # Draw labels
        labels = {n: G.nodes[n]['name'] for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels)
        
        plt.title("Causal Graph for Migraine Prediction")
        plt.axis('off')
        plt.savefig(os.path.join(self.output_dir, 'causal_graph.png'), dpi=300, bbox_inches='tight')
        plt.close()
```

### 2. Advanced Physiological Signal Processing

Implement advanced techniques for extracting meaningful features from physiological signals.

```python
import neurokit2 as nk
import pyhrv
from entropy import sample_entropy, permutation_entropy

class PhysiologicalFeatureExtractor:
    def __init__(self, output_dir='output/physio_features'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def extract_advanced_hrv_features(self, hrv_data):
        """Extract advanced HRV features."""
        # Reshape if needed
        if len(hrv_data.shape) > 2:
            hrv_data = hrv_data.reshape(hrv_data.shape[0], -1)
        
        n_samples = hrv_data.shape[0]
        features = np.zeros((n_samples, 15))
        
        for i in range(n_samples):
            # Simulate RR intervals from HRV data
            rr_intervals = self._simulate_rr_intervals(hrv_data[i])
            
            # Time domain features
            time_features = pyhrv.time_domain.time_domain(rr_intervals, plot=False)
            
            # Frequency domain features
            freq_features = pyhrv.frequency_domain.frequency_domain(rr_intervals, plot=False)
            
            # Non-linear features
            nonlinear_features = pyhrv.nonlinear.nonlinear(rr_intervals, plot=False)
            
            # Entropy measures
            sampen = sample_entropy(rr_intervals, m=2, r=0.2*np.std(rr_intervals))
            permen = permutation_entropy(rr_intervals, order=3, delay=1)
            
            # Combine features
            features[i, 0] = time_features['rmssd']  # Root mean square of successive differences
            features[i, 1] = time_features['sdnn']   # Standard deviation of NN intervals
            features[i, 2] = time_features['pnn50']  # Proportion of NN50
            features[i, 3] = freq_features['lf/hf']  # LF/HF ratio
            features[i, 4] = freq_features['vlf']    # Very low frequency power
            features[i, 5] = freq_features['lf']     # Low frequency power
            features[i, 6] = freq_features['hf']     # High frequency power
            features[i, 7] = nonlinear_features['sd1']  # Poincaré plot SD1
            features[i, 8] = nonlinear_features['sd2']  # Poincaré plot SD2
            features[i, 9] = nonlinear_features['sd1/sd2']  # SD1/SD2 ratio
            features[i, 10] = nonlinear_features['dfa_alpha1']  # DFA alpha1
            features[i, 11] = nonlinear_features['dfa_alpha2']  # DFA alpha2
            features[i, 12] = sampen  # Sample entropy
            features[i, 13] = permen  # Permutation entropy
            features[i, 14] = self._calculate_hrv_complexity(rr_intervals)  # HRV complexity
        
        # Plot feature distributions
        self._plot_feature_distributions(features)
        
        return features
    
    def extract_circadian_features(self, sleep_data, activity_data):
        """Extract circadian rhythm features."""
        # Reshape if needed
        if len(sleep_data.shape) > 2:
            sleep_data = sleep_data.reshape(sleep_data.shape[0], -1)
        if len(activity_data.shape) > 2:
            activity_data = activity_data.reshape(activity_data.shape[0], -1)
        
        n_samples = sleep_data.shape[0]
        features = np.zeros((n_samples, 8))
        
        for i in range(n_samples):
            # Simulate 24-hour data
            sleep_24h = self._simulate_24h_data(sleep_data[i])
            activity_24h = self._simulate_24h_data(activity_data[i])
            
            # Calculate cosinor parameters
            sleep_cosinor = nk.signal_cosinor(sleep_24h, sampling_rate=24, plot=False)
            activity_cosinor = nk.signal_cosinor(activity_24h, sampling_rate=24, plot=False)
            
            # Extract features
            features[i, 0] = sleep_cosinor['Mesor']  # Rhythm-adjusted mean
            features[i, 1] = sleep_cosinor['Amplitude']  # Amplitude
            features[i, 2] = sleep_cosinor['Acrophase']  # Phase
            features[i, 3] = sleep_cosinor['Power']  # Power
            features[i, 4] = activity_cosinor['Mesor']
            features[i, 5] = activity_cosinor['Amplitude']
            features[i, 6] = activity_cosinor['Acrophase']
            features[i, 7] = activity_cosinor['Power']
        
        # Plot feature distributions
        self._plot_feature_distributions(features)
        
        return features
    
    def _simulate_rr_intervals(self, hrv_data):
        """Simulate RR intervals from HRV data."""
        # This is a placeholder - in a real implementation, you would use actual RR interval data
        # or convert from raw ECG/PPG signals
        mean_hr = 60 + 20 * hrv_data[0]  # Simulate mean heart rate
        hrv_std = 5 + 3 * hrv_data[1]    # Simulate HRV standard deviation
        
        # Generate RR intervals (in ms)
        rr_intervals = np.random.normal(60000/mean_hr, hrv_std, size=300)
        return rr_intervals
    
    def _simulate_24h_data(self, data):
        """Simulate 24-hour data from feature vector."""
        # This is a placeholder - in a real implementation, you would use actual 24-hour data
        t = np.linspace(0, 24, 24)
        amplitude = 0.5 + 0.3 * data[0]
        phase = 12 + 2 * data[1]
        mesor = 0.5 + 0.2 * data[2]
        
        # Generate 24-hour rhythm
        signal = mesor + amplitude * np.cos(2*np.pi*(t - phase)/24)
        return signal
    
    def _calculate_hrv_complexity(self, rr_intervals):
        """Calculate HRV complexity using multiscale entropy."""
        # Compute multiscale entropy (MSE)
        scales = range(1, 5)
        mse = np.zeros(len(scales))
        
        for i, scale in enumerate(scales):
            # Coarse-grain the time series
            coarse_grained = self._coarse_grain(rr_intervals, scale)
            
            # Calculate sample entropy
            mse[i] = sample_entropy(coarse_grained, m=2, r=0.2*np.std(coarse_grained))
        
        # Return area under MSE curve as complexity measure
        return np.trapz(mse, scales)
    
    def _coarse_grain(self, time_series, scale):
        """Perform coarse-graining for multiscale entropy."""
        n = len(time_series)
        coarse = np.zeros(n // scale)
        
        for i in range(n // scale):
            coarse[i] = np.mean(time_series[i*scale:(i+1)*scale])
            
        return coarse
    
    def _plot_feature_distributions(self, features):
        """Plot feature distributions."""
        n_features = features.shape[1]
        
        plt.figure(figsize=(15, 10))
        for i in range(n_features):
            plt.subplot(3, 5, i+1)
            plt.hist(features[:, i], bins=20, alpha=0.7)
            plt.title(f"Feature {i+1}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_distributions.png'), dpi=300)
        plt.close()
```

### 3. Cross-Modal Feature Fusion

Implement techniques for combining features from different modalities.

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

class CrossModalFeatureFusion:
    def __init__(self, output_dir='output/fusion_features'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def early_fusion(self, data_dict):
        """Perform early fusion of features."""
        # Extract data
        X_sleep = data_dict['X_sleep']
        X_weather = data_dict['X_weather']
        X_stress_diet = data_dict['X_stress_diet']
        X_physio = data_dict['X_physio']
        
        # Flatten 3D features if needed
        if len(X_sleep.shape) > 2:
            X_sleep = X_sleep.reshape(X_sleep.shape[0], -1)
        if len(X_stress_diet.shape) > 2:
            X_stress_diet = X_stress_diet.reshape(X_stress_diet.shape[0], -1)
        
        # Combine all features
        X_combined = np.hstack([X_sleep, X_weather, X_stress_diet, X_physio])
        
        # Apply dimensionality reduction
        pca = PCA(n_components=min(50, X_combined.shape[1]))
        X_fused = pca.fit_transform(X_combined)
        
        # Plot explained variance
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Explained Variance')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'pca_variance.png'), dpi=300)
        plt.close()
        
        return X_fused
    
    def create_fusion_model(self, input_shapes):
        """Create a neural network model for feature fusion."""
        # Create input layers
        sleep_input = layers.Input(shape=input_shapes['sleep'], name='sleep_input')
        weather_input = layers.Input(shape=input_shapes['weather'], name='weather_input')
        stress_diet_input = layers.Input(shape=input_shapes['stress_diet'], name='stress_diet_input')
        physio_input = layers.Input(shape=input_shapes['physio'], name='physio_input')
        
        # Process sleep data
        sleep_features = layers.Conv1D(32, 3, activation='relu')(sleep_input)
        sleep_features = layers.MaxPooling1D(2)(sleep_features)
        sleep_features = layers.Flatten()(sleep_features)
        sleep_features = layers.Dense(32, activation='relu')(sleep_features)
        
        # Process weather data
        weather_features = layers.Dense(16, activation='relu')(weather_input)
        
        # Process stress/diet data
        stress_diet_features = layers.Conv1D(32, 3, activation='relu')(stress_diet_input)
        stress_diet_features = layers.MaxPooling1D(2)(stress_diet_features)
        stress_diet_features = layers.Flatten()(stress_diet_features)
        stress_diet_features = layers.Dense(32, activation='relu')(stress_diet_features)
        
        # Process physio data
        physio_features = layers.Dense(16, activation='relu')(physio_input)
        
        # Attention-based fusion
        sleep_attention = layers.Dense(1, activation='tanh')(sleep_features)
        weather_attention = layers.Dense(1, activation='tanh')(weather_features)
        stress_diet_attention = layers.Dense(1, activation='tanh')(stress_diet_features)
        physio_attention = layers.Dense(1, activation='tanh')(physio_features)
        
        attention_scores = layers.Concatenate()(
            [sleep_attention, weather_attention, stress_diet_attention, physio_attention]
        )
        attention_weights = layers.Softmax()(attention_scores)
        
        # Apply attention weights
        sleep_weighted = layers.Multiply()([sleep_features, attention_weights[:, 0:1]])
        weather_weighted = layers.Multiply()([weather_features, attention_weights[:, 1:2]])
        stress_diet_weighted = layers.Multiply()([stress_diet_features, attention_weights[:, 2:3]])
        physio_weighted = layers.Multiply()([physio_features, attention_weights[:, 3:4]])
        
        # Concatenate weighted features
        fused_features = layers.Concatenate()(
            [sleep_weighted, weather_weighted, stress_diet_weighted, physio_weighted]
        )
        
        # Final processing
        fused_features = layers.Dense(64, activation='relu')(fused_features)
        fused_features = layers.Dropout(0.3)(fused_features)
        output = layers.Dense(1, activation='sigmoid')(fused_features)
        
        # Create model
        model = Model(
            inputs=[sleep_input, weather_input, stress_diet_input, physio_input],
            outputs=output,
            name='fusion_model'
        )
        
        return model
    
    def hybrid_fusion(self, train_data, val_data, test_data):
        """Perform hybrid fusion using neural network."""
        # Define input shapes
        input_shapes = {
            'sleep': train_data['X_sleep'].shape[1:],
            'weather': train_data['X_weather'].shape[1:],
            'stress_diet': train_data['X_stress_diet'].shape[1:],
            'physio': train_data['X_physio'].shape[1:]
        }
        
        # Create fusion model
        model = self.create_fusion_model(input_shapes)
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = model.fit(
            [train_data['X_sleep'], train_data['X_weather'], 
             train_data['X_stress_diet'], train_data['X_physio']],
            train_data['y'],
            validation_data=(
                [val_data['X_sleep'], val_data['X_weather'], 
                 val_data['X_stress_diet'], val_data['X_physio']],
                val_data['y']
            ),
            epochs=50,
            batch_size=32,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ]
        )
        
        # Plot training history
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.title('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fusion_training.png'), dpi=300)
        plt.close()
        
        # Extract fused features from intermediate layer
        feature_extractor = Model(
            inputs=model.inputs,
            outputs=model.get_layer('concatenate_1').output
        )
        
        # Extract fused features
        train_fused = feature_extractor.predict(
            [train_data['X_sleep'], train_data['X_weather'], 
             train_data['X_stress_diet'], train_data['X_physio']]
        )
        
        val_fused = feature_extractor.predict(
            [val_data['X_sleep'], val_data['X_weather'], 
             val_data['X_stress_diet'], val_data['X_physio']]
        )
        
        test_fused = feature_extractor.predict(
            [test_data['X_sleep'], test_data['X_weather'], 
             test_data['X_stress_diet'], test_data['X_physio']]
        )
        
        return {
            'train_fused': train_fused,
            'val_fused': val_fused,
            'test_fused': test_fused,
            'model': model,
            'feature_extractor': feature_extractor
        }
```

### 4. Temporal Pattern Extraction

Implement techniques for extracting temporal patterns from time series data.

```python
import tslearn
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.metrics import dtw

class TemporalPatternExtractor:
    def __init__(self, output_dir='output/temporal_patterns'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def extract_temporal_patterns(self, time_series_data, n_clusters=5):
        """Extract temporal patterns using time series clustering."""
        # Reshape if needed
        if len(time_series_data.shape) < 3:
            # Assume shape is (n_samples, n_features)
            # Reshape to (n_samples, n_timesteps, 1)
            n_samples, n_features = time_series_data.shape
            n_timesteps = n_features
            time_series_data = time_series_data.reshape(n_samples, n_timesteps, 1)
        
        # Normalize time series
        scaler = TimeSeriesScalerMeanVariance()
        time_series_scaled = scaler.fit_transform(time_series_data)
        
        # Perform time series clustering
        km = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=42)
        labels = km.fit_predict(time_series_scaled)
        
        # Get cluster centers
        centers = km.cluster_centers_
        
        # Plot cluster centers
        plt.figure(figsize=(15, 10))
        for i in range(n_clusters):
            plt.subplot(n_clusters, 1, i+1)
            plt.plot(centers[i].ravel())
            plt.title(f"Cluster {i+1}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'cluster_centers.png'), dpi=300)
        plt.close()
        
        # Create features based on cluster distances
        n_samples = time_series_data.shape[0]
        features = np.zeros((n_samples, n_clusters))
        
        for i in range(n_samples):
            for j in range(n_clusters):
                # Calculate DTW distance to each cluster center
                distance = dtw(time_series_scaled[i], centers[j])
                features[i, j] = distance
        
        # Plot feature distributions
        plt.figure(figsize=(15, 5))
        for i in range(n_clusters):
            plt.subplot(1, n_clusters, i+1)
            plt.hist(features[:, i], bins=20, alpha=0.7)
            plt.title(f"Distance to Cluster {i+1}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'cluster_distances.png'), dpi=300)
        plt.close()
        
        return features, labels, centers
    
    def extract_change_points(self, time_series_data, penalty=1.0):
        """Extract change points in time series data."""
        from ruptures import Pelt
        
        # Reshape if needed
        if len(time_series_data.shape) > 2:
            # Assume shape is (n_samples, n_timesteps, n_features)
            # Reshape to (n_samples, n_timesteps * n_features)
            n_samples = time_series_data.shape[0]
            time_series_data = time_series_data.reshape(n_samples, -1)
        
        n_samples, n_features = time_series_data.shape
        features = np.zeros((n_samples, 5))
        
        for i in range(n_samples):
            # Get time series
            signal = time_series_data[i]
            
            # Detect change points
            algo = Pelt(model="rbf").fit(signal.reshape(-1, 1))
            change_points = algo.predict(pen=penalty)
            
            # Calculate features
            if len(change_points) > 1:  # Exclude the last point which is always the length
                change_points = change_points[:-1]
                features[i, 0] = len(change_points)  # Number of change points
                features[i, 1] = np.mean(np.diff(change_points))  # Mean segment length
                features[i, 2] = np.std(np.diff(change_points))   # Std of segment length
                features[i, 3] = change_points[0]  # First change point
                features[i, 4] = change_points[-1] if len(change_points) > 0 else 0  # Last change point
            
        # Plot feature distributions
        plt.figure(figsize=(15, 5))
        feature_names = ['Num Changes', 'Mean Segment', 'Std Segment', 'First Change', 'Last Change']
        for i in range(5):
            plt.subplot(1, 5, i+1)
            plt.hist(features[:, i], bins=20, alpha=0.7)
            plt.title(feature_names[i])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'change_point_features.png'), dpi=300)
        plt.close()
        
        return features
```

## Integration with Existing Feature Engineering

These advanced techniques can be integrated with the existing feature engineering pipeline:

```python
class EnhancedFeatureEngineer:
    def __init__(self, output_dir='output/enhanced_features'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize specialized feature extractors
        self.causal_discovery = CausalFeatureDiscovery(output_dir=os.path.join(output_dir, 'causal'))
        self.physio_extractor = PhysiologicalFeatureExtractor(output_dir=os.path.join(output_dir, 'physio'))
        self.fusion = CrossModalFeatureFusion(output_dir=os.path.join(output_dir, 'fusion'))
        self.temporal_extractor = TemporalPatternExtractor(output_dir=os.path.join(output_dir, 'temporal'))
        
    def enhance_features(self, train_data, val_data, test_data):
        """Apply all feature enhancement techniques."""
        print("Applying advanced feature engineering techniques...")
        
        # 1. Extract causal features
        print("\nDiscovering causal features...")
        causal_features_train, direct_causes = self.causal_discovery.discover_causal_features(train_data)
        
        # Apply same transformation to validation and test data
        X_combined_val = self.causal_discovery._combine_features(val_data)
        X_combined_test = self.causal_discovery._combine_features(test_data)
        causal_features_val = X_combined_val[:, direct_causes]
        causal_features_test = X_combined_test[:, direct_causes]
        
        # 2. Extract advanced physiological features
        print("\nExtracting advanced physiological features...")
        physio_features_train = self.physio_extractor.extract_advanced_hrv_features(train_data['X_physio'])
        physio_features_val = self.physio_extractor.extract_advanced_hrv_features(val_data['X_physio'])
        physio_features_test = self.physio_extractor.extract_advanced_hrv_features(test_data['X_physio'])
        
        # 3. Extract circadian features
        print("\nExtracting circadian features...")
        circadian_features_train = self.physio_extractor.extract_circadian_features(
            train_data['X_sleep'], train_data['X_physio'])
        circadian_features_val = self.physio_extractor.extract_circadian_features(
            val_data['X_sleep'], val_data['X_physio'])
        circadian_features_test = self.physio_extractor.extract_circadian_features(
            test_data['X_sleep'], test_data['X_physio'])
        
        # 4. Extract temporal patterns
        print("\nExtracting temporal patterns...")
        temporal_features_train, _, _ = self.temporal_extractor.extract_temporal_patterns(train_data['X_sleep'])
        temporal_features_val, _, _ = self.temporal_extractor.extract_temporal_patterns(val_data['X_sleep'])
        temporal_features_test, _, _ = self.temporal_extractor.extract_temporal_patterns(test_data['X_sleep'])
        
        # 5. Extract change points
        print("\nExtracting change points...")
        change_features_train = self.temporal_extractor.extract_change_points(train_data['X_stress_diet'])
        change_features_val = self.temporal_extractor.extract_change_points(val_data['X_stress_diet'])
        change_features_test = self.temporal_extractor.extract_change_points(test_data['X_stress_diet'])
        
        # 6. Perform hybrid fusion
        print("\nPerforming hybrid feature fusion...")
        fusion_results = self.fusion.hybrid_fusion(train_data, val_data, test_data)
        
        # Combine all enhanced features
        enhanced_train_features = np.hstack([
            causal_features_train,
            physio_features_train,
            circadian_features_train,
            temporal_features_train,
            change_features_train,
            fusion_results['train_fused']
        ])
        
        enhanced_val_features = np.hstack([
            causal_features_val,
            physio_features_val,
            circadian_features_val,
            temporal_features_val,
            change_features_val,
            fusion_results['val_fused']
        ])
        
        enhanced_test_features = np.hstack([
            causal_features_test,
            physio_features_test,
            circadian_features_test,
            temporal_features_test,
            change_features_test,
            fusion_results['test_fused']
        ])
        
        # Create enhanced data dictionaries
        enhanced_train_data = {
            'X': enhanced_train_features,
            'y': train_data['y'],
            'X_original': train_data
        }
        
        enhanced_val_data = {
            'X': enhanced_val_features,
            'y': val_data['y'],
            'X_original': val_data
        }
        
        enhanced_test_data = {
            'X': enhanced_test_features,
            'y': test_data['y'],
            'X_original': test_data
        }
        
        print(f"\nEnhanced feature dimensions:")
        print(f"Train: {enhanced_train_features.shape}")
        print(f"Validation: {enhanced_val_features.shape}")
        print(f"Test: {enhanced_test_features.shape}")
        
        return enhanced_train_data, enhanced_val_data, enhanced_test_data
```

## Expected Benefits

1. **Improved Causality**: Causal feature discovery helps identify true migraine triggers rather than mere correlations.
2. **Better Physiological Insights**: Advanced HRV and circadian rhythm analysis captures subtle physiological patterns.
3. **Enhanced Pattern Recognition**: Temporal pattern extraction identifies recurring patterns that precede migraines.
4. **Improved Feature Integration**: Cross-modal fusion techniques better capture interactions between different data types.

## Performance Expectations

We expect the following improvements from these advanced feature engineering techniques:
- AUC: Increase from 0.9325 to 0.95+ (2% improvement)
- F1 Score: Increase from 0.8659 to 0.88+ (1.5% improvement)
- Precision: Increase from 0.8571 to 0.87+ (1.5% improvement)
- Recall: Increase from 0.8750 to 0.89+ (1.5% improvement)

## Implementation Timeline

1. Causal Feature Discovery Implementation: 1 week
2. Advanced Physiological Signal Processing: 1 week
3. Cross-Modal Feature Fusion: 1 week
4. Temporal Pattern Extraction: 1 week
5. Integration and Testing: 1 week

Total estimated time: 5 weeks
