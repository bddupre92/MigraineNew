import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

# Import our optimization modules
from model.threshold_optimization import ThresholdOptimizer
from model.class_balancing import ClassBalancer
from model.feature_engineering import FeatureEngineer
from model.ensemble_methods import EnsembleModels

# Create output directories
os.makedirs('output/models', exist_ok=True)
os.makedirs('output/results', exist_ok=True)
os.makedirs('output/evaluation', exist_ok=True)

def load_data(data_dir='output/data'):
    """
    Load the generated data.
    
    Args:
        data_dir (str): Directory containing the data files
        
    Returns:
        tuple: Train, validation, and test data
    """
    print("Loading data from", data_dir)
    
    # Load training data
    train_data = np.load(os.path.join(data_dir, 'train_data.npz'))
    X_sleep_train = train_data['X_sleep']
    X_weather_train = train_data['X_weather']
    X_stress_diet_train = train_data['X_stress_diet']
    X_physio_train = train_data['X_physio']
    y_train = train_data['y']
    
    # Load validation data
    val_data = np.load(os.path.join(data_dir, 'val_data.npz'))
    X_sleep_val = val_data['X_sleep']
    X_weather_val = val_data['X_weather']
    X_stress_diet_val = val_data['X_stress_diet']
    X_physio_val = val_data['X_physio']
    y_val = val_data['y']
    
    # Load test data
    test_data = np.load(os.path.join(data_dir, 'test_data.npz'))
    X_sleep_test = test_data['X_sleep']
    X_weather_test = test_data['X_weather']
    X_stress_diet_test = test_data['X_stress_diet']
    X_physio_test = test_data['X_physio']
    y_test = test_data['y']
    
    # Create data dictionaries
    train_data_dict = {
        'X_sleep': X_sleep_train,
        'X_weather': X_weather_train,
        'X_stress_diet': X_stress_diet_train,
        'X_physio': X_physio_train,
        'y': y_train
    }
    
    val_data_dict = {
        'X_sleep': X_sleep_val,
        'X_weather': X_weather_val,
        'X_stress_diet': X_stress_diet_val,
        'X_physio': X_physio_val,
        'y': y_val
    }
    
    test_data_dict = {
        'X_sleep': X_sleep_test,
        'X_weather': X_weather_test,
        'X_stress_diet': X_stress_diet_test,
        'X_physio': X_physio_test,
        'y': y_test
    }
    
    print(f"Train set: {len(y_train)} samples, positive ratio: {np.mean(y_train):.2f}")
    print(f"Validation set: {len(y_val)} samples, positive ratio: {np.mean(y_val):.2f}")
    print(f"Test set: {len(y_test)} samples, positive ratio: {np.mean(y_test):.2f}")
    
    return train_data_dict, val_data_dict, test_data_dict

def enhance_features(train_data, val_data, test_data):
    """
    Enhance features using the FeatureEngineer.
    
    Args:
        train_data (dict): Training data dictionary
        val_data (dict): Validation data dictionary
        test_data (dict): Test data dictionary
        
    Returns:
        tuple: Enhanced train, validation, and test data
    """
    print("\n=== Enhancing Features ===")
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer(output_dir='output/feature_engineering')
    
    # Enhance sleep features
    print("\nEnhancing sleep features...")
    X_sleep_train_enhanced = feature_engineer.enhance_sleep_features(train_data['X_sleep'])
    X_sleep_val_enhanced = feature_engineer.enhance_sleep_features(val_data['X_sleep'])
    X_sleep_test_enhanced = feature_engineer.enhance_sleep_features(test_data['X_sleep'])
    
    # Enhance weather features
    print("\nEnhancing weather features...")
    X_weather_train_enhanced = feature_engineer.enhance_weather_features(train_data['X_weather'].reshape(train_data['X_weather'].shape[0], 1, -1))
    X_weather_val_enhanced = feature_engineer.enhance_weather_features(val_data['X_weather'].reshape(val_data['X_weather'].shape[0], 1, -1))
    X_weather_test_enhanced = feature_engineer.enhance_weather_features(test_data['X_weather'].reshape(test_data['X_weather'].shape[0], 1, -1))
    
    # Enhance stress/diet features
    print("\nEnhancing stress/diet features...")
    X_stress_diet_train_enhanced = feature_engineer.enhance_stress_diet_features(train_data['X_stress_diet'])
    X_stress_diet_val_enhanced = feature_engineer.enhance_stress_diet_features(val_data['X_stress_diet'])
    X_stress_diet_test_enhanced = feature_engineer.enhance_stress_diet_features(test_data['X_stress_diet'])
    
    # Enhance physiological features
    print("\nEnhancing physiological features...")
    X_physio_train_enhanced = feature_engineer.enhance_physio_features(train_data['X_physio'].reshape(train_data['X_physio'].shape[0], 1, -1))
    X_physio_val_enhanced = feature_engineer.enhance_physio_features(val_data['X_physio'].reshape(val_data['X_physio'].shape[0], 1, -1))
    X_physio_test_enhanced = feature_engineer.enhance_physio_features(test_data['X_physio'].reshape(test_data['X_physio'].shape[0], 1, -1))
    
    # Create enhanced data dictionaries
    enhanced_train_data = {
        'X_sleep': X_sleep_train_enhanced,
        'X_weather': X_weather_train_enhanced,
        'X_stress_diet': X_stress_diet_train_enhanced,
        'X_physio': X_physio_train_enhanced,
        'y': train_data['y']
    }
    
    enhanced_val_data = {
        'X_sleep': X_sleep_val_enhanced,
        'X_weather': X_weather_val_enhanced,
        'X_stress_diet': X_stress_diet_val_enhanced,
        'X_physio': X_physio_val_enhanced,
        'y': val_data['y']
    }
    
    enhanced_test_data = {
        'X_sleep': X_sleep_test_enhanced,
        'X_weather': X_weather_test_enhanced,
        'X_stress_diet': X_stress_diet_test_enhanced,
        'X_physio': X_physio_test_enhanced,
        'y': test_data['y']
    }
    
    print("\nFeature enhancement completed.")
    print(f"Enhanced sleep features shape: {X_sleep_train_enhanced.shape}")
    print(f"Enhanced weather features shape: {X_weather_train_enhanced.shape}")
    print(f"Enhanced stress/diet features shape: {X_stress_diet_train_enhanced.shape}")
    print(f"Enhanced physiological features shape: {X_physio_train_enhanced.shape}")
    
    return enhanced_train_data, enhanced_val_data, enhanced_test_data

def apply_class_balancing(train_data):
    """
    Apply class balancing techniques to the training data.
    
    Args:
        train_data (dict): Training data dictionary
        
    Returns:
        dict: Balanced training data
    """
    print("\n=== Applying Class Balancing ===")
    
    # Initialize class balancer
    class_balancer = ClassBalancer(output_dir='output/class_balancing')
    
    # Combine all features for SMOTE
    X_sleep = train_data['X_sleep']
    X_weather = train_data['X_weather']
    X_stress_diet = train_data['X_stress_diet']
    X_physio = train_data['X_physio']
    y = train_data['y']
    
    # Flatten 2D features if needed
    X_combined = np.hstack([
        X_sleep,
        X_weather,
        X_stress_diet,
        X_physio
    ])
    
    # Plot original class distribution
    class_balancer.plot_class_distribution(y, save_path='output/class_balancing/original_distribution.png')
    
    # Apply advanced SMOTE (BorderlineSMOTE)
    print("\nApplying BorderlineSMOTE...")
    X_combined_balanced, y_balanced = class_balancer.apply_advanced_smote(
        X_combined, y, method='borderline', sampling_strategy=0.5
    )
    
    # Plot balanced class distribution
    class_balancer.plot_class_distribution(y, y_balanced, save_path='output/class_balancing/balanced_distribution.png')
    
    # Calculate class weights for the balanced data
    class_weights = class_balancer.calculate_class_weights(y_balanced)
    
    # Split back into separate feature types
    sleep_size = X_sleep.shape[1]
    weather_size = X_weather.shape[1]
    stress_diet_size = X_stress_diet.shape[1]
    physio_size = X_physio.shape[1]
    
    X_sleep_balanced = X_combined_balanced[:, :sleep_size]
    X_weather_balanced = X_combined_balanced[:, sleep_size:sleep_size+weather_size]
    X_stress_diet_balanced = X_combined_balanced[:, sleep_size+weather_size:sleep_size+weather_size+stress_diet_size]
    X_physio_balanced = X_combined_balanced[:, sleep_size+weather_size+stress_diet_size:]
    
    # Create balanced data dictionary
    balanced_train_data = {
        'X_sleep': X_sleep_balanced,
        'X_weather': X_weather_balanced,
        'X_stress_diet': X_stress_diet_balanced,
        'X_physio': X_physio_balanced,
        'y': y_balanced,
        'class_weights': class_weights
    }
    
    print(f"Original training data: {len(y)} samples, positive ratio: {np.mean(y):.2f}")
    print(f"Balanced training data: {len(y_balanced)} samples, positive ratio: {np.mean(y_balanced):.2f}")
    
    return balanced_train_data

def create_expert_models(train_data, val_data):
    """
    Create and train expert models for each data type.
    
    Args:
        train_data (dict): Training data dictionary
        val_data (dict): Validation data dictionary
        
    Returns:
        dict: Dictionary of trained expert models
    """
    print("\n=== Creating Expert Models ===")
    
    # Extract data
    X_sleep_train = train_data['X_sleep']
    X_weather_train = train_data['X_weather']
    X_stress_diet_train = train_data['X_stress_diet']
    X_physio_train = train_data['X_physio']
    y_train = train_data['y']
    class_weights = train_data.get('class_weights')
    
    X_sleep_val = val_data['X_sleep']
    X_weather_val = val_data['X_weather']
    X_stress_diet_val = val_data['X_stress_diet']
    X_physio_val = val_data['X_physio']
    y_val = val_data['y']
    
    # Create callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.0001
    )
    
    # Create and train sleep expert
    print("\nCreating and training sleep expert...")
    sleep_expert = models.Sequential(name='sleep_expert')
    sleep_expert.add(layers.Input(shape=(X_sleep_train.shape[1],)))
    sleep_expert.add(layers.Dense(64, activation='relu'))
    sleep_expert.add(layers.BatchNormalization())
    sleep_expert.add(layers.Dropout(0.3))
    sleep_expert.add(layers.Dense(32, activation='relu'))
    sleep_expert.add(layers.BatchNormalization())
    sleep_expert.add(layers.Dropout(0.3))
    sleep_expert.add(layers.Dense(1, activation='sigmoid'))
    
    sleep_expert.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    sleep_history = sleep_expert.fit(
        X_sleep_train, y_train,
        validation_data=(X_sleep_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weights,
        verbose=0
    )
    
    # Create and train weather expert
    print("Creating and training weather expert...")
    weather_expert = models.Sequential(name='weather_expert')
    weather_expert.add(layers.Input(shape=(X_weather_train.shape[1],)))
    weather_expert.add(layers.Dense(32, activation='relu'))
    weather_expert.add(layers.BatchNormalization())
    weather_expert.add(layers.Dropout(0.3))
    weather_expert.add(layers.Dense(16, activation='relu'))
    weather_expert.add(layers.BatchNormalization())
    weather_expert.add(layers.Dropout(0.3))
    weather_expert.add(layers.Dense(1, activation='sigmoid'))
    
    weather_expert.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    weather_history = weather_expert.fit(
        X_weather_train, y_train,
        validation_data=(X_weather_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weights,
        verbose=0
    )
    
    # Create and train stress/diet expert
    print("Creating and training stress/diet expert...")
    stress_diet_expert = models.Sequential(name='stress_diet_expert')
    stress_diet_expert.add(layers.Input(shape=(X_stress_diet_train.shape[1],)))
    stress_diet_expert.add(layers.Dense(64, activation='relu'))
    stress_diet_expert.add(layers.BatchNormalization())
    stress_diet_expert.add(layers.Dropout(0.3))
    stress_diet_expert.add(layers.Dense(32, activation='relu'))
    stress_diet_expert.add(layers.BatchNormalization())
    stress_diet_expert.add(layers.Dropout(0.3))
    stress_diet_expert.add(layers.Dense(1, activation='sigmoid'))
    
    stress_diet_expert.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    stress_diet_history = stress_diet_expert.fit(
        X_stress_diet_train, y_train,
        validation_data=(X_stress_diet_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weights,
        verbose=0
    )
    
    # Create and train physio expert
    print("Creating and training physiological expert...")
    physio_expert = models.Sequential(name='physio_expert')
    physio_expert.add(layers.Input(shape=(X_physio_train.shape[1],)))
    physio_expert.add(layers.Dense(32, activation='relu'))
    physio_expert.add(layers.BatchNormalization())
    physio_expert.add(layers.Dropout(0.3))
    physio_expert.add(layers.Dense(16, activation='relu'))
    physio_expert.add(layers.BatchNormalization())
    physio_expert.add(layers.Dropout(0.3))
    physio_expert.add(layers.Dense(1, activation='sigmoid'))
    
    physio_expert.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    physio_history = physio_expert.fit(
        X_physio_train, y_train,
        validation_data=(X_physio_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weights,
        verbose=0
    )
    
    # Save expert models
    sleep_expert.save('output/models/sleep_expert.keras')
    weather_expert.save('output/models/weather_expert.keras')
    stress_diet_expert.save('output/models/stress_diet_expert.keras')
    physio_expert.save('output/models/physio_expert.keras')
    
    # Evaluate expert models
    print("\nExpert Model Evaluation:")
    
    # Sleep expert
    sleep_val_loss, sleep_val_acc = sleep_expert.evaluate(X_sleep_val, y_val, verbose=0)
    sleep_val_pred = sleep_expert.predict(X_sleep_val, verbose=0)
    sleep_val_auc = roc_auc_score(y_val, sleep_val_pred)
    print(f"Sleep Expert - Val Loss: {sleep_val_loss:.4f}, Val Acc: {sleep_val_acc:.4f}, Val AUC: {sleep_val_auc:.4f}")
    
    # Weather expert
    weather_val_loss, weather_val_acc = weather_expert.evaluate(X_weather_val, y_val, verbose=0)
    weather_val_pred = weather_expert.predict(X_weather_val, verbose=0)
    weather_val_auc = roc_auc_score(y_val, weather_val_pred)
    print(f"Weather Expert - Val Loss: {weather_val_loss:.4f}, Val Acc: {weather_val_acc:.4f}, Val AUC: {weather_val_auc:.4f}")
    
    # Stress/diet expert
    stress_diet_val_loss, stress_diet_val_acc = stress_diet_expert.evaluate(X_stress_diet_val, y_val, verbose=0)
    stress_diet_val_pred = stress_diet_expert.predict(X_stress_diet_val, verbose=0)
    stress_diet_val_auc = roc_auc_score(y_val, stress_diet_val_pred)
    print(f"Stress/Diet Expert - Val Loss: {stress_diet_val_loss:.4f}, Val Acc: {stress_diet_val_acc:.4f}, Val AUC: {stress_diet_val_auc:.4f}")
    
    # Physio expert
    physio_val_loss, physio_val_acc = physio_expert.evaluate(X_physio_val, y_val, verbose=0)
    physio_val_pred = physio_expert.predict(X_physio_val, verbose=0)
    physio_val_auc = roc_auc_score(y_val, physio_val_pred)
    print(f"Physio Expert - Val Loss: {physio_val_loss:.4f}, Val Acc: {physio_val_acc:.4f}, Val AUC: {physio_val_auc:.4f}")
    
    # Create expert models dictionary
    expert_models = {
        'sleep_expert': sleep_expert,
        'weather_expert': weather_expert,
        'stress_diet_expert': stress_diet_expert,
        'physio_expert': physio_expert
    }
    
    # Create expert histories dictionary
    expert_histories = {
        'sleep_expert': sleep_history.history,
        'weather_expert': weather_history.history,
        'stress_diet_expert': stress_diet_history.history,
        'physio_expert': physio_history.history
    }
    
    return expert_models, expert_histories

def create_ensemble_model(expert_models, train_data, val_data):
    """
    Create and train an ensemble model using the expert models.
    
    Args:
        expert_models (dict): Dictionary of expert models
        train_data (dict): Training data dictionary
        val_data (dict): Validation data dictionary
        
    Returns:
        tuple: Ensemble model and training history
    """
    print("\n=== Creating Ensemble Model ===")
    
    # Extract expert models
    sleep_expert = expert_models['sleep_expert']
    weather_expert = expert_models['weather_expert']
    stress_diet_expert = expert_models['stress_diet_expert']
    physio_expert = expert_models['physio_expert']
    
    # Extract data
    X_sleep_train = train_data['X_sleep']
    X_weather_train = train_data['X_weather']
    X_stress_diet_train = train_data['X_stress_diet']
    X_physio_train = train_data['X_physio']
    y_train = train_data['y']
    class_weights = train_data.get('class_weights')
    
    X_sleep_val = val_data['X_sleep']
    X_weather_val = val_data['X_weather']
    X_stress_diet_val = val_data['X_stress_diet']
    X_physio_val = val_data['X_physio']
    y_val = val_data['y']
    
    # Get expert predictions
    sleep_train_pred = sleep_expert.predict(X_sleep_train, verbose=0)
    weather_train_pred = weather_expert.predict(X_weather_train, verbose=0)
    stress_diet_train_pred = stress_diet_expert.predict(X_stress_diet_train, verbose=0)
    physio_train_pred = physio_expert.predict(X_physio_train, verbose=0)
    
    sleep_val_pred = sleep_expert.predict(X_sleep_val, verbose=0)
    weather_val_pred = weather_expert.predict(X_weather_val, verbose=0)
    stress_diet_val_pred = stress_diet_expert.predict(X_stress_diet_val, verbose=0)
    physio_val_pred = physio_expert.predict(X_physio_val, verbose=0)
    
    # Combine expert predictions with original features
    X_ensemble_train = np.hstack([
        sleep_train_pred,
        weather_train_pred,
        stress_diet_train_pred,
        physio_train_pred,
        X_sleep_train[:, :10],  # Include top features from each expert
        X_weather_train[:, :5],
        X_stress_diet_train[:, :10],
        X_physio_train[:, :5]
    ])
    
    X_ensemble_val = np.hstack([
        sleep_val_pred,
        weather_val_pred,
        stress_diet_val_pred,
        physio_val_pred,
        X_sleep_val[:, :10],  # Include top features from each expert
        X_weather_val[:, :5],
        X_stress_diet_val[:, :10],
        X_physio_val[:, :5]
    ])
    
    # Create callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.0001
    )
    
    # Create ensemble model
    print("Creating and training ensemble model...")
    ensemble_model = models.Sequential(name='ensemble_model')
    ensemble_model.add(layers.Input(shape=(X_ensemble_train.shape[1],)))
    ensemble_model.add(layers.Dense(128, activation='relu'))
    ensemble_model.add(layers.BatchNormalization())
    ensemble_model.add(layers.Dropout(0.4))
    ensemble_model.add(layers.Dense(64, activation='relu'))
    ensemble_model.add(layers.BatchNormalization())
    ensemble_model.add(layers.Dropout(0.4))
    ensemble_model.add(layers.Dense(32, activation='relu'))
    ensemble_model.add(layers.BatchNormalization())
    ensemble_model.add(layers.Dropout(0.3))
    ensemble_model.add(layers.Dense(1, activation='sigmoid'))
    
    ensemble_model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    ensemble_history = ensemble_model.fit(
        X_ensemble_train, y_train,
        validation_data=(X_ensemble_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weights,
        verbose=0
    )
    
    # Save ensemble model
    ensemble_model.save('output/models/ensemble_model.keras')
    
    # Evaluate ensemble model
    ensemble_val_loss, ensemble_val_acc = ensemble_model.evaluate(X_ensemble_val, y_val, verbose=0)
    ensemble_val_pred = ensemble_model.predict(X_ensemble_val, verbose=0)
    ensemble_val_auc = roc_auc_score(y_val, ensemble_val_pred)
    print(f"Ensemble Model - Val Loss: {ensemble_val_loss:.4f}, Val Acc: {ensemble_val_acc:.4f}, Val AUC: {ensemble_val_auc:.4f}")
    
    # Create ensemble data dictionary for later use
    ensemble_data = {
        'X_train': X_ensemble_train,
        'y_train': y_train,
        'X_val': X_ensemble_val,
        'y_val': y_val
    }
    
    return ensemble_model, ensemble_history, ensemble_data

def optimize_threshold(ensemble_model, ensemble_data):
    """
    Optimize the classification threshold for the ensemble model.
    
    Args:
        ensemble_model (tf.keras.Model): Trained ensemble model
        ensemble_data (dict): Ensemble data dictionary
        
    Returns:
        float: Optimal threshold
    """
    print("\n=== Optimizing Classification Threshold ===")
    
    # Initialize threshold optimizer
    threshold_optimizer = ThresholdOptimizer(output_dir='output/threshold_optimization')
    
    # Get validation predictions
    X_val = ensemble_data['X_val']
    y_val = ensemble_data['y_val']
    y_pred_val = ensemble_model.predict(X_val, verbose=0).flatten()
    
    # Plot precision-recall curve
    threshold_optimizer.plot_precision_recall_curve(
        y_val, y_pred_val,
        save_path='output/threshold_optimization/precision_recall_curve.png'
    )
    
    # Plot threshold metrics
    threshold_optimizer.plot_threshold_metrics(
        y_val, y_pred_val,
        save_path='output/threshold_optimization/threshold_metrics.png'
    )
    
    # Find optimal threshold using different methods
    print("\nFinding optimal thresholds:")
    
    # F1 optimization
    optimal_f1_threshold = threshold_optimizer.find_optimal_threshold(y_val, y_pred_val, method='f1')
    
    # Precision-recall balance
    optimal_pr_threshold = threshold_optimizer.find_optimal_threshold(y_val, y_pred_val, method='precision_recall_balance')
    
    # Cost-based optimization (false negatives are more costly)
    optimal_cost_threshold = threshold_optimizer.find_optimal_threshold(y_val, y_pred_val, method='cost_based')
    
    # Compare thresholds
    thresholds_dict = {
        'Default': 0.5,
        'F1 Optimized': optimal_f1_threshold,
        'PR Balance': optimal_pr_threshold,
        'Cost-Based': optimal_cost_threshold
    }
    
    threshold_results = threshold_optimizer.compare_thresholds(
        y_val, y_pred_val, thresholds_dict,
        save_path='output/threshold_optimization/threshold_comparison.png'
    )
    
    # Choose the F1-optimized threshold as our final threshold
    optimal_threshold = optimal_f1_threshold
    print(f"\nSelected optimal threshold: {optimal_threshold:.4f}")
    
    # Evaluate with optimal threshold
    metrics = threshold_optimizer.evaluate_with_optimal_threshold(y_val, y_pred_val, optimal_threshold)
    
    return optimal_threshold, metrics

def evaluate_model(ensemble_model, expert_models, test_data, optimal_threshold):
    """
    Evaluate the ensemble model and expert models on the test set.
    
    Args:
        ensemble_model (tf.keras.Model): Trained ensemble model
        expert_models (dict): Dictionary of expert models
        test_data (dict): Test data dictionary
        optimal_threshold (float): Optimal classification threshold
        
    Returns:
        dict: Evaluation metrics
    """
    print("\n=== Evaluating Models on Test Set ===")
    
    # Extract expert models
    sleep_expert = expert_models['sleep_expert']
    weather_expert = expert_models['weather_expert']
    stress_diet_expert = expert_models['stress_diet_expert']
    physio_expert = expert_models['physio_expert']
    
    # Extract test data
    X_sleep_test = test_data['X_sleep']
    X_weather_test = test_data['X_weather']
    X_stress_diet_test = test_data['X_stress_diet']
    X_physio_test = test_data['X_physio']
    y_test = test_data['y']
    
    # Get expert predictions
    sleep_test_pred = sleep_expert.predict(X_sleep_test, verbose=0)
    weather_test_pred = weather_expert.predict(X_weather_test, verbose=0)
    stress_diet_test_pred = stress_diet_expert.predict(X_stress_diet_test, verbose=0)
    physio_test_pred = physio_expert.predict(X_physio_test, verbose=0)
    
    # Combine expert predictions with original features
    X_ensemble_test = np.hstack([
        sleep_test_pred,
        weather_test_pred,
        stress_diet_test_pred,
        physio_test_pred,
        X_sleep_test[:, :10],  # Include top features from each expert
        X_weather_test[:, :5],
        X_stress_diet_test[:, :10],
        X_physio_test[:, :5]
    ])
    
    # Get ensemble predictions
    ensemble_test_pred = ensemble_model.predict(X_ensemble_test, verbose=0).flatten()
    
    # Apply optimal threshold
    ensemble_test_pred_binary = (ensemble_test_pred >= optimal_threshold).astype(int)
    
    # Calculate metrics
    ensemble_test_acc = np.mean(ensemble_test_pred_binary == y_test)
    ensemble_test_auc = roc_auc_score(y_test, ensemble_test_pred)
    ensemble_test_precision = precision_score(y_test, ensemble_test_pred_binary)
    ensemble_test_recall = recall_score(y_test, ensemble_test_pred_binary)
    ensemble_test_f1 = f1_score(y_test, ensemble_test_pred_binary)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, ensemble_test_pred_binary)
    
    # Print metrics
    print("\nEnsemble Model Test Metrics:")
    print(f"Accuracy: {ensemble_test_acc:.4f}")
    print(f"AUC: {ensemble_test_auc:.4f}")
    print(f"Precision: {ensemble_test_precision:.4f}")
    print(f"Recall: {ensemble_test_recall:.4f}")
    print(f"F1 Score: {ensemble_test_f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    # Evaluate expert models
    print("\nExpert Models Test Metrics:")
    
    # Sleep expert
    sleep_test_pred_binary = (sleep_test_pred.flatten() >= 0.5).astype(int)
    sleep_test_acc = np.mean(sleep_test_pred_binary == y_test)
    sleep_test_auc = roc_auc_score(y_test, sleep_test_pred)
    print(f"Sleep Expert - Acc: {sleep_test_acc:.4f}, AUC: {sleep_test_auc:.4f}")
    
    # Weather expert
    weather_test_pred_binary = (weather_test_pred.flatten() >= 0.5).astype(int)
    weather_test_acc = np.mean(weather_test_pred_binary == y_test)
    weather_test_auc = roc_auc_score(y_test, weather_test_pred)
    print(f"Weather Expert - Acc: {weather_test_acc:.4f}, AUC: {weather_test_auc:.4f}")
    
    # Stress/diet expert
    stress_diet_test_pred_binary = (stress_diet_test_pred.flatten() >= 0.5).astype(int)
    stress_diet_test_acc = np.mean(stress_diet_test_pred_binary == y_test)
    stress_diet_test_auc = roc_auc_score(y_test, stress_diet_test_pred)
    print(f"Stress/Diet Expert - Acc: {stress_diet_test_acc:.4f}, AUC: {stress_diet_test_auc:.4f}")
    
    # Physio expert
    physio_test_pred_binary = (physio_test_pred.flatten() >= 0.5).astype(int)
    physio_test_acc = np.mean(physio_test_pred_binary == y_test)
    physio_test_auc = roc_auc_score(y_test, physio_test_pred)
    print(f"Physio Expert - Acc: {physio_test_acc:.4f}, AUC: {physio_test_auc:.4f}")
    
    # Save test predictions for dashboard
    np.savez(
        'output/test_predictions.npz',
        y_true=y_test,
        y_pred=ensemble_test_pred,
        sleep_pred=sleep_test_pred.flatten(),
        weather_pred=weather_test_pred.flatten(),
        stress_diet_pred=stress_diet_test_pred.flatten(),
        physio_pred=physio_test_pred.flatten(),
        X_sleep=X_sleep_test,
        X_weather=X_weather_test,
        X_stress_diet=X_stress_diet_test,
        X_physio=X_physio_test,
        optimal_threshold=optimal_threshold
    )
    
    # Create metrics dictionary
    metrics = {
        'ensemble': {
            'accuracy': ensemble_test_acc,
            'auc': ensemble_test_auc,
            'precision': ensemble_test_precision,
            'recall': ensemble_test_recall,
            'f1': ensemble_test_f1,
            'confusion_matrix': cm,
            'threshold': optimal_threshold
        },
        'sleep_expert': {
            'accuracy': sleep_test_acc,
            'auc': sleep_test_auc
        },
        'weather_expert': {
            'accuracy': weather_test_acc,
            'auc': weather_test_auc
        },
        'stress_diet_expert': {
            'accuracy': stress_diet_test_acc,
            'auc': stress_diet_test_auc
        },
        'physio_expert': {
            'accuracy': physio_test_acc,
            'auc': physio_test_auc
        }
    }
    
    return metrics, ensemble_test_pred, y_test

def plot_results(metrics, ensemble_test_pred, y_test, optimal_threshold):
    """
    Plot evaluation results.
    
    Args:
        metrics (dict): Evaluation metrics
        ensemble_test_pred (np.ndarray): Ensemble model predictions
        y_test (np.ndarray): True test labels
        optimal_threshold (float): Optimal classification threshold
    """
    print("\n=== Plotting Results ===")
    
    # Create output directory
    os.makedirs('output/results', exist_ok=True)
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    
    # Calculate ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, ensemble_test_pred)
    roc_auc = metrics['ensemble']['auc']
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig('output/results/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    
    cm = metrics['ensemble']['confusion_matrix']
    
    # Plot confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add labels
    classes = ['No Migraine', 'Migraine']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save plot
    plt.savefig('output/results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot precision-recall curve
    plt.figure(figsize=(10, 8))
    
    # Calculate precision-recall curve
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, thresholds = precision_recall_curve(y_test, ensemble_test_pred)
    avg_precision = average_precision_score(y_test, ensemble_test_pred)
    
    # Plot precision-recall curve
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.4f})')
    
    # Mark optimal threshold
    # Find the closest threshold to our optimal threshold
    idx = np.argmin(np.abs(thresholds - optimal_threshold)) if len(thresholds) > 0 else 0
    if idx < len(precision) and idx < len(recall):
        plt.plot(recall[idx], precision[idx], 'ro', markersize=8)
        plt.annotate(f'Threshold: {optimal_threshold:.2f}\nPrecision: {precision[idx]:.2f}\nRecall: {recall[idx]:.2f}',
                    xy=(recall[idx], precision[idx]),
                    xytext=(recall[idx] - 0.2, precision[idx] - 0.2),
                    arrowprops=dict(arrowstyle='->', lw=1.5))
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig('output/results/precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot expert contributions
    plt.figure(figsize=(12, 6))
    
    # Extract expert metrics
    expert_names = ['Sleep', 'Weather', 'Stress/Diet', 'Physio', 'Ensemble']
    expert_aucs = [
        metrics['sleep_expert']['auc'],
        metrics['weather_expert']['auc'],
        metrics['stress_diet_expert']['auc'],
        metrics['physio_expert']['auc'],
        metrics['ensemble']['auc']
    ]
    
    # Plot expert AUCs
    plt.bar(expert_names, expert_aucs, color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12'])
    
    # Add value labels
    for i, v in enumerate(expert_aucs):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center')
    
    plt.ylim([0, 1.0])
    plt.ylabel('AUC Score')
    plt.title('Expert Model Contributions')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Save plot
    plt.savefig('output/results/expert_contributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Results plotted and saved to output/results/")

def main():
    """
    Main function to run the unified solution.
    """
    print("=== Starting Unified Migraine Prediction Solution ===")
    start_time = time.time()
    
    # Load data
    train_data, val_data, test_data = load_data()
    
    # Enhance features
    enhanced_train_data, enhanced_val_data, enhanced_test_data = enhance_features(train_data, val_data, test_data)
    
    # Apply class balancing
    balanced_train_data = apply_class_balancing(enhanced_train_data)
    
    # Create and train expert models
    expert_models, expert_histories = create_expert_models(balanced_train_data, enhanced_val_data)
    
    # Create and train ensemble model
    ensemble_model, ensemble_history, ensemble_data = create_ensemble_model(expert_models, balanced_train_data, enhanced_val_data)
    
    # Optimize threshold
    optimal_threshold, threshold_metrics = optimize_threshold(ensemble_model, ensemble_data)
    
    # Evaluate model
    metrics, ensemble_test_pred, y_test = evaluate_model(ensemble_model, expert_models, enhanced_test_data, optimal_threshold)
    
    # Plot results
    plot_results(metrics, ensemble_test_pred, y_test, optimal_threshold)
    
    # Print completion message
    elapsed_time = time.time() - start_time
    print(f"\n=== Unified Solution Completed in {elapsed_time:.2f} seconds ===")
    print(f"Final Ensemble Model Performance:")
    print(f"AUC: {metrics['ensemble']['auc']:.4f}")
    print(f"Accuracy: {metrics['ensemble']['accuracy']:.4f}")
    print(f"Precision: {metrics['ensemble']['precision']:.4f}")
    print(f"Recall: {metrics['ensemble']['recall']:.4f}")
    print(f"F1 Score: {metrics['ensemble']['f1']:.4f}")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    
    # Save metrics for dashboard
    np.savez(
        'output/metrics.npz',
        ensemble_metrics=metrics['ensemble'],
        sleep_expert_metrics=metrics['sleep_expert'],
        weather_expert_metrics=metrics['weather_expert'],
        stress_diet_expert_metrics=metrics['stress_diet_expert'],
        physio_expert_metrics=metrics['physio_expert'],
        optimal_threshold=optimal_threshold
    )
    
    return metrics

if __name__ == "__main__":
    main()
