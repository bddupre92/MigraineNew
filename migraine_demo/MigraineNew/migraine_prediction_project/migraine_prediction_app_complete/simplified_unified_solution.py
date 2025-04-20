import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, BorderlineSMOTE

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

def preprocess_data(train_data, val_data, test_data):
    """
    Preprocess data by flattening and scaling.
    
    Args:
        train_data (dict): Training data dictionary
        val_data (dict): Validation data dictionary
        test_data (dict): Test data dictionary
        
    Returns:
        tuple: Preprocessed train, validation, and test data
    """
    print("\n=== Preprocessing Data ===")
    
    # Extract data
    X_sleep_train = train_data['X_sleep']
    X_weather_train = train_data['X_weather']
    X_stress_diet_train = train_data['X_stress_diet']
    X_physio_train = train_data['X_physio']
    y_train = train_data['y']
    
    X_sleep_val = val_data['X_sleep']
    X_weather_val = val_data['X_weather']
    X_stress_diet_val = val_data['X_stress_diet']
    X_physio_val = val_data['X_physio']
    y_val = val_data['y']
    
    X_sleep_test = test_data['X_sleep']
    X_weather_test = test_data['X_weather']
    X_stress_diet_test = test_data['X_stress_diet']
    X_physio_test = test_data['X_physio']
    y_test = test_data['y']
    
    # Flatten 3D data
    if len(X_sleep_train.shape) > 2:
        X_sleep_train_flat = X_sleep_train.reshape(X_sleep_train.shape[0], -1)
        X_sleep_val_flat = X_sleep_val.reshape(X_sleep_val.shape[0], -1)
        X_sleep_test_flat = X_sleep_test.reshape(X_sleep_test.shape[0], -1)
    else:
        X_sleep_train_flat = X_sleep_train
        X_sleep_val_flat = X_sleep_val
        X_sleep_test_flat = X_sleep_test
    
    if len(X_stress_diet_train.shape) > 2:
        X_stress_diet_train_flat = X_stress_diet_train.reshape(X_stress_diet_train.shape[0], -1)
        X_stress_diet_val_flat = X_stress_diet_val.reshape(X_stress_diet_val.shape[0], -1)
        X_stress_diet_test_flat = X_stress_diet_test.reshape(X_stress_diet_test.shape[0], -1)
    else:
        X_stress_diet_train_flat = X_stress_diet_train
        X_stress_diet_val_flat = X_stress_diet_val
        X_stress_diet_test_flat = X_stress_diet_test
    
    # Scale data
    scaler_sleep = StandardScaler()
    X_sleep_train_scaled = scaler_sleep.fit_transform(X_sleep_train_flat)
    X_sleep_val_scaled = scaler_sleep.transform(X_sleep_val_flat)
    X_sleep_test_scaled = scaler_sleep.transform(X_sleep_test_flat)
    
    scaler_weather = StandardScaler()
    X_weather_train_scaled = scaler_weather.fit_transform(X_weather_train)
    X_weather_val_scaled = scaler_weather.transform(X_weather_val)
    X_weather_test_scaled = scaler_weather.transform(X_weather_test)
    
    scaler_stress_diet = StandardScaler()
    X_stress_diet_train_scaled = scaler_stress_diet.fit_transform(X_stress_diet_train_flat)
    X_stress_diet_val_scaled = scaler_stress_diet.transform(X_stress_diet_val_flat)
    X_stress_diet_test_scaled = scaler_stress_diet.transform(X_stress_diet_test_flat)
    
    scaler_physio = StandardScaler()
    X_physio_train_scaled = scaler_physio.fit_transform(X_physio_train)
    X_physio_val_scaled = scaler_physio.transform(X_physio_val)
    X_physio_test_scaled = scaler_physio.transform(X_physio_test)
    
    # Create preprocessed data dictionaries
    preprocessed_train_data = {
        'X_sleep': X_sleep_train_scaled,
        'X_weather': X_weather_train_scaled,
        'X_stress_diet': X_stress_diet_train_scaled,
        'X_physio': X_physio_train_scaled,
        'y': y_train
    }
    
    preprocessed_val_data = {
        'X_sleep': X_sleep_val_scaled,
        'X_weather': X_weather_val_scaled,
        'X_stress_diet': X_stress_diet_val_scaled,
        'X_physio': X_physio_val_scaled,
        'y': y_val
    }
    
    preprocessed_test_data = {
        'X_sleep': X_sleep_test_scaled,
        'X_weather': X_weather_test_scaled,
        'X_stress_diet': X_stress_diet_test_scaled,
        'X_physio': X_physio_test_scaled,
        'y': y_test
    }
    
    print(f"Preprocessed sleep features shape: {X_sleep_train_scaled.shape}")
    print(f"Preprocessed weather features shape: {X_weather_train_scaled.shape}")
    print(f"Preprocessed stress/diet features shape: {X_stress_diet_train_scaled.shape}")
    print(f"Preprocessed physiological features shape: {X_physio_train_scaled.shape}")
    
    return preprocessed_train_data, preprocessed_val_data, preprocessed_test_data

def apply_class_balancing(train_data):
    """
    Apply class balancing techniques to the training data.
    
    Args:
        train_data (dict): Training data dictionary
        
    Returns:
        dict: Balanced training data
    """
    print("\n=== Applying Class Balancing ===")
    
    # Extract data
    X_sleep = train_data['X_sleep']
    X_weather = train_data['X_weather']
    X_stress_diet = train_data['X_stress_diet']
    X_physio = train_data['X_physio']
    y = train_data['y']
    
    # Combine all features for SMOTE
    X_combined = np.hstack([
        X_sleep,
        X_weather,
        X_stress_diet,
        X_physio
    ])
    
    # Plot original class distribution
    plt.figure(figsize=(10, 6))
    unique_original, counts_original = np.unique(y, return_counts=True)
    plt.bar(unique_original - 0.2, counts_original, width=0.4, label='Original', color='blue', alpha=0.7)
    for i, count in enumerate(counts_original):
        percentage = count / len(y) * 100
        plt.text(unique_original[i] - 0.2, count + 5, f"{percentage:.1f}%", ha='center')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.xticks(unique_original, [f'Class {int(cls)}' for cls in unique_original])
    plt.grid(True, alpha=0.3)
    os.makedirs('output/class_balancing', exist_ok=True)
    plt.savefig('output/class_balancing/original_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Apply BorderlineSMOTE
    print("\nApplying BorderlineSMOTE...")
    try:
        smote = BorderlineSMOTE(sampling_strategy=0.5, random_state=42)
        X_combined_balanced, y_balanced = smote.fit_resample(X_combined, y)
    except Exception as e:
        print(f"Error with BorderlineSMOTE: {e}, falling back to regular SMOTE")
        smote = SMOTE(sampling_strategy=0.5, random_state=42)
        X_combined_balanced, y_balanced = smote.fit_resample(X_combined, y)
    
    # Plot balanced class distribution
    plt.figure(figsize=(10, 6))
    plt.bar(unique_original - 0.2, counts_original, width=0.4, label='Original', color='blue', alpha=0.7)
    for i, count in enumerate(counts_original):
        percentage = count / len(y) * 100
        plt.text(unique_original[i] - 0.2, count + 5, f"{percentage:.1f}%", ha='center')
    
    unique_balanced, counts_balanced = np.unique(y_balanced, return_counts=True)
    plt.bar(unique_balanced + 0.2, counts_balanced, width=0.4, label='Balanced', color='red', alpha=0.7)
    for i, count in enumerate(counts_balanced):
        percentage = count / len(y_balanced) * 100
        plt.text(unique_balanced[i] + 0.2, count + 5, f"{percentage:.1f}%", ha='center')
    
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.xticks(unique_original, [f'Class {int(cls)}' for cls in unique_original])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('output/class_balancing/balanced_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate class weights
    class_weights = {}
    for cls in np.unique(y_balanced):
        class_weights[cls] = len(y_balanced) / (len(np.unique(y_balanced)) * np.sum(y_balanced == cls))
    
    print("Class weights:")
    for cls, weight in class_weights.items():
        print(f"Class {int(cls)}: {weight:.4f}")
    
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
        X_sleep_train[:, :min(10, X_sleep_train.shape[1])],  # Include top features from each expert
        X_weather_train[:, :min(5, X_weather_train.shape[1])],
        X_stress_diet_train[:, :min(10, X_stress_diet_train.shape[1])],
        X_physio_train[:, :min(5, X_physio_train.shape[1])]
    ])
    
    X_ensemble_val = np.hstack([
        sleep_val_pred,
        weather_val_pred,
        stress_diet_val_pred,
        physio_val_pred,
        X_sleep_val[:, :min(10, X_sleep_val.shape[1])],  # Include top features from each expert
        X_weather_val[:, :min(5, X_weather_val.shape[1])],
        X_stress_diet_val[:, :min(10, X_stress_diet_val.shape[1])],
        X_physio_val[:, :min(5, X_physio_val.shape[1])]
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
    
    # Get validation predictions
    X_val = ensemble_data['X_val']
    y_val = ensemble_data['y_val']
    y_pred_val = ensemble_model.predict(X_val, verbose=0).flatten()
    
    # Plot precision-recall curve
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    precision, recall, thresholds = precision_recall_curve(y_val, y_pred_val)
    
    # Add a threshold of 1.0 to match the length of precision and recall
    thresholds = np.append(thresholds, 1.0)
    
    # Calculate F1 score for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # Find threshold that maximizes F1 score
    optimal_idx = np.argmax(f1_scores)
    optimal_f1_threshold = thresholds[optimal_idx]
    
    print(f"Optimal F1 threshold: {optimal_f1_threshold:.4f}")
    print(f"Optimal F1 score: {f1_scores[optimal_idx]:.4f}")
    print(f"Precision at optimal threshold: {precision[optimal_idx]:.4f}")
    print(f"Recall at optimal threshold: {recall[optimal_idx]:.4f}")
    
    # Find threshold where precision equals recall
    diff = np.abs(precision - recall)
    optimal_pr_idx = np.argmin(diff)
    optimal_pr_threshold = thresholds[optimal_pr_idx]
    
    print(f"Optimal balanced threshold: {optimal_pr_threshold:.4f}")
    print(f"Precision at balanced threshold: {precision[optimal_pr_idx]:.4f}")
    print(f"Recall at balanced threshold: {recall[optimal_pr_idx]:.4f}")
    
    # Create output directory
    os.makedirs('output/threshold_optimization', exist_ok=True)
    
    # Plot precision-recall curve
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, 'b-', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    
    # Mark optimal threshold
    plt.plot(recall[optimal_idx], precision[optimal_idx], 'ro', markersize=8)
    plt.annotate(f'F1 Threshold: {optimal_f1_threshold:.2f}\nPrecision: {precision[optimal_idx]:.2f}\nRecall: {recall[optimal_idx]:.2f}',
                xy=(recall[optimal_idx], precision[optimal_idx]),
                xytext=(recall[optimal_idx] - 0.2, precision[optimal_idx] - 0.2),
                arrowprops=dict(arrowstyle='->', lw=1.5))
    
    # Mark balanced threshold
    plt.plot(recall[optimal_pr_idx], precision[optimal_pr_idx], 'go', markersize=8)
    plt.annotate(f'PR Threshold: {optimal_pr_threshold:.2f}\nPrecision: {precision[optimal_pr_idx]:.2f}\nRecall: {recall[optimal_pr_idx]:.2f}',
                xy=(recall[optimal_pr_idx], precision[optimal_pr_idx]),
                xytext=(recall[optimal_pr_idx] + 0.1, precision[optimal_pr_idx] + 0.1),
                arrowprops=dict(arrowstyle='->', lw=1.5))
    
    plt.savefig('output/threshold_optimization/precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot threshold metrics
    plt.figure(figsize=(12, 6))
    plt.plot(thresholds, precision[:-1], 'b-', label='Precision', linewidth=2)
    plt.plot(thresholds, recall[:-1], 'r-', label='Recall', linewidth=2)
    plt.plot(thresholds, f1_scores[:-1], 'g-', label='F1 Score', linewidth=2)
    
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1 Score vs. Threshold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Mark optimal thresholds
    plt.axvline(x=optimal_f1_threshold, color='g', linestyle='--', alpha=0.7)
    plt.annotate(f'F1 Threshold: {optimal_f1_threshold:.2f}',
                xy=(optimal_f1_threshold, f1_scores[optimal_idx]),
                xytext=(optimal_f1_threshold + 0.1, f1_scores[optimal_idx]),
                arrowprops=dict(arrowstyle='->', lw=1.5))
    
    plt.axvline(x=optimal_pr_threshold, color='purple', linestyle='--', alpha=0.7)
    plt.annotate(f'PR Threshold: {optimal_pr_threshold:.2f}',
                xy=(optimal_pr_threshold, precision[optimal_pr_idx]),
                xytext=(optimal_pr_threshold + 0.1, precision[optimal_pr_idx] - 0.1),
                arrowprops=dict(arrowstyle='->', lw=1.5))
    
    plt.savefig('output/threshold_optimization/threshold_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Compare thresholds
    thresholds_dict = {
        'Default': 0.5,
        'F1 Optimized': optimal_f1_threshold,
        'PR Balance': optimal_pr_threshold
    }
    
    print("\nThreshold Comparison:")
    print(f"{'Threshold':<20} {'Value':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10}")
    print("-" * 60)
    
    for name, threshold in thresholds_dict.items():
        # Apply threshold to get binary predictions
        y_pred_binary = (y_pred_val >= threshold).astype(int)
        
        # Calculate metrics
        precision_val = precision_score(y_val, y_pred_binary)
        recall_val = recall_score(y_val, y_pred_binary)
        f1_val = f1_score(y_val, y_pred_binary)
        
        print(f"{name:<20} {threshold:<10.4f} {precision_val:<10.4f} {recall_val:<10.4f} {f1_val:<10.4f}")
    
    # Choose the F1-optimized threshold as our final threshold
    optimal_threshold = optimal_f1_threshold
    print(f"\nSelected optimal threshold: {optimal_threshold:.4f}")
    
    return optimal_threshold

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
        X_sleep_test[:, :min(10, X_sleep_test.shape[1])],  # Include top features from each expert
        X_weather_test[:, :min(5, X_weather_test.shape[1])],
        X_stress_diet_test[:, :min(10, X_stress_diet_test.shape[1])],
        X_physio_test[:, :min(5, X_physio_test.shape[1])]
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
    if len(thresholds) > 0:
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
    
    # Preprocess data
    preprocessed_train_data, preprocessed_val_data, preprocessed_test_data = preprocess_data(train_data, val_data, test_data)
    
    # Apply class balancing
    balanced_train_data = apply_class_balancing(preprocessed_train_data)
    
    # Create and train expert models
    expert_models, expert_histories = create_expert_models(balanced_train_data, preprocessed_val_data)
    
    # Create and train ensemble model
    ensemble_model, ensemble_history, ensemble_data = create_ensemble_model(expert_models, balanced_train_data, preprocessed_val_data)
    
    # Optimize threshold
    optimal_threshold = optimize_threshold(ensemble_model, ensemble_data)
    
    # Evaluate model
    metrics, ensemble_test_pred, y_test = evaluate_model(ensemble_model, expert_models, preprocessed_test_data, optimal_threshold)
    
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
