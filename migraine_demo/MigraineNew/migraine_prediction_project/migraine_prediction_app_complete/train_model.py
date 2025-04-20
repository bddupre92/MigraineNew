import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
import os
import time
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import seaborn as sns

# Create output directories
os.makedirs('output/models', exist_ok=True)
os.makedirs('output/results', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class SleepExpert(layers.Layer):
    """Sleep expert model for migraine prediction."""
    
    def __init__(self, units=64, dropout_rate=0.3, **kwargs):
        super(SleepExpert, self).__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        
        # LSTM layers for temporal data
        self.lstm1 = layers.LSTM(units, return_sequences=True)
        self.lstm2 = layers.LSTM(units)
        
        # Dense layers
        self.dense1 = layers.Dense(units, activation='relu')
        self.dropout = layers.Dropout(dropout_rate)
        self.dense2 = layers.Dense(units // 2, activation='relu')
        self.output_layer = layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs, training=False):
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return self.output_layer(x)
    
    def get_config(self):
        config = super(SleepExpert, self).get_config()
        config.update({
            'units': self.units,
            'dropout_rate': self.dropout_rate
        })
        return config

class WeatherExpert(layers.Layer):
    """Weather expert model for migraine prediction."""
    
    def __init__(self, units=64, dropout_rate=0.3, **kwargs):
        super(WeatherExpert, self).__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        
        # Dense layers
        self.dense1 = layers.Dense(units, activation='relu')
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dense2 = layers.Dense(units // 2, activation='relu')
        self.dropout2 = layers.Dropout(dropout_rate)
        self.output_layer = layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return self.output_layer(x)
    
    def get_config(self):
        config = super(WeatherExpert, self).get_config()
        config.update({
            'units': self.units,
            'dropout_rate': self.dropout_rate
        })
        return config

class StressDietExpert(layers.Layer):
    """Stress and diet expert model for migraine prediction."""
    
    def __init__(self, units=64, dropout_rate=0.3, **kwargs):
        super(StressDietExpert, self).__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        
        # LSTM layers for temporal data
        self.lstm1 = layers.LSTM(units, return_sequences=True)
        self.lstm2 = layers.LSTM(units)
        
        # Dense layers
        self.dense1 = layers.Dense(units, activation='relu')
        self.dropout = layers.Dropout(dropout_rate)
        self.dense2 = layers.Dense(units // 2, activation='relu')
        self.output_layer = layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs, training=False):
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return self.output_layer(x)
    
    def get_config(self):
        config = super(StressDietExpert, self).get_config()
        config.update({
            'units': self.units,
            'dropout_rate': self.dropout_rate
        })
        return config

class PhysioExpert(layers.Layer):
    """Physiological expert model for migraine prediction."""
    
    def __init__(self, units=64, dropout_rate=0.3, **kwargs):
        super(PhysioExpert, self).__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        
        # Dense layers
        self.dense1 = layers.Dense(units, activation='relu')
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dense2 = layers.Dense(units // 2, activation='relu')
        self.dropout2 = layers.Dropout(dropout_rate)
        self.output_layer = layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return self.output_layer(x)
    
    def get_config(self):
        config = super(PhysioExpert, self).get_config()
        config.update({
            'units': self.units,
            'dropout_rate': self.dropout_rate
        })
        return config

class GatingNetwork(layers.Layer):
    """Gating network for mixture of experts."""
    
    def __init__(self, num_experts=4, units=32, **kwargs):
        super(GatingNetwork, self).__init__(**kwargs)
        self.num_experts = num_experts
        self.units = units
        
        # Dense layers will be created in build method
        self.dense1 = None
        self.dense2 = None
    
    def build(self, input_shape):
        # Calculate total input dimension
        total_dim = 0
        for shape in input_shape:
            if isinstance(shape, tf.TensorShape):
                total_dim += shape[-1]
            else:
                total_dim += shape
        
        # Create dense layers with explicit input shape
        self.dense1 = layers.Dense(self.units, activation='relu', input_shape=(total_dim,))
        self.dense2 = layers.Dense(self.num_experts, activation='softmax')
        
        super(GatingNetwork, self).build(input_shape)
    
    def call(self, inputs):
        # Flatten all inputs to 2D
        flattened_inputs = []
        for inp in inputs:
            if len(inp.shape) > 2:
                # For 3D inputs (batch_size, time_steps, features)
                batch_size = tf.shape(inp)[0]
                flattened = tf.reshape(inp, [batch_size, -1])
                flattened_inputs.append(flattened)
            else:
                flattened_inputs.append(inp)
        
        # Concatenate all inputs
        x = tf.concat(flattened_inputs, axis=1)
        x = self.dense1(x)
        return self.dense2(x)
    
    def get_config(self):
        config = super(GatingNetwork, self).get_config()
        config.update({
            'num_experts': self.num_experts,
            'units': self.units
        })
        return config

class MigraineMoEModel(tf.keras.Model):
    """Mixture of Experts model for migraine prediction."""
    
    def __init__(self, **kwargs):
        super(MigraineMoEModel, self).__init__(**kwargs)
        
        # Expert models
        self.sleep_expert = SleepExpert(units=64, dropout_rate=0.3)
        self.weather_expert = WeatherExpert(units=32, dropout_rate=0.2)
        self.stress_diet_expert = StressDietExpert(units=64, dropout_rate=0.3)
        self.physio_expert = PhysioExpert(units=32, dropout_rate=0.2)
        
        # Gating network
        self.gating_network = GatingNetwork(num_experts=4, units=32)
        
        # Final dense layer
        self.final_dense = layers.Dense(1, activation='sigmoid')
        
        # Build the model with some dummy data to initialize shapes
        self._build_model_graph()
    
    def _build_model_graph(self):
        """Initialize the model graph with dummy inputs to set shapes."""
        # Create dummy inputs with batch size 2
        dummy_sleep = tf.zeros((2, 7, 6))
        dummy_weather = tf.zeros((2, 4))
        dummy_stress_diet = tf.zeros((2, 7, 6))
        dummy_physio = tf.zeros((2, 5))
        
        # Call the model once to build all layers
        self([dummy_sleep, dummy_weather, dummy_stress_diet, dummy_physio], training=False)
    
    def call(self, inputs, training=False):
        # Unpack inputs
        sleep_input, weather_input, stress_diet_input, physio_input = inputs
        
        # Get expert outputs
        sleep_output = self.sleep_expert(sleep_input, training=training)
        weather_output = self.weather_expert(weather_input, training=training)
        stress_diet_output = self.stress_diet_expert(stress_diet_input, training=training)
        physio_output = self.physio_expert(physio_input, training=training)
        
        # Stack expert outputs
        expert_outputs = tf.stack([
            sleep_output, 
            weather_output, 
            stress_diet_output, 
            physio_output
        ], axis=1)
        
        # Get gating weights
        # Flatten sleep and stress_diet inputs for gating
        sleep_flat = tf.reshape(sleep_input, [tf.shape(sleep_input)[0], -1])
        stress_diet_flat = tf.reshape(stress_diet_input, [tf.shape(stress_diet_input)[0], -1])
        
        gating_inputs = [sleep_flat, weather_input, stress_diet_flat, physio_input]
        gating_weights = self.gating_network(gating_inputs)
        
        # Apply gating weights
        gating_weights = tf.expand_dims(gating_weights, axis=2)
        weighted_outputs = expert_outputs * gating_weights
        combined_output = tf.reduce_sum(weighted_outputs, axis=1)
        
        # Final prediction
        final_output = self.final_dense(combined_output)
        
        return final_output, gating_weights, expert_outputs

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal loss for imbalanced classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        alpha: Weighting factor
        gamma: Focusing parameter
        
    Returns:
        float: Loss value
    """
    # Convert to tensor
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    
    # Calculate focal loss
    loss = -alpha * (1 - y_pred) ** gamma * y_true * tf.math.log(y_pred) - \
           (1 - alpha) * y_pred ** gamma * (1 - y_true) * tf.math.log(1 - y_pred)
    
    return tf.reduce_mean(loss)

def load_data(data_dir='output/data'):
    """
    Load data from files.
    
    Args:
        data_dir: Data directory
        
    Returns:
        tuple: Train, validation, and test data
    """
    print(f"Loading data from {data_dir}")
    
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
    
    print(f"Train set: {len(y_train)} samples, positive ratio: {np.mean(y_train):.2f}")
    print(f"Validation set: {len(y_val)} samples, positive ratio: {np.mean(y_val):.2f}")
    print(f"Test set: {len(y_test)} samples, positive ratio: {np.mean(y_test):.2f}")
    
    return train_data, val_data, test_data

def train_model(train_data, val_data, epochs=50, batch_size=32, learning_rate=0.001, early_stopping_patience=10):
    """
    Train the migraine prediction model.
    
    Args:
        train_data: Training data dictionary
        val_data: Validation data dictionary
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        early_stopping_patience: Early stopping patience
        
    Returns:
        tuple: Trained model and training history
    """
    print("Training migraine prediction model")
    
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
    
    # Create model
    model = MigraineMoEModel()
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=focal_loss,
        metrics=['accuracy']
    )
    
    # Create early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=early_stopping_patience,
        restore_best_weights=True
    )
    
    # Create learning rate scheduler
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    # Create model checkpoint callback
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'output/models/best_model.keras',
        monitor='val_loss',
        save_best_only=True
    )
    
    # Train model
    history = model.fit(
        [X_sleep_train, X_weather_train, X_stress_diet_train, X_physio_train],
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=([X_sleep_val, X_weather_val, X_stress_diet_val, X_physio_val], y_val),
        callbacks=[early_stopping, lr_scheduler, checkpoint],
        verbose=1
    )
    
    # Save final model
    model.save('output/models/final_model.keras')
    
    print("Model training completed")
    
    return model, history

def evaluate_model(model, test_data, threshold=0.5):
    """
    Evaluate the migraine prediction model.
    
    Args:
        model: Trained model
        test_data: Test data dictionary
        threshold: Classification threshold
        
    Returns:
        dict: Evaluation metrics
    """
    print("Evaluating model performance")
    
    # Extract data
    X_sleep_test = test_data['X_sleep']
    X_weather_test = test_data['X_weather']
    X_stress_diet_test = test_data['X_stress_diet']
    X_physio_test = test_data['X_physio']
    y_test = test_data['y']
    
    # Make predictions
    y_pred_prob, gating_weights, expert_outputs = model([X_sleep_test, X_weather_test, X_stress_diet_test, X_physio_test], training=False)
    y_pred_prob = y_pred_prob.numpy().flatten()
    
    # Convert to binary predictions
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    # Calculate metrics
    accuracy = np.mean(y_pred == y_test)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    
    # Calculate precision-recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(recall_curve, precision_curve)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate expert contributions
    gating_weights = gating_weights.numpy().mean(axis=0)
    
    # Create metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm,
        'gating_weights': gating_weights,
        'y_test': y_test,
        'y_pred_prob': y_pred_prob,
        'y_pred': y_pred
    }
    
    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Expert Contributions: {gating_weights}")
    
    return metrics

def optimize_threshold(metrics):
    """
    Find optimal threshold to maximize F1 score.
    
    Args:
        metrics: Evaluation metrics
        
    Returns:
        float: Optimal threshold
    """
    print("Optimizing classification threshold")
    
    # Extract data
    y_test = metrics['y_test']
    y_pred_prob = metrics['y_pred_prob']
    
    # Calculate precision and recall at different thresholds
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
    
    # Calculate F1 score at each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # Find threshold that maximizes F1 score
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    # Calculate metrics at optimal threshold
    y_pred_optimal = (y_pred_prob >= optimal_threshold).astype(int)
    accuracy = np.mean(y_pred_optimal == y_test)
    precision_optimal = precision_score(y_test, y_pred_optimal, zero_division=0)
    recall_optimal = recall_score(y_test, y_pred_optimal)
    f1_optimal = f1_score(y_test, y_pred_optimal, zero_division=0)
    
    # Print results
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print(f"Accuracy at optimal threshold: {accuracy:.4f}")
    print(f"Precision at optimal threshold: {precision_optimal:.4f}")
    print(f"Recall at optimal threshold: {recall_optimal:.4f}")
    print(f"F1 Score at optimal threshold: {f1_optimal:.4f}")
    
    # Update metrics
    metrics['optimal_threshold'] = optimal_threshold
    metrics['accuracy_optimal'] = accuracy
    metrics['precision_optimal'] = precision_optimal
    metrics['recall_optimal'] = recall_optimal
    metrics['f1_score_optimal'] = f1_optimal
    
    return metrics

def plot_training_history(history, output_dir='output/results'):
    """
    Plot training history.
    
    Args:
        history: Training history
        output_dir: Output directory
    """
    print("Plotting training history")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'accuracy_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Training history plots saved")

def plot_roc_curve(metrics, output_dir='output/results'):
    """
    Plot ROC curve.
    
    Args:
        metrics: Evaluation metrics
        output_dir: Output directory
    """
    print("Plotting ROC curve")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    y_test = metrics['y_test']
    y_pred_prob = metrics['y_pred_prob']
    roc_auc = metrics['roc_auc']
    
    # Calculate ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("ROC curve plot saved")

def plot_precision_recall_curve(metrics, output_dir='output/results'):
    """
    Plot precision-recall curve.
    
    Args:
        metrics: Evaluation metrics
        output_dir: Output directory
    """
    print("Plotting precision-recall curve")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    y_test = metrics['y_test']
    y_pred_prob = metrics['y_pred_prob']
    pr_auc = metrics['pr_auc']
    
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    
    # Plot precision-recall curve
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (area = {pr_auc:.4f})')
    plt.axhline(y=np.mean(y_test), color='navy', lw=2, linestyle='--', label=f'Baseline (y_mean = {np.mean(y_test):.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Precision-recall curve plot saved")

def plot_confusion_matrix(metrics, output_dir='output/results'):
    """
    Plot confusion matrix.
    
    Args:
        metrics: Evaluation metrics
        output_dir: Output directory
    """
    print("Plotting confusion matrix")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    cm = metrics['confusion_matrix']
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot confusion matrix at optimal threshold
    if 'optimal_threshold' in metrics:
        y_test = metrics['y_test']
        y_pred_prob = metrics['y_pred_prob']
        optimal_threshold = metrics['optimal_threshold']
        
        # Calculate confusion matrix at optimal threshold
        y_pred_optimal = (y_pred_prob >= optimal_threshold).astype(int)
        cm_optimal = confusion_matrix(y_test, y_pred_optimal)
        
        # Plot confusion matrix at optimal threshold
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix at Optimal Threshold ({optimal_threshold:.4f})')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix_optimal.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Confusion matrix plots saved")

def plot_expert_contributions(metrics, output_dir='output/results'):
    """
    Plot expert contributions.
    
    Args:
        metrics: Evaluation metrics
        output_dir: Output directory
    """
    print("Plotting expert contributions")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    gating_weights = metrics['gating_weights']
    
    # Expert names
    expert_names = ['Sleep Expert', 'Weather Expert', 'Stress/Diet Expert', 'Physio Expert']
    
    # Plot expert contributions
    plt.figure(figsize=(12, 8))
    bars = plt.bar(expert_names, gating_weights, color=['skyblue', 'lightgreen', 'salmon', 'purple'])
    plt.xlabel('Expert')
    plt.ylabel('Contribution Weight')
    plt.title('Expert Contributions to Migraine Prediction')
    plt.ylim(0, max(gating_weights) * 1.2)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom')
    
    plt.savefig(os.path.join(output_dir, 'expert_contributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Expert contributions plot saved")

def plot_threshold_analysis(metrics, output_dir='output/results'):
    """
    Plot threshold analysis.
    
    Args:
        metrics: Evaluation metrics
        output_dir: Output directory
    """
    print("Plotting threshold analysis")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    y_test = metrics['y_test']
    y_pred_prob = metrics['y_pred_prob']
    optimal_threshold = metrics.get('optimal_threshold', 0.5)
    
    # Calculate metrics at different thresholds
    thresholds = np.linspace(0.01, 0.99, 99)
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_pred_prob >= threshold).astype(int)
        accuracies.append(np.mean(y_pred == y_test))
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
    
    # Plot metrics vs threshold
    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, accuracies, label='Accuracy')
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, recalls, label='Recall')
    plt.plot(thresholds, f1_scores, label='F1 Score')
    plt.axvline(x=optimal_threshold, color='red', linestyle='--', label=f'Optimal Threshold = {optimal_threshold:.4f}')
    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.title('Performance Metrics vs. Classification Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'threshold_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Threshold analysis plot saved")

def save_metrics(metrics, output_dir='output/results'):
    """
    Save metrics to file.
    
    Args:
        metrics: Evaluation metrics
        output_dir: Output directory
    """
    print("Saving metrics to file")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create metrics dictionary for saving
    metrics_to_save = {
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1_score': metrics['f1_score'],
        'roc_auc': metrics['roc_auc'],
        'pr_auc': metrics['pr_auc']
    }
    
    # Add optimal threshold metrics if available
    if 'optimal_threshold' in metrics:
        metrics_to_save['optimal_threshold'] = metrics['optimal_threshold']
        metrics_to_save['accuracy_optimal'] = metrics['accuracy_optimal']
        metrics_to_save['precision_optimal'] = metrics['precision_optimal']
        metrics_to_save['recall_optimal'] = metrics['recall_optimal']
        metrics_to_save['f1_score_optimal'] = metrics['f1_score_optimal']
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame([metrics_to_save])
    
    # Save to CSV
    metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
    
    # Save expert contributions
    expert_names = ['Sleep Expert', 'Weather Expert', 'Stress/Diet Expert', 'Physio Expert']
    expert_contributions = pd.DataFrame({
        'Expert': expert_names,
        'Contribution': metrics['gating_weights']
    })
    expert_contributions.to_csv(os.path.join(output_dir, 'expert_contributions.csv'), index=False)
    
    # Save predictions for dashboard
    np.savez(
        os.path.join(output_dir, 'test_predictions.npz'),
        y_true=metrics['y_test'],
        y_pred=metrics['y_pred_prob']
    )
    
    print("Metrics saved to file")

def main():
    """
    Main function to train and evaluate the migraine prediction model.
    """
    print("Starting migraine prediction model training and evaluation")
    start_time = time.time()
    
    # Load data
    train_data, val_data, test_data = load_data()
    
    # Train model
    model, history = train_model(
        train_data, 
        val_data, 
        epochs=50, 
        batch_size=32, 
        learning_rate=0.001, 
        early_stopping_patience=10
    )
    
    # Evaluate model
    metrics = evaluate_model(model, test_data)
    
    # Optimize threshold
    metrics = optimize_threshold(metrics)
    
    # Plot results
    plot_training_history(history)
    plot_roc_curve(metrics)
    plot_precision_recall_curve(metrics)
    plot_confusion_matrix(metrics)
    plot_expert_contributions(metrics)
    plot_threshold_analysis(metrics)
    
    # Save metrics
    save_metrics(metrics)
    
    # Print completion message
    elapsed_time = time.time() - start_time
    print(f"Model training and evaluation completed in {elapsed_time:.2f} seconds")
    print(f"Results saved to output/results/")

if __name__ == "__main__":
    main()
