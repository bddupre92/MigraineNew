import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
import time
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

# Import our unified solution
from unified_solution import EnhancedMigrainePrediction

# Set up output directory
OUTPUT_DIR = 'output/evaluation'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_synthetic_data(n_samples=1000, positive_ratio=0.1):
    """
    Generate synthetic data for evaluation.
    
    Args:
        n_samples (int): Number of samples
        positive_ratio (float): Ratio of positive samples
        
    Returns:
        tuple: Sleep data, weather data, stress/diet data, labels
    """
    print(f"Generating synthetic data with {n_samples} samples ({positive_ratio*100:.1f}% positive)...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Sleep data: (n_samples, 7, 6) - 7 days, 6 features per day
    X_sleep = np.random.randn(n_samples, 7, 6)
    
    # Weather data: (n_samples, 4) - 4 weather features
    X_weather = np.random.randn(n_samples, 4)
    
    # Stress/diet data: (n_samples, 7, 6) - 7 days, 6 features per day
    X_stress_diet = np.random.randn(n_samples, 7, 6)
    
    # Labels: positive_ratio positive, (1-positive_ratio) negative
    y = np.zeros(n_samples)
    n_positive = int(n_samples * positive_ratio)
    y[:n_positive] = 1
    
    # Shuffle data
    indices = np.random.permutation(n_samples)
    X_sleep = X_sleep[indices]
    X_weather = X_weather[indices]
    X_stress_diet = X_stress_diet[indices]
    y = y[indices]
    
    return X_sleep, X_weather, X_stress_diet, y

def load_real_data():
    """
    Load real data from the dataset if available.
    
    Returns:
        tuple: Sleep data, weather data, stress/diet data, labels
    """
    try:
        print("Attempting to load real data...")
        
        # Try to find data files
        data_dir = '/home/ubuntu/migraine_demo/MigraineNew/migraine_prediction_project/migraine_prediction_app_complete/data'
        
        # Check if directory exists
        if not os.path.exists(data_dir):
            print(f"Data directory not found: {data_dir}")
            return None
        
        # Look for data files
        sleep_file = os.path.join(data_dir, 'sleep_data.npy')
        weather_file = os.path.join(data_dir, 'weather_data.npy')
        stress_diet_file = os.path.join(data_dir, 'stress_diet_data.npy')
        labels_file = os.path.join(data_dir, 'labels.npy')
        
        # Check if all files exist
        if not all(os.path.exists(f) for f in [sleep_file, weather_file, stress_diet_file, labels_file]):
            print("Not all data files found")
            return None
        
        # Load data
        X_sleep = np.load(sleep_file)
        X_weather = np.load(weather_file)
        X_stress_diet = np.load(stress_diet_file)
        y = np.load(labels_file)
        
        print(f"Loaded real data: {len(y)} samples")
        print(f"Data shapes: Sleep {X_sleep.shape}, Weather {X_weather.shape}, Stress/Diet {X_stress_diet.shape}")
        print(f"Positive samples: {np.sum(y)} ({np.mean(y)*100:.1f}%)")
        
        return X_sleep, X_weather, X_stress_diet, y
        
    except Exception as e:
        print(f"Error loading real data: {e}")
        return None

def create_baseline_model(input_shape_sleep, input_shape_weather, input_shape_stress_diet):
    """
    Create a simple baseline model for comparison.
    
    Args:
        input_shape_sleep (tuple): Shape of sleep data
        input_shape_weather (tuple): Shape of weather data
        input_shape_stress_diet (tuple): Shape of stress/diet data
        
    Returns:
        tf.keras.Model: Baseline model
    """
    print("Creating baseline model...")
    
    # Sleep input
    sleep_input = tf.keras.layers.Input(shape=input_shape_sleep, name='sleep_input')
    sleep_flatten = tf.keras.layers.Flatten()(sleep_input)
    sleep_dense = tf.keras.layers.Dense(32, activation='relu')(sleep_flatten)
    
    # Weather input
    weather_input = tf.keras.layers.Input(shape=input_shape_weather, name='weather_input')
    weather_dense = tf.keras.layers.Dense(16, activation='relu')(weather_input)
    
    # Stress/diet input
    stress_diet_input = tf.keras.layers.Input(shape=input_shape_stress_diet, name='stress_diet_input')
    stress_diet_flatten = tf.keras.layers.Flatten()(stress_diet_input)
    stress_diet_dense = tf.keras.layers.Dense(32, activation='relu')(stress_diet_flatten)
    
    # Concatenate
    concat = tf.keras.layers.Concatenate()([sleep_dense, weather_dense, stress_diet_dense])
    
    # Dense layers
    x = tf.keras.layers.Dense(64, activation='relu')(concat)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    
    # Output
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = tf.keras.Model(
        inputs=[sleep_input, weather_input, stress_diet_input],
        outputs=output,
        name='baseline_model'
    )
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_baseline_model(model, X_sleep, X_weather, X_stress_diet, y, test_size=0.2, val_size=0.1):
    """
    Train the baseline model.
    
    Args:
        model (tf.keras.Model): Baseline model
        X_sleep (np.ndarray): Sleep data
        X_weather (np.ndarray): Weather data
        X_stress_diet (np.ndarray): Stress/diet data
        y (np.ndarray): Labels
        test_size (float): Proportion of data for testing
        val_size (float): Proportion of data for validation
        
    Returns:
        tuple: Trained model, test data, test predictions
    """
    print("Training baseline model...")
    
    # Split data
    n_samples = len(y)
    n_test = int(n_samples * test_size)
    n_val = int(n_samples * val_size)
    n_train = n_samples - n_test - n_val
    
    # Training data
    X_sleep_train = X_sleep[:n_train]
    X_weather_train = X_weather[:n_train]
    X_stress_diet_train = X_stress_diet[:n_train]
    y_train = y[:n_train]
    
    # Validation data
    X_sleep_val = X_sleep[n_train:n_train+n_val]
    X_weather_val = X_weather[n_train:n_train+n_val]
    X_stress_diet_val = X_stress_diet[n_train:n_train+n_val]
    y_val = y[n_train:n_train+n_val]
    
    # Test data
    X_sleep_test = X_sleep[n_train+n_val:]
    X_weather_test = X_weather[n_train+n_val:]
    X_stress_diet_test = X_stress_diet[n_train+n_val:]
    y_test = y[n_train+n_val:]
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]
    
    # Train model
    history = model.fit(
        [X_sleep_train, X_weather_train, X_stress_diet_train],
        y_train,
        validation_data=(
            [X_sleep_val, X_weather_val, X_stress_diet_val],
            y_val
        ),
        epochs=20,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    y_pred = model.predict([X_sleep_test, X_weather_test, X_stress_diet_test])
    
    # Save model
    model.save(os.path.join(OUTPUT_DIR, 'baseline_model.keras'))
    
    # Save test data and predictions
    np.savez(
        os.path.join(OUTPUT_DIR, 'baseline_results.npz'),
        X_sleep_test=X_sleep_test,
        X_weather_test=X_weather_test,
        X_stress_diet_test=X_stress_diet_test,
        y_test=y_test,
        y_pred=y_pred
    )
    
    # Return test data and predictions
    test_data = (X_sleep_test, X_weather_test, X_stress_diet_test, y_test)
    
    return model, test_data, y_pred

def evaluate_baseline_model(y_test, y_pred):
    """
    Evaluate the baseline model.
    
    Args:
        y_test (np.ndarray): True labels
        y_pred (np.ndarray): Predicted probabilities
        
    Returns:
        dict: Performance metrics
    """
    print("Evaluating baseline model...")
    
    # Convert predictions to binary
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    auc = roc_auc_score(y_test, y_pred)
    
    # Print metrics
    print(f"Baseline Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    
    # Return metrics
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

def run_enhanced_model(X_sleep, X_weather, X_stress_diet, y, baseline_model_path):
    """
    Run the enhanced model pipeline.
    
    Args:
        X_sleep (np.ndarray): Sleep data
        X_weather (np.ndarray): Weather data
        X_stress_diet (np.ndarray): Stress/diet data
        y (np.ndarray): Labels
        baseline_model_path (str): Path to baseline model
        
    Returns:
        dict: Performance metrics
    """
    print("Running enhanced model pipeline...")
    
    # Initialize enhanced migraine prediction
    enhanced_prediction = EnhancedMigrainePrediction(output_dir=OUTPUT_DIR)
    
    # Run full pipeline
    metrics = enhanced_prediction.run_full_pipeline(
        X_sleep, X_weather, X_stress_diet, y,
        original_model_path=baseline_model_path
    )
    
    return metrics

def compare_models(baseline_metrics, enhanced_metrics):
    """
    Compare baseline and enhanced models.
    
    Args:
        baseline_metrics (dict): Baseline model metrics
        enhanced_metrics (dict): Enhanced model metrics
        
    Returns:
        dict: Comparison metrics
    """
    print("Comparing models...")
    
    # Calculate improvement percentages
    accuracy_improvement = (enhanced_metrics['enhanced']['accuracy'] - baseline_metrics['accuracy']) / baseline_metrics['accuracy'] * 100
    precision_improvement = (enhanced_metrics['enhanced']['precision'] - baseline_metrics['precision']) / baseline_metrics['precision'] * 100 if baseline_metrics['precision'] > 0 else float('inf')
    recall_improvement = (enhanced_metrics['enhanced']['recall'] - baseline_metrics['recall']) / baseline_metrics['recall'] * 100 if baseline_metrics['recall'] > 0 else float('inf')
    f1_improvement = (enhanced_metrics['enhanced']['f1'] - baseline_metrics['f1']) / baseline_metrics['f1'] * 100 if baseline_metrics['f1'] > 0 else float('inf')
    auc_improvement = (enhanced_metrics['enhanced']['auc'] - baseline_metrics['auc']) / baseline_metrics['auc'] * 100
    
    # Print comparison
    print("\nModel Comparison:")
    print(f"Accuracy: {baseline_metrics['accuracy']:.4f} -> {enhanced_metrics['enhanced']['accuracy']:.4f} ({accuracy_improvement:.2f}%)")
    print(f"Precision: {baseline_metrics['precision']:.4f} -> {enhanced_metrics['enhanced']['precision']:.4f} ({precision_improvement:.2f}%)")
    print(f"Recall: {baseline_metrics['recall']:.4f} -> {enhanced_metrics['enhanced']['recall']:.4f} ({recall_improvement:.2f}%)")
    print(f"F1 Score: {baseline_metrics['f1']:.4f} -> {enhanced_metrics['enhanced']['f1']:.4f} ({f1_improvement:.2f}%)")
    print(f"AUC: {baseline_metrics['auc']:.4f} -> {enhanced_metrics['enhanced']['auc']:.4f} ({auc_improvement:.2f}%)")
    
    # Check if performance target is met
    target_met = enhanced_metrics['enhanced']['f1'] >= 0.95 or enhanced_metrics['enhanced']['auc'] >= 0.95
    
    if target_met:
        print("\n✅ Performance target of >95% metrics achieved!")
    else:
        print("\n⚠️ Performance target of >95% metrics not yet achieved.")
        print("Consider further enhancements or additional data.")
    
    # Return comparison
    return {
        'baseline': baseline_metrics,
        'enhanced': enhanced_metrics['enhanced'],
        'improvement': {
            'accuracy': accuracy_improvement,
            'precision': precision_improvement,
            'recall': recall_improvement,
            'f1': f1_improvement,
            'auc': auc_improvement
        },
        'target_met': target_met
    }

def plot_comparison(comparison):
    """
    Plot comparison of baseline and enhanced models.
    
    Args:
        comparison (dict): Comparison metrics
        
    Returns:
        str: Path to saved plot
    """
    print("Plotting model comparison...")
    
    # Extract metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    baseline_values = [
        comparison['baseline']['accuracy'],
        comparison['baseline']['precision'],
        comparison['baseline']['recall'],
        comparison['baseline']['f1'],
        comparison['baseline']['auc']
    ]
    enhanced_values = [
        comparison['enhanced']['accuracy'],
        comparison['enhanced']['precision'],
        comparison['enhanced']['recall'],
        comparison['enhanced']['f1'],
        comparison['enhanced']['auc']
    ]
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Set width of bars
    bar_width = 0.35
    index = np.arange(len(metrics))
    
    # Plot bars
    plt.bar(index, baseline_values, bar_width, label='Baseline Model', color='blue', alpha=0.7)
    plt.bar(index + bar_width, enhanced_values, bar_width, label='Enhanced Model', color='red', alpha=0.7)
    
    # Add target line
    plt.axhline(y=0.95, color='green', linestyle='--', label='Target (95%)')
    
    # Add labels and legend
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(index + bar_width / 2, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(baseline_values):
        plt.text(i - 0.1, v + 0.02, f'{v:.3f}', color='blue', fontweight='bold')
    
    for i, v in enumerate(enhanced_values):
        plt.text(i + bar_width - 0.1, v + 0.02, f'{v:.3f}', color='red', fontweight='bold')
    
    # Save plot
    comparison_path = os.path.join(OUTPUT_DIR, 'model_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return comparison_path

def save_evaluation_results(comparison, comparison_plot_path):
    """
    Save evaluation results to a report file.
    
    Args:
        comparison (dict): Comparison metrics
        comparison_plot_path (str): Path to comparison plot
        
    Returns:
        str: Path to report file
    """
    print("Saving evaluation results...")
    
    # Create report
    report = "# Enhanced Migraine Prediction Model Evaluation\n\n"
    
    # Add date and time
    report += f"Evaluation completed on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Add performance summary
    report += "## Performance Summary\n\n"
    
    if comparison['target_met']:
        report += "✅ **Performance target of >95% metrics achieved!**\n\n"
    else:
        report += "⚠️ **Performance target of >95% metrics not yet achieved.**\n\n"
    
    # Add comparison table
    report += "### Model Comparison\n\n"
    report += "| Metric | Baseline Model | Enhanced Model | Improvement |\n"
    report += "|--------|---------------|----------------|-------------|\n"
    report += f"| Accuracy | {comparison['baseline']['accuracy']:.4f} | {comparison['enhanced']['accuracy']:.4f} | {comparison['improvement']['accuracy']:.2f}% |\n"
    report += f"| Precision | {comparison['baseline']['precision']:.4f} | {comparison['enhanced']['precision']:.4f} | {comparison['improvement']['precision']:.2f}% |\n"
    report += f"| Recall | {comparison['baseline']['recall']:.4f} | {comparison['enhanced']['recall']:.4f} | {comparison['improvement']['recall']:.2f}% |\n"
    report += f"| F1 Score | {comparison['baseline']['f1']:.4f} | {comparison['enhanced']['f1']:.4f} | {comparison['improvement']['f1']:.2f}% |\n"
    report += f"| AUC | {comparison['baseline']['auc']:.4f} | {comparison['enhanced']['auc']:.4f} | {comparison['improvement']['auc']:.2f}% |\n\n"
    
    # Add comparison plot
    report += "### Visual Comparison\n\n"
    report += f"![Model Comparison]({os.path.basename(comparison_plot_path)})\n\n"
    
    # Add enhancements summary
    report += "## Enhancements Implemented\n\n"
    report += "1. **Threshold Optimization**\n"
    report += "   - Implemented precision-recall curve analysis\n"
    report += "   - Found optimal threshold to balance precision and recall\n"
    report += "   - Applied cost-sensitive learning\n\n"
    
    report += "2. **Class Balancing**\n"
    report += "   - Applied SMOTE for oversampling minority class\n"
    report += "   - Implemented class weights in loss function\n"
    report += "   - Used focal loss to focus on hard examples\n\n"
    
    report += "3. **Feature Engineering**\n"
    report += "   - Enhanced sleep features with temporal patterns\n"
    report += "   - Improved weather features with pressure change rates\n"
    report += "   - Added stress/diet interaction features\n"
    report += "   - Created cross-domain features\n\n"
    
    report += "4. **Ensemble Methods**\n"
    report += "   - Implemented expert ensemble with domain-specific models\n"
    report += "   - Used stacking ensemble with meta-learner\n"
    report += "   - Applied bagging to reduce variance\n"
    report += "   - Created super ensemble for final predictions\n\n"
    
    # Add recommendations
    report += "## Recommendations for Further Improvement\n\n"
    
    if not comparison['target_met']:
        report += "To achieve the target of >95% performance metrics, consider:\n\n"
    else:
        report += "To further improve the model's performance, consider:\n\n"
    
    report += "1. **Data Collection**\n"
    report += "   - Gather more migraine event data to balance the dataset\n"
    report += "   - Collect additional physiological data\n"
    report += "   - Include medication and treatment response data\n\n"
    
    report += "2. **Advanced Modeling**\n"
    report += "   - Implement attention mechanisms for temporal data\n"
    report += "   - Use transformer-based models for sequence modeling\n"
    report += "   - Explore deep reinforcement learning for personalization\n\n"
    
    report += "3. **Personalization**\n"
    report += "   - Develop user-specific models\n"
    report += "   - Implement online learning for adaptation\n"
    report += "   - Create personalized threshold optimization\n\n"
    
    # Save report
    report_path = os.path.join(OUTPUT_DIR, 'evaluation_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Evaluation report saved to {report_path}")
    
    return report_path

def main():
    """
    Main function to run the evaluation.
    """
    print("Starting enhanced migraine prediction model evaluation...")
    
    # Try to load real data first
    data = load_real_data()
    
    # If real data not available, generate synthetic data
    if data is None:
        print("Using synthetic data for evaluation...")
        data = generate_synthetic_data(n_samples=1000, positive_ratio=0.1)
    
    # Unpack data
    X_sleep, X_weather, X_stress_diet, y = data
    
    # Create and train baseline model
    baseline_model = create_baseline_model(
        input_shape_sleep=X_sleep.shape[1:],
        input_shape_weather=X_weather.shape[1:],
        input_shape_stress_diet=X_stress_diet.shape[1:]
    )
    
    baseline_model, test_data, baseline_pred = train_baseline_model(
        baseline_model, X_sleep, X_weather, X_stress_diet, y
    )
    
    # Evaluate baseline model
    baseline_metrics = evaluate_baseline_model(test_data[3], baseline_pred)
    
    # Run enhanced model
    enhanced_metrics = run_enhanced_model(
        X_sleep, X_weather, X_stress_diet, y,
        baseline_model_path=os.path.join(OUTPUT_DIR, 'baseline_model.keras')
    )
    
    # Compare models
    comparison = compare_models(baseline_metrics, enhanced_metrics)
    
    # Plot comparison
    comparison_plot_path = plot_comparison(comparison)
    
    # Save evaluation results
    report_path = save_evaluation_results(comparison, comparison_plot_path)
    
    print(f"\nEvaluation completed successfully!")
    print(f"Results saved to {OUTPUT_DIR}")
    print(f"Report: {report_path}")
    print(f"Comparison plot: {comparison_plot_path}")

if __name__ == "__main__":
    main()
