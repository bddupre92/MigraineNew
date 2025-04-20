"""
Migraine Prediction Model for Migraine Prediction App

This module implements the complete migraine prediction model using the MoE architecture
with PyGMO optimization. It handles data loading, preprocessing, model training,
evaluation, and prediction.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics, callbacks
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Import MoE components using correct relative paths
from .moe_architecture.experts.sleep_expert import SleepExpert
from .moe_architecture.experts.weather_expert import WeatherExpert
from .moe_architecture.experts.stress_diet_expert import StressDietExpert
from .moe_architecture.gating_network import GatingNetwork, FusionMechanism
from .moe_architecture.pygmo_integration import MigraineMoEModel, ExpertHyperparamOptimization, GatingHyperparamOptimization, EndToEndMoEOptimization

class MigrainePredictionModel:
    """
    Complete Migraine Prediction Model using MoE architecture with PyGMO optimization.
    
    Attributes:
        data_dir (str): Directory containing the data files
        output_dir (str): Directory to save model outputs
        config (dict): Configuration parameters for the model
        seed (int): Random seed for reproducibility
    """
    
    def __init__(self, data_dir, output_dir='./output', config=None, seed=None):
        """
        Initialize the Migraine Prediction Model.
        
        Args:
            data_dir (str): Directory containing the data files
            output_dir (str): Directory to save model outputs
            config (dict): Configuration parameters for the model
            seed (int): Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.seed = seed
        
        # Default configuration
        self.config = {
            'test_size': 0.2,
            'val_size': 0.2,
            'sequence_length': 7,
            'batch_size': 32,
            'epochs': 50,
            'learning_rate': 0.001,
            'early_stopping_patience': 10,
            'load_balance_coef': 0.01,
            'optimize_experts': True,
            'optimize_gating': True,
            'optimize_end_to_end': True,
            'expert_pop_size': 20,
            'expert_generations': 10,
            'gating_pop_size': 20,
            'gating_generations': 10,
            'e2e_pop_size': 20,
            'e2e_generations': 10
        }
        
        # Update with provided configuration
        if config:
            self.config.update(config)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model components
        self.sleep_expert = None
        self.weather_expert = None
        self.stress_diet_expert = None
        self.gating_network = None
        self.fusion_mechanism = None
        self.model = None
        
        # Set random seeds for reproducibility
        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)
    
    def load_data(self):
        """
        Load and preprocess data for the migraine prediction model.

        Returns:
            tuple: (X_train_list, y_train, X_val_list, y_val, X_test_list, y_test)
        """
        print("Loading data...")

        # Load combined dataset
        combined_data_path = os.path.join(self.data_dir, 'combined_data.csv')
        print(f"DEBUG: Combined data path determined as: {combined_data_path}") # Add print statement
        combined_data = pd.read_csv(combined_data_path)

        # Extract target variable
        y = combined_data['next_day_migraine'].values

        # Split data into train, validation, and test sets
        # First split into train+val and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            combined_data, y, test_size=self.config['test_size'],
            random_state=self.seed, stratify=y
        )

        # Then split train+val into train and validation
        val_size_adjusted = self.config['val_size'] / (1 - self.config['test_size'])
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted,
            random_state=self.seed, stratify=y_train_val
        )

        print(f"Data split: Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

        # Preprocess data for each expert
        X_train_list, X_val_list, X_test_list = self._preprocess_data(X_train, X_val, X_test)

        # Adjust target variables AND potentially non-sequential inputs to match sequence length
        seq_offset = self.config['sequence_length'] - 1
        if seq_offset > 0:
            print(f"Applying sequence offset: {seq_offset}") # Add print statement for debugging
            # Slice y arrays first
            original_y_len = len(y_train)
            y_train = y_train[seq_offset:]
            y_val = y_val[seq_offset:]
            y_test = y_test[seq_offset:]
            print(f"Sliced y_train from {original_y_len} to {len(y_train)}")

            # Slice X arrays that were NOT sequenced during _preprocess_data (like weather)
            # Assuming X_train_list = [sleep, weather, stress]
            for i in range(len(X_train_list)):
                 if X_train_list[i].shape[0] != len(y_train): # Check against the NEW y_train length
                     print(f"Slicing X_train_list[{i}] from {X_train_list[i].shape[0]} to {len(y_train)}")
                     X_train_list[i] = X_train_list[i][seq_offset:]
                 if X_val_list[i].shape[0] != len(y_val):
                     print(f"Slicing X_val_list[{i}] from {X_val_list[i].shape[0]} to {len(y_val)}")
                     X_val_list[i] = X_val_list[i][seq_offset:]
                 if X_test_list[i].shape[0] != len(y_test):
                      print(f"Slicing X_test_list[{i}] from {X_test_list[i].shape[0]} to {len(y_test)}")
                      X_test_list[i] = X_test_list[i][seq_offset:]

        # --- Add Dtype Logging Here ---
        print("--- Final Shapes and Dtypes before return --- ")
        try:
            print(f"y_train shape: {y_train.shape}, dtype: {y_train.dtype}")
            print(f"X_train_list shapes: {[x.shape for x in X_train_list]}, dtypes: {[x.dtype for x in X_train_list]}")
            print(f"y_val shape: {y_val.shape}, dtype: {y_val.dtype}")
            print(f"X_val_list shapes: {[x.shape for x in X_val_list]}, dtypes: {[x.dtype for x in X_val_list]}")
            print(f"y_test shape: {y_test.shape}, dtype: {y_test.dtype}")
            print(f"X_test_list shapes: {[x.shape for x in X_test_list]}, dtypes: {[x.dtype for x in X_test_list]}")
        except Exception as e:
            print(f"Error printing final shapes/dtypes: {e}")
        print("--- End Final Shapes and Dtypes --- ")
        # --- End Dtype Logging ---

        # Return the potentially modified lists and arrays
        return X_train_list, y_train, X_val_list, y_val, X_test_list, y_test

    def _preprocess_data(self, X_train, X_val, X_test):
        """
        Preprocess data for each expert.
        
        Args:
            X_train (DataFrame): Training data
            X_val (DataFrame): Validation data
            X_test (DataFrame): Test data
            
        Returns:
            tuple: (X_train_list, X_val_list, X_test_list)
        """
        # Create temporary expert instances for preprocessing
        temp_sleep_expert = SleepExpert()
        temp_weather_expert = WeatherExpert()
        temp_stress_diet_expert = StressDietExpert()
        
        # Extract and preprocess sleep data
        sleep_features = [
            'total_sleep_hours', 'deep_sleep_pct', 'rem_sleep_pct',
            'light_sleep_pct', 'awake_time_mins', 'sleep_quality'
        ]
        
        # Create sequences for sleep data
        X_train_sleep = temp_sleep_expert.preprocess(
            X_train[sleep_features], 
            sequence_length=self.config['sequence_length']
        )
        X_val_sleep = temp_sleep_expert.preprocess(
            X_val[sleep_features], 
            sequence_length=self.config['sequence_length']
        )
        X_test_sleep = temp_sleep_expert.preprocess(
            X_test[sleep_features], 
            sequence_length=self.config['sequence_length']
        )
        
        # Extract and preprocess weather data
        weather_features = [
            'temperature', 'humidity', 'pressure', 'pressure_change_24h'
        ]
        
        X_train_weather = temp_weather_expert.preprocess(X_train[weather_features])
        X_val_weather = temp_weather_expert.preprocess(X_val[weather_features])
        X_test_weather = temp_weather_expert.preprocess(X_test[weather_features])
        
        # Extract and preprocess stress/diet data
        stress_diet_features = [
            'stress_level', 'alcohol_consumed', 'caffeine_consumed',
            'chocolate_consumed', 'processed_food_consumed', 'water_consumed_liters'
        ]
        
        # Create sequences for stress/diet data
        X_train_stress_diet = temp_stress_diet_expert.preprocess(
            X_train[stress_diet_features], 
            sequence_length=self.config['sequence_length']
        )
        X_val_stress_diet = temp_stress_diet_expert.preprocess(
            X_val[stress_diet_features], 
            sequence_length=self.config['sequence_length']
        )
        X_test_stress_diet = temp_stress_diet_expert.preprocess(
            X_test[stress_diet_features], 
            sequence_length=self.config['sequence_length']
        )
        
        # Create lists of inputs for each set
        X_train_list = [X_train_sleep, X_train_weather, X_train_stress_diet]
        X_val_list = [X_val_sleep, X_val_weather, X_val_stress_diet]
        X_test_list = [X_test_sleep, X_test_weather, X_test_stress_diet]
        
        return X_train_list, X_val_list, X_test_list
    
    def optimize_experts(self, X_train_list, y_train, X_val_list, y_val):
        """
        Optimize hyperparameters for each expert using PyGMO.
        
        Args:
            X_train_list (list): List of training data for each expert
            y_train (array): Training target variable
            X_val_list (list): List of validation data for each expert
            y_val (array): Validation target variable
            
        Returns:
            list: List of optimized expert configurations
        """
        print("Optimizing expert hyperparameters...")
        
        expert_configs = []
        
        # Optimize Sleep Expert
        print("Optimizing Sleep Expert...")
        sleep_optimizer = ExpertHyperparamOptimization(
            expert_type='sleep',
            train_data=(X_train_list[0], y_train),
            val_data=(X_val_list[0], y_val),
            seed=self.seed
        )
        
        sleep_config, sleep_fitness = sleep_optimizer.optimize(
            pop_size=self.config['expert_pop_size'],
            generations=self.config['expert_generations'],
            algorithm='de'
        )
        
        print(f"Sleep Expert optimized: {sleep_config}, fitness: {-sleep_fitness}")
        expert_configs.append(sleep_config)
        
        # Optimize Weather Expert
        print("Optimizing Weather Expert...")
        weather_optimizer = ExpertHyperparamOptimization(
            expert_type='weather',
            train_data=(X_train_list[1], y_train),
            val_data=(X_val_list[1], y_val),
            seed=self.seed
        )
        
        weather_config, weather_fitness = weather_optimizer.optimize(
            pop_size=self.config['expert_pop_size'],
            generations=self.config['expert_generations'],
            algorithm='cmaes'
        )
        
        print(f"Weather Expert optimized: {weather_config}, fitness: {-weather_fitness}")
        expert_configs.append(weather_config)
        
        # Optimize Stress/Diet Expert
        print("Optimizing Stress/Diet Expert...")
        stress_diet_optimizer = ExpertHyperparamOptimization(
            expert_type='stress_diet',
            train_data=(X_train_list[2], y_train),
            val_data=(X_val_list[2], y_val),
            seed=self.seed
        )
        
        stress_diet_config, stress_diet_fitness = stress_diet_optimizer.optimize(
            pop_size=self.config['expert_pop_size'],
            generations=self.config['expert_generations'],
            algorithm='de'
        )
        
        print(f"Stress/Diet Expert optimized: {stress_diet_config}, fitness: {-stress_diet_fitness}")
        expert_configs.append(stress_diet_config)
        
        return expert_configs
    
    def optimize_gating(self, expert_configs, X_train_list, y_train, X_val_list, y_val):
        """
        Optimize hyperparameters for the gating network using PyGMO.
        
        Args:
            expert_configs (list): List of expert configurations
            X_train_list (list): List of training data for each expert
            y_train (array): Training target variable
            X_val_list (list): List of validation data for each expert
            y_val (array): Validation target variable
            
        Returns:
            dict: Optimized gating network configuration
        """
        print("Optimizing gating network hyperparameters...")
        
        # Create expert models with optimized configurations
        experts = [
            SleepExpert(config=expert_configs[0]),
            WeatherExpert(config=expert_configs[1]),
            StressDietExpert(config=expert_configs[2])
        ]
        
        # Optimize Gating Network
        gating_optimizer = GatingHyperparamOptimization(
            experts=experts,
            train_data=(X_train_list, y_train),
            val_data=(X_val_list, y_val),
            seed=self.seed
        )
        
        gating_config, gating_fitness = gating_optimizer.optimize(
            pop_size=self.config['gating_pop_size'],
            generations=self.config['gating_generations'],
            algorithm='pso'
        )
        
        print(f"Gating Network optimized: {gating_config}, fitness: {-gating_fitness}")
        
        return gating_config
    
    def optimize_end_to_end(self, expert_configs, gating_config, X_train_list, y_train, X_val_list, y_val):
        """
        Perform end-to-end optimization of the MoE model using PyGMO.
        
        Args:
            expert_configs (list): List of expert configurations
            gating_config (dict): Gating network configuration
            X_train_list (list): List of training data for each expert
            y_train (array): Training target variable
            X_val_list (list): List of validation data for each expert
            y_val (array): Validation target variable
            
        Returns:
            dict: Optimized end-to-end configuration
        """
        print("Performing end-to-end optimization...")
        
        # Optimize End-to-End
        e2e_optimizer = EndToEndMoEOptimization(
            expert_configs=expert_configs,
            gating_config=gating_config,
            train_data=(X_train_list, y_train),
            val_data=(X_val_list, y_val),
            seed=self.seed
        )
        
        e2e_config, e2e_fitness = e2e_optimizer.optimize(
            pop_size=self.config['e2e_pop_size'],
            generations=self.config['e2e_generations']
        )
        
        print(f"End-to-End optimized: {e2e_config}, fitness: {e2e_fitness}")
        
        return e2e_config
    
    def build_model(self, expert_configs, gating_config, e2e_config=None):
        """
        Build the MoE model with the optimized configurations.
        
        Args:
            expert_configs (list): List of expert configurations
            gating_config (dict): Gating network configuration
            e2e_config (dict): End-to-end configuration
            
        Returns:
            Model: Built MoE model
        """
        print("Building MoE model...")
        
        # Create expert models
        self.sleep_expert = SleepExpert(config=expert_configs[0])
        self.weather_expert = WeatherExpert(config=expert_configs[1])
        self.stress_diet_expert = StressDietExpert(config=expert_configs[2])
        
        experts = [self.sleep_expert, self.weather_expert, self.stress_diet_expert]
        
        # Create gating network
        self.gating_network = GatingNetwork(
            num_experts=len(experts),
            config={
                'hidden_size': gating_config['gate_hidden_size'],
                'top_k': gating_config['gate_top_k'],
                'dropout_rate': 0.2  # Fixed dropout rate
            }
        )
        
        # Create fusion mechanism
        self.fusion_mechanism = FusionMechanism(top_k=gating_config['gate_top_k'])
        
        # Create MoE model
        self.model = MigraineMoEModel(
            experts=experts,
            gating_network=self.gating_network,
            fusion_mechanism=self.fusion_mechanism,
            load_balance_coef=gating_config['load_balance_coef']
        )
        
        # Apply end-to-end configuration if provided
        if e2e_config:
            # Add L2 regularization
            for layer in self.model.layers:
                if hasattr(layer, 'kernel_regularizer'):
                    layer.kernel_regularizer = tf.keras.regularizers.l2(e2e_config['l2_regularization'])
            
            # Compile with optimized learning rate
            self.model.compile(
                optimizer=optimizers.Adam(learning_rate=e2e_config['learning_rate'], clipvalue=0.5),
                loss=losses.BinaryCrossentropy(),
                metrics=[
                    metrics.AUC(name='auc'),
                    metrics.Recall(name='recall'),
                    metrics.Precision(name='precision'),
                    metrics.F1Score(name='f1_score')
                ]
            )
            
            # Update batch size
            self.config['batch_size'] = int(e2e_config['batch_size'])
        else:
            # Compile with default configuration
            self.model.compile(
                optimizer=optimizers.Adam(learning_rate=1e-6, clipvalue=0.5),  # Use clipvalue instead of clipnorm
                loss=losses.BinaryCrossentropy(),
                metrics=[
                    metrics.AUC(name='auc'),
                    metrics.Recall(name='recall'),
                    metrics.Precision(name='precision'),
                    metrics.F1Score(name='f1_score')
                ]
            )
        
        return self.model
    
    def train_model(self, X_train_list, y_train, X_val_list, y_val):
        """
        Train the MoE model.
        
        Args:
            X_train_list (list): List of training data for each expert
            y_train (array): Training target variable
            X_val_list (list): List of validation data for each expert
            y_val (array): Validation target variable
            
        Returns:
            History: Training history
        """
        print("Training MoE model...")
        
        # Create callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_auc',
            patience=self.config['early_stopping_patience'],
            restore_best_weights=True,
            mode='max'
        )
        
        model_checkpoint = callbacks.ModelCheckpoint(
            filepath=os.path.join(self.output_dir, 'best_model.h5'),
            monitor='val_auc',
            save_best_only=True,
            mode='max'
        )
        
        # --- Pre-fit Data Checks ---
        print("--- Running Pre-fit Data Checks ---")
        try:
            print(f"y_train | Shape: {y_train.shape}, Dtype: {y_train.dtype}, NaNs: {np.isnan(y_train).sum()}, Infs: {np.isinf(y_train).sum()}")
            for i, X_train in enumerate(X_train_list):
                print(f"X_train_{i} | Shape: {X_train.shape}, Dtype: {X_train.dtype}, NaNs: {np.isnan(X_train).sum()}, Infs: {np.isinf(X_train).sum()}")
            
            print(f"y_val | Shape: {y_val.shape}, Dtype: {y_val.dtype}, NaNs: {np.isnan(y_val).sum()}, Infs: {np.isinf(y_val).sum()}")
            for i, X_val in enumerate(X_val_list):
                print(f"X_val_{i} | Shape: {X_val.shape}, Dtype: {X_val.dtype}, NaNs: {np.isnan(X_val).sum()}, Infs: {np.isinf(X_val).sum()}")
        except Exception as e:
            print(f"Error during pre-fit data checks: {e}")
        print("--- Finished Pre-fit Data Checks ---")
        # --- End Pre-fit Data Checks ---

        # Train the model
        history = self.model.fit(
            X_train_list, y_train,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            validation_data=(X_val_list, y_val),
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        
        # Save training history
        self._save_training_history(history)
        
        return history
    
    def evaluate_model(self, X_test_list, y_test):
        """
        Evaluate the MoE model on the test set.
        
        Args:
            X_test_list (list): List of test data for each expert
            y_test (array): Test target variable
            
        Returns:
            dict: Evaluation metrics
        """
        print("Evaluating MoE model...")
        
        # Evaluate the model
        test_results = self.model.evaluate(X_test_list, y_test, verbose=1)
        
        # Get predictions
        y_pred_prob = self.model.predict(X_test_list)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # Calculate additional metrics
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        # Save evaluation results
        self._save_evaluation_results(test_results, conf_matrix, class_report, fpr, tpr, roc_auc)
        
        # Create a dictionary of metrics
        metrics_dict = {
            'loss': test_results[0],
            'auc': test_results[1],
            'recall': test_results[2],
            'precision': test_results[3],
            'f1_score': test_results[4],
            'conf_matrix': conf_matrix,
            'class_report': class_report,
            'roc_auc': roc_auc
        }
        
        return metrics_dict
    
    def _save_training_history(self, history):
        """
        Save training history plots.
        
        Args:
            history (History): Training history
        """
        # Create directory for plots
        plots_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot training & validation accuracy
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['auc'])
        plt.plot(history.history['val_auc'])
        plt.title('Model AUC')
        plt.ylabel('AUC')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        # Plot training & validation loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'training_history.png'))
        plt.close()
        
        # Save history to CSV
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(os.path.join(self.output_dir, 'training_history.csv'), index=False)
    
    def _save_evaluation_results(self, test_results, conf_matrix, class_report, fpr, tpr, roc_auc):
        """
        Save evaluation results and plots.
        
        Args:
            test_results (list): Test results
            conf_matrix (array): Confusion matrix
            class_report (dict): Classification report
            fpr (array): False positive rate
            tpr (array): True positive rate
            roc_auc (float): ROC AUC
        """
        # Create directory for plots
        plots_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Save test results
        metrics_names = ['loss', 'auc', 'recall', 'precision', 'f1_score']
        test_results_dict = {name: value for name, value in zip(metrics_names, test_results)}
        test_results_df = pd.DataFrame([test_results_dict])
        test_results_df.to_csv(os.path.join(self.output_dir, 'test_results.csv'), index=False)
        
        # Save classification report
        class_report_df = pd.DataFrame(class_report).transpose()
        class_report_df.to_csv(os.path.join(self.output_dir, 'classification_report.csv'))
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Migraine', 'Migraine'],
                    yticklabels=['No Migraine', 'Migraine'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'))
        plt.close()
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'roc_curve.png'))
        plt.close()
    
    def save_model(self):
        """
        Save the trained model and configurations.
        
        Returns:
            str: Path to the saved model
        """
        print("Saving model...")
        
        # Save model
        model_path = os.path.join(self.output_dir, 'migraine_prediction_model')
        self.model.save(model_path)
        
        # Save configurations
        config_path = os.path.join(self.output_dir, 'model_config.json')
        with open(config_path, 'w') as f:
            import json
            json.dump(self.config, f, indent=4)
        
        return model_path
    
    def load_model(self, model_path):
        """
        Load a trained model.
        
        Args:
            model_path (str): Path to the saved model
            
        Returns:
            Model: Loaded model
        """
        print(f"Loading model from {model_path}...")
        
        # Load model
        self.model = tf.keras.models.load_model(model_path)
        
        return self.model
    
    def predict(self, X):
        """
        Make predictions with the trained model.
        
        Args:
            X (list): List of input data for each expert
            
        Returns:
            array: Predicted probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train_model() or load_model() first.")
        
        return self.model.predict(X)
    
    def run_pipeline(self):
        """
        Run the complete pipeline: load data, optimize, build, train, evaluate, and save model.
        
        Returns:
            dict: Evaluation metrics
        """
        # Load data
        X_train_list, y_train, X_val_list, y_val, X_test_list, y_test = self.load_data()
        
        # Optimize experts
        if self.config['optimize_experts']:
            expert_configs = self.optimize_experts(X_train_list, y_train, X_val_list, y_val)
        else:
            # Use default configurations
            expert_configs = [
                {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 128, 'dropout_rate': 0.3, 'output_dim': 64},
                {'hidden_units': 128, 'activation': 'relu', 'dropout_rate': 0.3, 'output_dim': 64},
                {'embedding_dim': 64, 'num_heads': 4, 'transformer_dim': 64, 'dropout_rate': 0.2, 'output_dim': 64}
            ]
        
        # Optimize gating
        if self.config['optimize_gating']:
            gating_config = self.optimize_gating(expert_configs, X_train_list, y_train, X_val_list, y_val)
        else:
            # Use default configuration
            gating_config = {
                'gate_hidden_size': 128,
                'gate_top_k': 2,
                'load_balance_coef': 0.01
            }
        
        # Optimize end-to-end
        if self.config['optimize_end_to_end']:
            e2e_config = self.optimize_end_to_end(expert_configs, gating_config, X_train_list, y_train, X_val_list, y_val)
        else:
            e2e_config = None
        
        # Build model
        self.build_model(expert_configs, gating_config, e2e_config)
        
        # Train model
        self.train_model(X_train_list, y_train, X_val_list, y_val)
        
        # Evaluate model
        metrics = self.evaluate_model(X_test_list, y_test)
        
        # Save model
        self.save_model()
        
        return metrics


if __name__ == "__main__":
    # Example usage
    model = MigrainePredictionModel(
        data_dir='./data',
        output_dir='./output',
        config={
            'optimize_experts': True,
            'optimize_gating': True,
            'optimize_end_to_end': True,
            'expert_generations': 5,  # Reduced for testing
            'gating_generations': 5,  # Reduced for testing
            'e2e_generations': 5      # Reduced for testing
        },
        seed=42
    )
    
    # Run the complete pipeline
    metrics = model.run_pipeline()
    
    # Print evaluation metrics
    print("\nEvaluation Metrics:")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
