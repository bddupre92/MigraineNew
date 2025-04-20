import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os
import joblib

class EnsembleModels:
    """
    A class for implementing ensemble methods to improve migraine prediction performance.
    Combines multiple models using various ensemble techniques.
    """
    
    def __init__(self, output_dir='output'):
        """
        Initialize the EnsembleModels.
        
        Args:
            output_dir (str): Directory to save results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.models = {}
        self.ensemble_models = {}
    
    def create_base_models(self, input_shape, model_configs=None):
        """
        Create base models for ensemble.
        
        Args:
            input_shape (tuple): Shape of input data
            model_configs (list, optional): List of model configurations
            
        Returns:
            dict: Dictionary of base models
        """
        if model_configs is None:
            # Default model configurations
            model_configs = [
                {'name': 'model_1', 'units': [64, 32], 'dropout': 0.3, 'learning_rate': 0.001},
                {'name': 'model_2', 'units': [128, 64], 'dropout': 0.4, 'learning_rate': 0.0005},
                {'name': 'model_3', 'units': [32, 16], 'dropout': 0.2, 'learning_rate': 0.002},
                {'name': 'model_4', 'units': [64, 64, 32], 'dropout': 0.3, 'learning_rate': 0.001},
                {'name': 'model_5', 'units': [128, 32], 'dropout': 0.5, 'learning_rate': 0.0003}
            ]
        
        print("Creating base models...")
        
        for config in model_configs:
            name = config['name']
            units = config['units']
            dropout = config['dropout']
            learning_rate = config['learning_rate']
            
            # Create model
            model = models.Sequential(name=name)
            
            # Input layer
            model.add(layers.Input(shape=input_shape))
            
            # Hidden layers
            for i, unit in enumerate(units):
                model.add(layers.Dense(unit, activation='relu', name=f'{name}_dense_{i+1}'))
                model.add(layers.BatchNormalization(name=f'{name}_bn_{i+1}'))
                model.add(layers.Dropout(dropout, name=f'{name}_dropout_{i+1}'))
            
            # Output layer
            model.add(layers.Dense(1, activation='sigmoid', name=f'{name}_output'))
            
            # Compile model
            model.compile(
                optimizer=optimizers.Adam(learning_rate=learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Store model
            self.models[name] = model
            print(f"Created model: {name}")
        
        return self.models
    
    def create_expert_ensemble(self, sleep_expert, weather_expert, physio_expert, stress_diet_expert):
        """
        Create an ensemble of domain-specific expert models.
        
        Args:
            sleep_expert (tf.keras.Model): Sleep expert model
            weather_expert (tf.keras.Model): Weather expert model
            physio_expert (tf.keras.Model): Physiological expert model
            stress_diet_expert (tf.keras.Model): Stress/diet expert model
            
        Returns:
            tf.keras.Model: Ensemble model
        """
        print("Creating expert ensemble model...")
        
        # Get output from each expert
        sleep_output = sleep_expert.output
        weather_output = weather_expert.output
        physio_output = physio_expert.output
        stress_diet_output = stress_diet_expert.output
        
        # Concatenate expert outputs
        concat = layers.Concatenate(name='expert_concat')([
            sleep_output, weather_output, physio_output, stress_diet_output
        ])
        
        # Add ensemble layers
        x = layers.Dense(32, activation='relu', name='ensemble_dense_1')(concat)
        x = layers.BatchNormalization(name='ensemble_bn_1')(x)
        x = layers.Dropout(0.3, name='ensemble_dropout_1')(x)
        x = layers.Dense(16, activation='relu', name='ensemble_dense_2')(x)
        x = layers.BatchNormalization(name='ensemble_bn_2')(x)
        x = layers.Dropout(0.3, name='ensemble_dropout_2')(x)
        
        # Output layer
        output = layers.Dense(1, activation='sigmoid', name='ensemble_output')(x)
        
        # Create model
        ensemble_model = models.Model(
            inputs=[
                sleep_expert.input, 
                weather_expert.input, 
                physio_expert.input, 
                stress_diet_expert.input
            ],
            outputs=output,
            name='expert_ensemble'
        )
        
        # Compile model
        ensemble_model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Store model
        self.ensemble_models['expert_ensemble'] = ensemble_model
        print("Created expert ensemble model")
        
        return ensemble_model
    
    def create_voting_ensemble(self, X_train, y_train, voting='soft'):
        """
        Create a voting ensemble using scikit-learn.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            voting (str): Voting type ('hard' or 'soft')
            
        Returns:
            VotingClassifier: Voting ensemble model
        """
        print(f"Creating voting ensemble with {voting} voting...")
        
        # Create base classifiers
        classifiers = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('svm', SVC(probability=True, random_state=42)),
            ('lr', LogisticRegression(random_state=42))
        ]
        
        # Create voting ensemble
        voting_ensemble = VotingClassifier(
            estimators=classifiers,
            voting=voting
        )
        
        # Fit model
        voting_ensemble.fit(X_train, y_train)
        
        # Store model
        self.ensemble_models['voting_ensemble'] = voting_ensemble
        print("Created voting ensemble model")
        
        return voting_ensemble
    
    def create_stacking_ensemble(self, X_train, y_train):
        """
        Create a stacking ensemble using scikit-learn.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            
        Returns:
            StackingClassifier: Stacking ensemble model
        """
        print("Creating stacking ensemble...")
        
        # Create base classifiers
        classifiers = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('svm', SVC(probability=True, random_state=42))
        ]
        
        # Create stacking ensemble with logistic regression meta-classifier
        stacking_ensemble = StackingClassifier(
            estimators=classifiers,
            final_estimator=LogisticRegression(random_state=42),
            cv=5
        )
        
        # Fit model
        stacking_ensemble.fit(X_train, y_train)
        
        # Store model
        self.ensemble_models['stacking_ensemble'] = stacking_ensemble
        print("Created stacking ensemble model")
        
        return stacking_ensemble
    
    def create_bagging_ensemble(self, input_shape, n_models=5):
        """
        Create a bagging ensemble of neural networks.
        
        Args:
            input_shape (tuple): Shape of input data
            n_models (int): Number of models in ensemble
            
        Returns:
            list: List of bagged models
        """
        print(f"Creating bagging ensemble with {n_models} models...")
        
        bagged_models = []
        
        for i in range(n_models):
            # Create model
            model = models.Sequential(name=f'bagged_model_{i+1}')
            
            # Input layer
            model.add(layers.Input(shape=input_shape))
            
            # Hidden layers
            model.add(layers.Dense(64, activation='relu', name=f'bagged_{i+1}_dense_1'))
            model.add(layers.BatchNormalization(name=f'bagged_{i+1}_bn_1'))
            model.add(layers.Dropout(0.3, name=f'bagged_{i+1}_dropout_1'))
            model.add(layers.Dense(32, activation='relu', name=f'bagged_{i+1}_dense_2'))
            model.add(layers.BatchNormalization(name=f'bagged_{i+1}_bn_2'))
            model.add(layers.Dropout(0.3, name=f'bagged_{i+1}_dropout_2'))
            
            # Output layer
            model.add(layers.Dense(1, activation='sigmoid', name=f'bagged_{i+1}_output'))
            
            # Compile model
            model.compile(
                optimizer=optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            bagged_models.append(model)
        
        # Store models
        self.ensemble_models['bagging_ensemble'] = bagged_models
        print(f"Created {n_models} models for bagging ensemble")
        
        return bagged_models
    
    def train_base_models(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=50):
        """
        Train base models.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation labels
            batch_size (int): Batch size
            epochs (int): Number of epochs
            
        Returns:
            dict: Dictionary of training histories
        """
        print("Training base models...")
        
        histories = {}
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Store history
            histories[name] = history.history
            
            # Evaluate model
            loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
            print(f"{name} - Validation Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            
            # Save model
            model.save(os.path.join(self.output_dir, f'{name}.keras'))
        
        return histories
    
    def train_expert_ensemble(self, X_train_list, y_train, X_val_list, y_val, batch_size=32, epochs=50):
        """
        Train expert ensemble model.
        
        Args:
            X_train_list (list): List of training features for each expert
            y_train (np.ndarray): Training labels
            X_val_list (list): List of validation features for each expert
            y_val (np.ndarray): Validation labels
            batch_size (int): Batch size
            epochs (int): Number of epochs
            
        Returns:
            dict: Training history
        """
        print("Training expert ensemble model...")
        
        # Get expert ensemble model
        ensemble_model = self.ensemble_models.get('expert_ensemble')
        
        if ensemble_model is None:
            print("Expert ensemble model not found. Please create it first.")
            return None
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        history = ensemble_model.fit(
            X_train_list, y_train,
            validation_data=(X_val_list, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate model
        loss, accuracy = ensemble_model.evaluate(X_val_list, y_val, verbose=0)
        print(f"Expert Ensemble - Validation Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Save model
        ensemble_model.save(os.path.join(self.output_dir, 'expert_ensemble.keras'))
        
        return history.history
    
    def train_bagging_ensemble(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=50):
        """
        Train bagging ensemble models.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation labels
            batch_size (int): Batch size
            epochs (int): Number of epochs
            
        Returns:
            list: List of training histories
        """
        print("Training bagging ensemble models...")
        
        # Get bagging ensemble models
        bagged_models = self.ensemble_models.get('bagging_ensemble')
        
        if bagged_models is None:
            print("Bagging ensemble models not found. Please create them first.")
            return None
        
        histories = []
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        for i, model in enumerate(bagged_models):
            print(f"Training bagged model {i+1}...")
            
            # Create bootstrap sample
            indices = np.random.choice(len(X_train), len(X_train), replace=True)
            X_bootstrap = X_train[indices]
            y_bootstrap = y_train[indices]
            
            # Train model
            history = model.fit(
                X_bootstrap, y_bootstrap,
                validation_data=(X_val, y_val),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=[early_stopping],
                verbose=0
            )
            
            histories.append(history.history)
            
            # Evaluate model
            loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
            print(f"Bagged Model {i+1} - Validation Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            
            # Save model
            model.save(os.path.join(self.output_dir, f'bagged_model_{i+1}.keras'))
        
        return histories
    
    def predict_with_bagging(self, X_test):
        """
        Make predictions using bagging ensemble.
        
        Args:
            X_test (np.ndarray): Test features
            
        Returns:
            np.ndarray: Ensemble predictions
        """
        # Get bagging ensemble models
        bagged_models = self.ensemble_models.get('bagging_ensemble')
        
        if bagged_models is None:
            print("Bagging ensemble models not found. Please create and train them first.")
            return None
        
        # Make predictions with each model
        predictions = []
        for model in bagged_models:
            pred = model.predict(X_test, verbose=0)
            predictions.append(pred)
        
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        
        return ensemble_pred
    
    def save_sklearn_ensemble(self, ensemble_name):
        """
        Save scikit-learn ensemble model.
        
        Args:
            ensemble_name (str): Name of ensemble model to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Get ensemble model
        ensemble_model = self.ensemble_models.get(ensemble_name)
        
        if ensemble_model is None:
            print(f"Ensemble model '{ensemble_name}' not found.")
            return False
        
        # Save model
        joblib.dump(ensemble_model, os.path.join(self.output_dir, f'{ensemble_name}.joblib'))
        print(f"Saved {ensemble_name} to {os.path.join(self.output_dir, f'{ensemble_name}.joblib')}")
        
        return True
    
    def load_sklearn_ensemble(self, ensemble_name):
        """
        Load scikit-learn ensemble model.
        
        Args:
            ensemble_name (str): Name of ensemble model to load
            
        Returns:
            object: Loaded ensemble model
        """
        # Load model
        model_path = os.path.join(self.output_dir, f'{ensemble_name}.joblib')
        
        if not os.path.exists(model_path):
            print(f"Model file '{model_path}' not found.")
            return None
        
        ensemble_model = joblib.load(model_path)
        
        # Store model
        self.ensemble_models[ensemble_name] = ensemble_model
        print(f"Loaded {ensemble_name} from {model_path}")
        
        return ensemble_model
    
    def evaluate_ensemble(self, ensemble_name, X_test, y_test):
        """
        Evaluate ensemble model.
        
        Args:
            ensemble_name (str): Name of ensemble model to evaluate
            X_test (np.ndarray or list): Test features
            y_test (np.ndarray): Test labels
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        print(f"Evaluating {ensemble_name}...")
        
        # Get ensemble model
        ensemble_model = self.ensemble_models.get(ensemble_name)
        
        if ensemble_model is None:
            print(f"Ensemble model '{ensemble_name}' not found.")
            return None
        
        # Make predictions
        if ensemble_name == 'expert_ensemble':
            # Expert ensemble expects a list of inputs
            if not isinstance(X_test, list):
                print("Expert ensemble expects a list of inputs.")
                return None
            
            y_pred = ensemble_model.predict(X_test, verbose=0)
        elif ensemble_name == 'bagging_ensemble':
            # Bagging ensemble uses custom prediction method
            y_pred = self.predict_with_bagging(X_test)
        else:
            # Scikit-learn ensembles
            y_pred = ensemble_model.predict_proba(X_test)[:, 1]
        
        # Convert predictions to binary
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = np.mean(y_pred_binary == y_test)
        precision = precision_score(y_test, y_pred_binary)
        recall = recall_score(y_test, y_pred_binary)
        f1 = f1_score(y_test, y_pred_binary)
        auc = roc_auc_score(y_test, y_pred)
        
        # Print metrics
        print(f"{ensemble_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'y_pred': y_pred,
            'y_pred_binary': y_pred_binary
        }
    
    def compare_ensembles(self, ensemble_metrics, save_path=None):
        """
        Compare performance of different ensemble models.
        
        Args:
            ensemble_metrics (dict): Dictionary of evaluation metrics for each ensemble
            save_path (str, optional): Path to save comparison plot
            
        Returns:
            dict: Dictionary of best ensemble for each metric
        """
        print("Comparing ensemble models...")
        
        # Extract metrics
        ensembles = list(ensemble_metrics.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        
        # Create comparison plot
        plt.figure(figsize=(12, 8))
        
        # Set width of bars
        bar_width = 0.15
        index = np.arange(len(metrics))
        
        # Plot bars for each ensemble
        for i, ensemble in enumerate(ensembles):
            values = [ensemble_metrics[ensemble][metric] for metric in metrics]
            plt.bar(index + i * bar_width, values, bar_width, label=ensemble)
        
        # Add labels and legend
        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.title('Ensemble Models Comparison')
        plt.xticks(index + bar_width * (len(ensembles) - 1) / 2, metrics)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        plt.close()
        
        # Find best ensemble for each metric
        best_ensembles = {}
        for metric in metrics:
            best_ensemble = max(ensembles, key=lambda x: ensemble_metrics[x][metric])
            best_score = ensemble_metrics[best_ensemble][metric]
            best_ensembles[metric] = (best_ensemble, best_score)
            print(f"Best ensemble for {metric}: {best_ensemble} ({best_score:.4f})")
        
        return best_ensembles
    
    def create_super_ensemble(self, ensemble_names, X_train, y_train):
        """
        Create a super ensemble by combining predictions from multiple ensembles.
        
        Args:
            ensemble_names (list): List of ensemble models to combine
            X_train (np.ndarray or list): Training features
            y_train (np.ndarray): Training labels
            
        Returns:
            object: Super ensemble model
        """
        print(f"Creating super ensemble from {ensemble_names}...")
        
        # Check if all ensembles exist
        for name in ensemble_names:
            if name not in self.ensemble_models:
                print(f"Ensemble model '{name}' not found.")
                return None
        
        # Get predictions from each ensemble
        ensemble_preds = []
        for name in ensemble_names:
            ensemble_model = self.ensemble_models[name]
            
            if name == 'expert_ensemble':
                # Expert ensemble expects a list of inputs
                if not isinstance(X_train, list):
                    print("Expert ensemble expects a list of inputs.")
                    return None
                
                pred = ensemble_model.predict(X_train, verbose=0)
            elif name == 'bagging_ensemble':
                # Bagging ensemble uses custom prediction method
                pred = self.predict_with_bagging(X_train)
            else:
                # Scikit-learn ensembles
                pred = ensemble_model.predict_proba(X_train)[:, 1]
            
            ensemble_preds.append(pred)
        
        # Combine predictions into a single feature matrix
        X_meta = np.column_stack(ensemble_preds)
        
        # Create meta-learner (logistic regression)
        meta_learner = LogisticRegression(random_state=42)
        meta_learner.fit(X_meta, y_train)
        
        # Store super ensemble
        self.ensemble_models['super_ensemble'] = {
            'meta_learner': meta_learner,
            'base_ensembles': ensemble_names
        }
        
        # Save super ensemble
        joblib.dump(self.ensemble_models['super_ensemble'], os.path.join(self.output_dir, 'super_ensemble.joblib'))
        print(f"Saved super ensemble to {os.path.join(self.output_dir, 'super_ensemble.joblib')}")
        
        return self.ensemble_models['super_ensemble']
    
    def predict_with_super_ensemble(self, X_test):
        """
        Make predictions using super ensemble.
        
        Args:
            X_test (np.ndarray or list): Test features
            
        Returns:
            np.ndarray: Super ensemble predictions
        """
        # Get super ensemble
        super_ensemble = self.ensemble_models.get('super_ensemble')
        
        if super_ensemble is None:
            print("Super ensemble not found. Please create it first.")
            return None
        
        meta_learner = super_ensemble['meta_learner']
        base_ensembles = super_ensemble['base_ensembles']
        
        # Get predictions from each base ensemble
        ensemble_preds = []
        for name in base_ensembles:
            ensemble_model = self.ensemble_models[name]
            
            if name == 'expert_ensemble':
                # Expert ensemble expects a list of inputs
                if not isinstance(X_test, list):
                    print("Expert ensemble expects a list of inputs.")
                    return None
                
                pred = ensemble_model.predict(X_test, verbose=0)
            elif name == 'bagging_ensemble':
                # Bagging ensemble uses custom prediction method
                pred = self.predict_with_bagging(X_test)
            else:
                # Scikit-learn ensembles
                pred = ensemble_model.predict_proba(X_test)[:, 1]
            
            ensemble_preds.append(pred)
        
        # Combine predictions into a single feature matrix
        X_meta = np.column_stack(ensemble_preds)
        
        # Make predictions with meta-learner
        y_pred_proba = meta_learner.predict_proba(X_meta)[:, 1]
        
        return y_pred_proba


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    # Create imbalanced dataset (10% positive, 90% negative)
    X = np.random.randn(n_samples, n_features)
    y = np.zeros(n_samples)
    y[:100] = 1  # 10% positive samples
    
    # Split data
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    # Initialize ensemble models
    ensemble = EnsembleModels(output_dir='output/ensemble_models')
    
    # Create base models
    base_models = ensemble.create_base_models(input_shape=(n_features,))
    
    # Create voting ensemble
    voting_ensemble = ensemble.create_voting_ensemble(X_train, y_train)
    
    # Create stacking ensemble
    stacking_ensemble = ensemble.create_stacking_ensemble(X_train, y_train)
    
    # Create bagging ensemble
    bagging_ensemble = ensemble.create_bagging_ensemble(input_shape=(n_features,), n_models=5)
    
    # Train base models
    base_histories = ensemble.train_base_models(X_train, y_train, X_val, y_val, epochs=10)
    
    # Train bagging ensemble
    bagging_histories = ensemble.train_bagging_ensemble(X_train, y_train, X_val, y_val, epochs=10)
    
    # Save scikit-learn ensembles
    ensemble.save_sklearn_ensemble('voting_ensemble')
    ensemble.save_sklearn_ensemble('stacking_ensemble')
    
    # Evaluate ensembles
    ensemble_metrics = {}
    
    for name in ['voting_ensemble', 'stacking_ensemble']:
        metrics = ensemble.evaluate_ensemble(name, X_test, y_test)
        ensemble_metrics[name] = metrics
    
    # Evaluate bagging ensemble
    bagging_metrics = ensemble.evaluate_ensemble('bagging_ensemble', X_test, y_test)
    ensemble_metrics['bagging_ensemble'] = bagging_metrics
    
    # Compare ensembles
    best_ensembles = ensemble.compare_ensembles(ensemble_metrics, save_path='output/ensemble_models/ensemble_comparison.png')
    
    # Create super ensemble
    super_ensemble = ensemble.create_super_ensemble(['voting_ensemble', 'stacking_ensemble', 'bagging_ensemble'], X_train, y_train)
    
    # Evaluate super ensemble
    super_pred = ensemble.predict_with_super_ensemble(X_test)
    super_pred_binary = (super_pred > 0.5).astype(int)
    
    accuracy = np.mean(super_pred_binary == y_test)
    precision = precision_score(y_test, super_pred_binary)
    recall = recall_score(y_test, super_pred_binary)
    f1 = f1_score(y_test, super_pred_binary)
    auc = roc_auc_score(y_test, super_pred)
    
    print(f"Super Ensemble - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
