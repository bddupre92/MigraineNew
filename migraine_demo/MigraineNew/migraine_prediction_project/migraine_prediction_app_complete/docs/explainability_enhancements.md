# Explainability and Interpretability Enhancements for Migraine Prediction

This document outlines advanced explainability and interpretability techniques to enhance the migraine prediction model, making it more transparent and understandable for both users and healthcare providers.

## Overview

While the current model achieves high performance, its complex nature (especially with the Mixture of Experts architecture) makes it difficult to understand why specific predictions are made. Implementing explainability techniques will help users understand their personal migraine triggers and enable healthcare providers to make more informed treatment decisions.

## Proposed Explainability Techniques

### 1. Counterfactual Explanations

Implement counterfactual explanations to show users what changes in their behavior or environment would alter the prediction outcome.

```python
import dice_ml
from dice_ml.utils import helpers

class CounterfactualExplainer:
    def __init__(self, model, feature_names, output_dir='output/counterfactuals'):
        self.model = model
        self.feature_names = feature_names
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_counterfactuals(self, X, y_pred, desired_outcome=0, num_cf=3):
        """Generate counterfactual explanations for predictions."""
        # Create a data frame for DiCE
        df = pd.DataFrame(X, columns=self.feature_names)
        
        # Add prediction column
        df['prediction'] = y_pred
        
        # Define data and model for DiCE
        d = dice_ml.Data(dataframe=df, 
                         continuous_features=self.feature_names,
                         outcome_name='prediction')
        
        # Create model wrapper
        m = dice_ml.Model(model=self.model, backend='tf2')
        
        # Create DiCE explainer
        exp = dice_ml.Dice(d, m)
        
        # Generate counterfactuals
        counterfactuals = []
        
        for i in range(len(X)):
            if y_pred[i] != desired_outcome:  # Only generate for predictions we want to change
                query_instance = df.iloc[i:i+1].drop(columns=['prediction'])
                
                # Generate counterfactual explanations
                cf = exp.generate_counterfactuals(query_instance, 
                                                 total_CFs=num_cf, 
                                                 desired_class=desired_outcome)
                
                # Store counterfactuals
                counterfactuals.append({
                    'instance_idx': i,
                    'original_features': X[i],
                    'original_prediction': y_pred[i],
                    'counterfactuals': cf.cf_examples_list[0].final_cfs_df.to_dict('records')
                })
                
                # Visualize counterfactuals
                if i < 5:  # Only visualize a few examples
                    cf.visualize_as_dataframe(show_only_changes=True)
                    plt.savefig(os.path.join(self.output_dir, f'counterfactual_{i}.png'), dpi=300, bbox_inches='tight')
                    plt.close()
        
        return counterfactuals
    
    def analyze_counterfactuals(self, counterfactuals):
        """Analyze counterfactuals to identify common patterns."""
        # Extract feature changes
        feature_changes = {feature: [] for feature in self.feature_names}
        
        for cf_data in counterfactuals:
            original = cf_data['original_features']
            
            for cf in cf_data['counterfactuals']:
                for i, feature in enumerate(self.feature_names):
                    if feature in cf:
                        change = cf[feature] - original[i]
                        feature_changes[feature].append(change)
        
        # Calculate average change per feature
        avg_changes = {feature: np.mean(changes) if changes else 0 
                      for feature, changes in feature_changes.items()}
        
        # Sort features by absolute average change
        sorted_features = sorted(avg_changes.items(), 
                                key=lambda x: abs(x[1]), 
                                reverse=True)
        
        # Plot top 10 features
        plt.figure(figsize=(12, 6))
        top_features = sorted_features[:10]
        
        plt.bar([f[0] for f in top_features], [f[1] for f in top_features])
        plt.title('Top 10 Features by Average Change in Counterfactuals')
        plt.xlabel('Feature')
        plt.ylabel('Average Change')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'top_features.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        return sorted_features
```

### 2. SHAP (SHapley Additive exPlanations)

Implement SHAP values to explain the contribution of each feature to the prediction.

```python
import shap

class SHAPExplainer:
    def __init__(self, model, output_dir='output/shap'):
        self.model = model
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def explain_predictions(self, X, feature_names=None):
        """Generate SHAP explanations for predictions."""
        # Create explainer
        explainer = shap.KernelExplainer(self.model.predict, shap.sample(X, 100))
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X)
        
        # Plot summary
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'shap_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot detailed explanations for a few examples
        for i in range(min(5, len(X))):
            plt.figure(figsize=(12, 4))
            shap.force_plot(explainer.expected_value, shap_values[i], X[i], 
                           feature_names=feature_names, matplotlib=True, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'shap_force_{i}.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        return shap_values, explainer
    
    def explain_temporal_data(self, X_temporal, feature_names=None):
        """Generate SHAP explanations for temporal data."""
        # Flatten temporal data
        X_flat = X_temporal.reshape(X_temporal.shape[0], -1)
        
        # Create explainer
        explainer = shap.KernelExplainer(
            lambda x: self.model.predict(x.reshape(-1, X_temporal.shape[1], X_temporal.shape[2])), 
            shap.sample(X_flat, 100)
        )
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_flat)
        
        # Reshape SHAP values to match temporal structure
        shap_values_temporal = shap_values.reshape(X_temporal.shape)
        
        # Plot heatmap for temporal features
        for i in range(min(5, len(X_temporal))):
            plt.figure(figsize=(15, 5))
            
            # Create heatmap
            sns.heatmap(shap_values_temporal[i].T, 
                       cmap='coolwarm', 
                       center=0,
                       yticklabels=feature_names if feature_names else 'auto',
                       xticklabels=range(X_temporal.shape[1]))
            
            plt.title(f'SHAP Values for Instance {i}')
            plt.xlabel('Time Step')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'shap_temporal_{i}.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        return shap_values_temporal, explainer
```

### 3. Attention Visualization

Implement attention mechanisms and visualize which features and time periods contribute most to predictions.

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

class AttentionVisualizer:
    def __init__(self, output_dir='output/attention'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def create_attention_model(self, input_shape, n_classes=1):
        """Create a model with attention mechanism for visualization."""
        # Input layer
        inputs = layers.Input(shape=input_shape)
        
        # Bidirectional LSTM layer
        lstm_out = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inputs)
        
        # Attention mechanism
        attention = layers.Dense(1, activation='tanh')(lstm_out)
        attention = layers.Flatten()(attention)
        attention_weights = layers.Activation('softmax', name='attention_weights')(attention)
        
        # Apply attention weights
        context = layers.Dot(axes=1)([lstm_out, layers.Reshape((input_shape[0], 1))(attention_weights)])
        context = layers.Flatten()(context)
        
        # Output layer
        if n_classes == 1:
            outputs = layers.Dense(1, activation='sigmoid')(context)
        else:
            outputs = layers.Dense(n_classes, activation='softmax')(context)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Create attention model (for extracting attention weights)
        attention_model = Model(inputs=model.input, 
                               outputs=[model.output, model.get_layer('attention_weights').output])
        
        return model, attention_model
    
    def visualize_attention(self, attention_model, X, y_true, feature_names=None, time_labels=None):
        """Visualize attention weights for temporal data."""
        # Get predictions and attention weights
        predictions, attention_weights = attention_model.predict(X)
        
        # Plot attention weights for a few examples
        for i in range(min(10, len(X))):
            plt.figure(figsize=(12, 6))
            
            # Plot time series data
            ax1 = plt.subplot(2, 1, 1)
            for j in range(X.shape[2]):
                plt.plot(X[i, :, j], label=feature_names[j] if feature_names else f'Feature {j}')
            
            plt.title(f'Instance {i} - True: {y_true[i]}, Pred: {predictions[i, 0]:.2f}')
            plt.legend()
            
            # Plot attention weights
            ax2 = plt.subplot(2, 1, 2, sharex=ax1)
            plt.bar(range(len(attention_weights[i])), attention_weights[i], alpha=0.7)
            plt.title('Attention Weights')
            
            if time_labels:
                plt.xticks(range(len(time_labels)), time_labels, rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'attention_{i}.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot average attention weights
        plt.figure(figsize=(10, 4))
        avg_attention = np.mean(attention_weights, axis=0)
        plt.bar(range(len(avg_attention)), avg_attention, alpha=0.7)
        plt.title('Average Attention Weights')
        
        if time_labels:
            plt.xticks(range(len(time_labels)), time_labels, rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'avg_attention.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        return predictions, attention_weights
```

### 4. Rule Extraction

Extract interpretable rules from the complex model to provide simple explanations.

```python
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus
from IPython.display import Image

class RuleExtractor:
    def __init__(self, output_dir='output/rules'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def extract_rules(self, X, y_pred, max_depth=3, feature_names=None):
        """Extract decision rules from complex model predictions."""
        # Train a decision tree to mimic the complex model
        tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        tree.fit(X, y_pred)
        
        # Evaluate tree performance
        tree_pred = tree.predict(X)
        accuracy = np.mean(tree_pred == y_pred)
        print(f"Decision tree mimics complex model with {accuracy:.2%} accuracy")
        
        # Visualize decision tree
        dot_data = export_graphviz(
            tree,
            out_file=None,
            feature_names=feature_names,
            class_names=['No Migraine', 'Migraine'],
            filled=True,
            rounded=True,
            special_characters=True
        )
        
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_png(os.path.join(self.output_dir, 'decision_tree.png'))
        
        # Extract rules
        rules = self._extract_rules_from_tree(tree, feature_names)
        
        # Save rules to file
        with open(os.path.join(self.output_dir, 'rules.txt'), 'w') as f:
            for rule in rules:
                f.write(f"{rule}\n")
        
        return rules, tree
    
    def _extract_rules_from_tree(self, tree, feature_names=None):
        """Extract rules from decision tree."""
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if feature_names else f"feature {i}"
            for i in tree_.feature
        ]
        
        rules = []
        
        def recurse(node, depth, rule):
            if tree_.feature[node] != -2:  # Not a leaf node
                name = feature_name[node]
                threshold = tree_.threshold[node]
                
                # Left branch (<=)
                recurse(tree_.children_left[node], depth + 1, 
                       rule + [f"{name} <= {threshold:.2f}"])
                
                # Right branch (>)
                recurse(tree_.children_right[node], depth + 1,
                       rule + [f"{name} > {threshold:.2f}"])
            else:  # Leaf node
                class_prob = tree_.value[node][0] / np.sum(tree_.value[node][0])
                pred_class = np.argmax(class_prob)
                prob = class_prob[pred_class]
                
                if pred_class == 1 and prob > 0.5:  # Only include rules predicting migraines
                    rules.append(f"IF {' AND '.join(rule)} THEN Migraine Risk: {prob:.2%}")
        
        recurse(0, 1, [])
        return rules
```

### 5. Interactive Explanation Dashboard

Create an interactive dashboard for users to explore model explanations.

```python
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

class ExplanationDashboard:
    def __init__(self, model, feature_names, time_labels=None):
        self.model = model
        self.feature_names = feature_names
        self.time_labels = time_labels
        
        # Initialize explainers
        self.counterfactual_explainer = CounterfactualExplainer(model, feature_names)
        self.shap_explainer = SHAPExplainer(model)
        self.attention_visualizer = AttentionVisualizer()
        self.rule_extractor = RuleExtractor()
        
    def run_dashboard(self):
        """Run the explanation dashboard."""
        st.title("Migraine Prediction Explanation Dashboard")
        
        # Sidebar for navigation
        explanation_type = st.sidebar.selectbox(
            "Select Explanation Type",
            ["Overview", "Counterfactuals", "Feature Importance", "Attention Visualization", "Decision Rules"]
        )
        
        # Load sample data
        X_sample, y_sample = self._load_sample_data()
        
        # Select instance to explain
        instance_idx = st.sidebar.slider("Select Instance", 0, len(X_sample)-1, 0)
        
        # Get prediction for selected instance
        X_instance = X_sample[instance_idx:instance_idx+1]
        y_pred = self.model.predict(X_instance)[0]
        
        # Display prediction
        st.write(f"### Instance {instance_idx}")
        st.write(f"**Prediction:** {'Migraine Risk' if y_pred > 0.5 else 'No Migraine Risk'} ({y_pred[0]:.2%})")
        
        # Display explanation based on selected type
        if explanation_type == "Overview":
            self._show_overview(X_instance[0], y_pred)
        
        elif explanation_type == "Counterfactuals":
            self._show_counterfactuals(X_instance[0], y_pred)
        
        elif explanation_type == "Feature Importance":
            self._show_feature_importance(X_instance[0])
        
        elif explanation_type == "Attention Visualization":
            self._show_attention(X_instance[0], y_pred)
        
        elif explanation_type == "Decision Rules":
            self._show_decision_rules()
    
    def _show_overview(self, X_instance, y_pred):
        """Show overview of all explanation types."""
        st.write("## Explanation Overview")
        
        # Show feature values
        st.write("### Feature Values")
        feature_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Value': X_instance
        })
        st.dataframe(feature_df)
        
        # Show top contributing features
        st.write("### Top Contributing Features")
        explainer = shap.KernelExplainer(self.model.predict, shap.sample(X_sample, 100))
        shap_values = explainer.shap_values(X_instance.reshape(1, -1))
        
        fig = go.Figure()
        sorted_idx = np.argsort(np.abs(shap_values[0]))[-10:]  # Top 10 features
        
        fig.add_trace(go.Bar(
            y=[self.feature_names[i] for i in sorted_idx],
            x=shap_values[0][sorted_idx],
            orientation='h'
        ))
        
        fig.update_layout(
            title="Top 10 Contributing Features",
            xaxis_title="SHAP Value (Impact on Prediction)",
            height=400
        )
        
        st.plotly_chart(fig)
        
        # Show counterfactual suggestion
        if y_pred > 0.5:  # If migraine predicted
            st.write("### Counterfactual Suggestion")
            st.write("To reduce migraine risk, consider these changes:")
            
            # Generate counterfactual
            cf = self.counterfactual_explainer.generate_counterfactuals(
                X_instance.reshape(1, -1), np.array([y_pred]), desired_outcome=0, num_cf=1
            )
            
            if cf and cf[0]['counterfactuals']:
                # Show top changes
                original = X_instance
                counterfactual = np.array(list(cf[0]['counterfactuals'][0].values()))
                
                changes = counterfactual - original
                sorted_idx = np.argsort(np.abs(changes))[-5:]  # Top 5 changes
                
                change_df = pd.DataFrame({
                    'Feature': [self.feature_names[i] for i in sorted_idx],
                    'Current Value': [original[i] for i in sorted_idx],
                    'Suggested Value': [counterfactual[i] for i in sorted_idx],
                    'Change': [changes[i] for i in sorted_idx]
                })
                
                st.dataframe(change_df)
    
    def _show_counterfactuals(self, X_instance, y_pred):
        """Show counterfactual explanations."""
        st.write("## Counterfactual Explanations")
        st.write("Counterfactuals show what changes would alter the prediction outcome.")
        
        # Generate counterfactuals
        desired_outcome = 0 if y_pred > 0.5 else 1
        outcome_label = "No Migraine Risk" if desired_outcome == 0 else "Migraine Risk"
        
        st.write(f"### Changes needed for prediction to become: {outcome_label}")
        
        cf = self.counterfactual_explainer.generate_counterfactuals(
            X_instance.reshape(1, -1), np.array([y_pred]), desired_outcome=desired_outcome, num_cf=3
        )
        
        if cf and cf[0]['counterfactuals']:
            # Show counterfactuals
            for i, counterfactual in enumerate(cf[0]['counterfactuals']):
                st.write(f"#### Counterfactual {i+1}")
                
                # Calculate changes
                original = X_instance
                cf_values = np.array(list(counterfactual.values()))
                changes = cf_values - original
                
                # Show only features that changed
                changed_idx = np.where(np.abs(changes) > 0.01)[0]
                
                if len(changed_idx) > 0:
                    change_df = pd.DataFrame({
                        'Feature': [self.feature_names[i] for i in changed_idx],
                        'Current Value': [original[i] for i in changed_idx],
                        'Counterfactual Value': [cf_values[i] for i in changed_idx],
                        'Change': [changes[i] for i in changed_idx]
                    })
                    
                    st.dataframe(change_df)
                    
                    # Visualize changes
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        y=[self.feature_names[i] for i in changed_idx],
                        x=[changes[i] for i in changed_idx],
                        orientation='h'
                    ))
                    
                    fig.update_layout(
                        title="Feature Changes",
                        xaxis_title="Change Amount",
                        height=400
                    )
                    
                    st.plotly_chart(fig)
                else:
                    st.write("No significant changes found.")
        else:
            st.write("Could not generate counterfactuals.")
    
    def _show_feature_importance(self, X_instance):
        """Show feature importance using SHAP values."""
        st.write("## Feature Importance")
        
        # Calculate SHAP values
        explainer = shap.KernelExplainer(self.model.predict, shap.sample(X_sample, 100))
        shap_values = explainer.shap_values(X_instance.reshape(1, -1))
        
        # Show waterfall plot
        st.write("### Impact of Each Feature")
        
        # Create waterfall chart
        features = self.feature_names
        values = shap_values[0]
        
        # Sort by absolute value
        sorted_idx = np.argsort(np.abs(values))
        
        fig = go.Figure(go.Waterfall(
            name="SHAP",
            orientation="h",
            measure=["relative"] * len(values),
            y=[features[i] for i in sorted_idx],
            x=[values[i] for i in sorted_idx],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(
            title="Feature Contributions to Prediction",
            showlegend=False,
            height=600
        )
        
        st.plotly_chart(fig)
        
        # Show feature value context
        st.write("### Feature Values in Context")
        
        # Create context plot for top features
        top_features = sorted_idx[-10:]  # Top 10 features
        
        for idx in top_features:
            feature_name = features[idx]
            feature_value = X_instance[idx]
            feature_impact = values[idx]
            
            st.write(f"#### {feature_name}")
            st.write(f"Value: {feature_value:.4f}, Impact: {feature_impact:.4f}")
            
            # Create histogram of feature distribution
            fig = px.histogram(X_sample[:, idx], nbins=20)
            fig.add_vline(x=feature_value, line_width=3, line_dash="dash", line_color="red")
            
            fig.update_layout(
                title=f"Distribution of {feature_name}",
                xaxis_title=feature_name,
                yaxis_title="Count",
                height=300
            )
            
            st.plotly_chart(fig)
    
    def _show_attention(self, X_instance, y_pred):
        """Show attention visualization for temporal data."""
        st.write("## Attention Visualization")
        
        # Check if we have temporal data
        if len(X_instance.shape) < 2:
            st.write("Attention visualization requires temporal data.")
            return
        
        # Create attention model if not already created
        if not hasattr(self, 'attention_model'):
            base_model, attention_model = self.attention_visualizer.create_attention_model(
                X_instance.shape, n_classes=1
            )
            
            # Train model on sample data (in real implementation, this would be pre-trained)
            base_model.compile(optimizer='adam', loss='binary_crossentropy')
            base_model.fit(X_sample, y_sample, epochs=10, verbose=0)
            
            self.attention_model = attention_model
        
        # Get attention weights
        _, attention_weights = self.attention_model.predict(X_instance.reshape(1, *X_instance.shape))
        
        # Visualize attention weights
        st.write("### Attention Weights")
        st.write("Highlights which time periods are most important for the prediction.")
        
        fig = go.Figure()
        
        # Add time series data
        for i in range(X_instance.shape[1]):
            feature_name = self.feature_names[i] if i < len(self.feature_names) else f"Feature {i}"
            
            fig.add_trace(go.Scatter(
                x=self.time_labels if self.time_labels else list(range(X_instance.shape[0])),
                y=X_instance[:, i],
                mode='lines',
                name=feature_name
            ))
        
        # Add attention weights
        fig.add_trace(go.Bar(
            x=self.time_labels if self.time_labels else list(range(X_instance.shape[0])),
            y=attention_weights[0],
            name="Attention",
            opacity=0.5,
            marker_color='rgba(255, 0, 0, 0.5)'
        ))
        
        fig.update_layout(
            title="Attention Weights Over Time",
            xaxis_title="Time",
            yaxis_title="Value / Attention Weight",
            height=500
        )
        
        st.plotly_chart(fig)
    
    def _show_decision_rules(self):
        """Show extracted decision rules."""
        st.write("## Decision Rules")
        st.write("Simple rules extracted from the complex model.")
        
        # Extract rules if not already done
        if not hasattr(self, 'rules'):
            self.rules, _ = self.rule_extractor.extract_rules(
                X_sample, self.model.predict(X_sample) > 0.5, 
                max_depth=3, feature_names=self.feature_names
            )
        
        # Display rules
        for i, rule in enumerate(self.rules):
            st.write(f"### Rule {i+1}")
            st.write(rule)
        
        # Show decision tree
        st.write("### Decision Tree Visualization")
        st.image(os.path.join(self.rule_extractor.output_dir, 'decision_tree.png'))
    
    def _load_sample_data(self):
        """Load sample data for explanations."""
        # In a real implementation, this would load actual data
        # For now, we'll just return a placeholder
        if not hasattr(self, 'X_sample'):
            self.X_sample = X_sample
            self.y_sample = y_sample
        
        return self.X_sample, self.y_sample
```

## Integration with Existing Dashboard

These explainability techniques can be integrated with the existing dashboard:

```python
class ExplainabilityEnhancer:
    def __init__(self, model, feature_names, output_dir='output/explainability'):
        self.model = model
        self.feature_names = feature_names
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize explainers
        self.counterfactual_explainer = CounterfactualExplainer(model, feature_names, 
                                                              output_dir=os.path.join(output_dir, 'counterfactuals'))
        self.shap_explainer = SHAPExplainer(model, 
                                           output_dir=os.path.join(output_dir, 'shap'))
        self.attention_visualizer = AttentionVisualizer(
                                           output_dir=os.path.join(output_dir, 'attention'))
        self.rule_extractor = RuleExtractor(
                                           output_dir=os.path.join(output_dir, 'rules'))
        
    def generate_explanations(self, X, y_true, y_pred):
        """Generate all types of explanations."""
        print("Generating explanations...")
        
        # Generate counterfactuals
        print("\nGenerating counterfactual explanations...")
        counterfactuals = self.counterfactual_explainer.generate_counterfactuals(X, y_pred)
        counterfactual_analysis = self.counterfactual_explainer.analyze_counterfactuals(counterfactuals)
        
        # Generate SHAP explanations
        print("\nGenerating SHAP explanations...")
        shap_values, shap_explainer = self.shap_explainer.explain_predictions(X, self.feature_names)
        
        # Generate attention visualizations
        print("\nGenerating attention visualizations...")
        # Create and train attention model
        if len(X.shape) == 2:  # Non-temporal data
            # Reshape to temporal format for attention (assume each feature is a time step)
            X_temporal = X.reshape(X.shape[0], X.shape[1], 1)
            time_labels = self.feature_names
        else:  # Already temporal
            X_temporal = X
            time_labels = [f"Time {i}" for i in range(X.shape[1])]
        
        attention_model, attention_extractor = self.attention_visualizer.create_attention_model(X_temporal.shape[1:])
        
        # Train attention model
        attention_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        attention_model.fit(X_temporal, y_true, epochs=20, batch_size=32, verbose=0)
        
        # Visualize attention
        predictions, attention_weights = self.attention_visualizer.visualize_attention(
            attention_extractor, X_temporal, y_true, 
            feature_names=self.feature_names, time_labels=time_labels
        )
        
        # Extract decision rules
        print("\nExtracting decision rules...")
        rules, rule_tree = self.rule_extractor.extract_rules(X, y_pred > 0.5, feature_names=self.feature_names)
        
        # Save all explanations
        explanation_data = {
            'counterfactuals': counterfactuals,
            'counterfactual_analysis': counterfactual_analysis,
            'shap_values': shap_values,
            'attention_weights': attention_weights,
            'rules': rules
        }
        
        # Save models
        attention_model.save(os.path.join(self.output_dir, 'attention_model.keras'))
        
        print("\nExplanations generated successfully.")
        return explanation_data
    
    def create_explanation_dashboard(self, X_sample, y_sample):
        """Create interactive explanation dashboard."""
        dashboard = ExplanationDashboard(self.model, self.feature_names)
        dashboard.X_sample = X_sample
        dashboard.y_sample = y_sample
        
        return dashboard
```

## Expected Benefits

1. **Improved User Understanding**: Users can understand why the model predicts migraines and what factors contribute most.
2. **Actionable Insights**: Counterfactual explanations provide actionable steps to reduce migraine risk.
3. **Increased Trust**: Transparent explanations increase user trust in the model's predictions.
4. **Clinical Utility**: Healthcare providers can use explanations to better understand patient-specific triggers.
5. **Regulatory Compliance**: Explainable AI is increasingly important for regulatory compliance in healthcare applications.

## Performance Impact

These explainability techniques have minimal impact on the model's predictive performance, as they are post-hoc explanations that don't modify the underlying model. However, they significantly enhance the model's utility and trustworthiness.

## Implementation Timeline

1. Counterfactual Explanations: 1 week
2. SHAP Implementation: 1 week
3. Attention Visualization: 1 week
4. Rule Extraction: 1 week
5. Interactive Dashboard: 2 weeks

Total estimated time: 6 weeks
