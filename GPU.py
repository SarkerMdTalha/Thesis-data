import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from cuml.ensemble import RandomForestClassifier as cuRF
from xgboost import XGBClassifier
import torch
import torch.nn as nn
import torch.optim as optim


warnings.filterwarnings('ignore')

class UserIDPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.feature_names = None
        self.trained = False
        
    def load_and_preprocess_data(self, file_path=None, df=None):
        """Load and preprocess the dataset"""
        if df is not None:
            self.df = df.copy()
        else:
            self.df = pd.read_csv(file_path)
        
        print("Dataset Overview:")
        print(f"Shape: {self.df.shape}")
        print(f"Users: {self.df['user_id'].nunique()}")
        print(f"Problems: {self.df['problem_id'].nunique()}")
        print("\nUser distribution:")
        print(self.df['user_id'].value_counts())
        
        # Feature engineering
        self.df = self._engineer_features(self.df)
        
        # Prepare features and target
        feature_columns = [col for col in self.df.columns if col not in ['user_id', 'problem_id']]
        self.feature_names = feature_columns
        
        X = self.df[feature_columns]
        y = self.df['user_id']
        
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        return X, y_encoded, y
    
    def _engineer_features(self, df):
        """Create additional features from existing ones"""
        df = df.copy()
        
        # Success rate features
        #df['success_rate'] = df['Accepted'] / df['submission_count']
        #df['error_rate'] = (df['Runtime Error'] + df['Time Limit Exceeded'] + df['Wrong Answer']) / df['submission_count']
        
        # Performance features
        df['cpu_time_range'] = df['cpu_time_max'] - df['cpu_time_min']
        df['cpu_time_stability'] = np.where(df['cpu_time_range'] == 0, 1, 0)  # Binary: consistent timing
        
        # Code complexity features
        df['code_density'] = df['code_size_mean'] / df['total_lines_mean']
        df['comment_ratio'] = df['total_comments_mean'] / df['total_lines_mean']
        df['variable_per_line'] = df['total_variables_mean'] / df['total_lines_mean']
        
        # Programming style features
        df['uses_loops'] = (df['loop_count_mean'] > 0).astype(int)
        df['uses_conditionals'] = (df['if_else_count_mean'] > 0).astype(int)
        #df['high_capitalization'] = (df['capitalized_variable_names_mean'] > df['total_variables_mean'] * 0.5).astype(int)
        
        # Handle NaN and infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        return df
    
    def analyze_features(self, X, y):
        """Analyze feature importance and correlations"""
        # Feature correlation with target
        df_analysis = X.copy()
        df_analysis['user_encoded'] = y
        
        # Calculate correlation with encoded user ID
        correlations = df_analysis.corr()['user_encoded'].abs().sort_values(ascending=False)
        
        print("\nTop 10 Features by Correlation with User ID:")
        print(correlations.head(11)[1:])  # Exclude self-correlation
        
        # Feature statistics by user
        df_with_user = X.copy()
        df_with_user['user_id'] = self.label_encoder.inverse_transform(y)
        
        print("\nFeature Means by User:")
        user_stats = df_with_user.groupby('user_id').agg({
            'cpu_time_min': 'mean',
            'memory_mean': 'mean',
            'code_size_mean': 'mean',
            'code_density': 'mean'
        }).round(3)
        print(user_stats)
        
        return correlations
    
    def train_models(self, X, y):
        """Train multiple models and compare performance"""
    
    # Check class distribution
        unique, counts = np.unique(y, return_counts=True)
        classes_with_single_sample = unique[counts == 1]

        if len(classes_with_single_sample) > 0:
            print(f"Warning: Found {len(classes_with_single_sample)} classes with only 1 sample.")
            print("These classes will be removed or stratification will be disabled.")
            
            # Option 1: Remove classes with single samples
            mask = ~np.isin(y, classes_with_single_sample)
            X_filtered = X[mask]
            y_filtered = y[mask]
            
            if len(np.unique(y_filtered)) < 2:
                print("Error: Not enough classes remaining after filtering.")
                raise ValueError("Dataset has insufficient samples per class for training.")
            
            print(f"Removed {len(classes_with_single_sample)} classes, proceeding with {len(np.unique(y_filtered))} classes.")
            X_train, X_test, y_train, y_test = train_test_split(
                X_filtered, y_filtered, test_size=0.3, random_state=42, stratify=y_filtered
            )
        else:
            # Original code - stratified split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Define models
        models = {
            'cuML Random Forest (GPU)': cuRF(n_estimators=100, random_state=42),
        'XGBoost (GPU)': XGBClassifier(tree_method='gpu_hist', use_label_encoder=False, eval_metric='mlogloss'),
        }


        results = {}

        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)

        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Use scaled data for models that need it
                if name in ['Logistic Regression', 'SVM', 'Neural Network']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    
                    # Cross-validation with error handling
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=min(5, len(np.unique(y_train))))
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    # Cross-validation with error handling
                    cv_scores = cross_val_score(model, X_train, y_train, cv=min(5, len(np.unique(y_train))))
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'predictions': y_pred
                }
                
                print(f"Test Accuracy: {accuracy:.4f}")
                print(f"CV Score: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
                
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue

        if not results:
            raise ValueError("No models could be trained successfully.")

        # Select best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['cv_mean'])
        self.model = results[best_model_name]['model']
        self.trained = True

        print(f"\n Best Model: {best_model_name}")
        print(f"Cross-validation Score: {results[best_model_name]['cv_mean']:.4f}")

        # Detailed analysis of best model
        self._analyze_best_model(X_test, y_test, results[best_model_name], best_model_name)

        return results, X_test, y_test
    
    def _analyze_best_model(self, X_test, y_test, best_result, model_name):
        print(f"\n{'='*40}")
        print(f"DETAILED ANALYSIS - {model_name}")
        print(f"{'='*40}")
        
        y_pred = best_result['predictions']
        
        # Get unique classes present in test set
        unique_classes_in_test = np.unique(y_test)
        
        # Get corresponding user names for classes present in test set
        user_names_in_test = self.label_encoder.inverse_transform(unique_classes_in_test)
        
        # Classification report
        print("\nClassification Report:")
        try:
            print(classification_report(y_test, y_pred, 
                                    labels=unique_classes_in_test,
                                    target_names=user_names_in_test))
        except Exception as e:
            print(f"Could not generate classification report: {e}")
            # Fallback - basic accuracy info
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Test Accuracy: {accuracy:.4f}")
        
        # Feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(feature_importance.head(10))
            
            # Plot feature importance
            try:
                plt.figure(figsize=(12, 8))
                top_features = feature_importance.head(15)
                plt.barh(range(len(top_features)), top_features['importance'])
                plt.yticks(range(len(top_features)), top_features['feature'])
                plt.xlabel('Feature Importance')
                plt.title(f'Feature Importance - {model_name}')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Could not generate feature importance plot: {e}")
        
        # Print confusion matrix info
        try:
            cm = confusion_matrix(y_test, y_pred, labels=unique_classes_in_test)
            print(f"\nConfusion Matrix Shape: {cm.shape}")
            print(f"Classes in test set: {len(unique_classes_in_test)}")
            print(f"User names: {list(user_names_in_test)}")
        except Exception as e:
            print(f"Could not generate confusion matrix info: {e}")
    
    def predict_user(self, features_dict):
        """Predict user ID for new data"""
        if not self.trained:
            raise ValueError("Model must be trained first!")
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([features_dict])
        
        # Apply same feature engineering
        input_df = self._engineer_features(input_df)
        
        # Select only the features used in training
        X_input = input_df[self.feature_names]
        
        # Scale if needed
        if isinstance(self.model, (LogisticRegression, SVC, MLPClassifier)):
            X_input = self.scaler.transform(X_input)
        
        # Make prediction
        prediction_encoded = self.model.predict(X_input)[0]
        prediction_proba = self.model.predict_proba(X_input)[0]
        
        # Get the classes that the model was actually trained on
        trained_classes = self.model.classes_
        
        # Decode prediction
        predicted_user = self.label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Get confidence scores for trained users only
        user_probabilities = {}
        
        # Map trained classes back to user names and get probabilities
        for i, class_encoded in enumerate(trained_classes):
            if i < len(prediction_proba):  # Safety check
                user_name = self.label_encoder.inverse_transform([class_encoded])[0]
                user_probabilities[user_name] = prediction_proba[i]
        
        # Sort by probability
        sorted_predictions = sorted(user_probabilities.items(), key=lambda x: x[1], reverse=True)
        
        print(f"Debug: Model trained on {len(trained_classes)} classes")
        print(f"Debug: Prediction probabilities shape: {prediction_proba.shape}")
        print(f"Debug: Available users: {list(user_probabilities.keys())}")
        
        return predicted_user, sorted_predictions

# Example usage and demonstration
def main():
    # Initialize predictor
    predictor = UserIDPredictor()
    
    # Load from CSV file
    while True:
        csv_file_path = "E:/trimmed_file4.csv"
        
        # Remove quotes if user copied path with quotes
        csv_file_path = csv_file_path.strip('"').strip("'")
        
        try:
            # Try to load the CSV file
            X, y_encoded, y_original = predictor.load_and_preprocess_data(file_path=csv_file_path)
            X = X.astype('float32')
            y_encoded = y_encoded.astype('int32')

            print(f"Successfully loaded data from: {csv_file_path}")
            break
        except FileNotFoundError:
            print(f"File not found: {csv_file_path}")
            print("Please check the file path and try again.")
            retry = input("Try again (y/n): ").strip().lower()
            if retry != 'y':
                print("Exiting...")
                return None
        except Exception as e:
            print(f"Error loading file: {str(e)}")
            print("Please check if the file format is correct.")
            retry = input("Try again? (y/n): ").strip().lower()
            if retry != 'y':
                print("Exiting...")
                return None
    
    # Analyze features
    correlations = predictor.analyze_features(X, y_encoded)
    
    # Train models
    
    results, X_test, y_test = predictor.train_models(X, y_encoded)
    X_test = X_test.astype('float32')
    
    print("\n" + "="*60)
    print("PREDICTION EXAMPLE")
    print("="*60)
    
    # Example prediction
    example_input = {
        'submission_count': 3,
        'cpu_time_min': 30,
        'cpu_time_max': 35,
        'memory_mean': 10000,
        'code_size_mean': 150,
        'total_lines_mean': 12,
        'avg_line_spacing_mean': 0.1,
        'total_comments_mean': 1,
        'if_else_count_mean': 2,
        'total_variables_mean': 4,
        'loop_count_mean': 1,
        'capitalized_variable_names_mean': 1,
        'percentage_for_loops_mean': 50.0,
        'Accepted': 2,
        'Runtime Error': 0,
        'Time Limit Exceeded': 0,
        'Wrong Answer': 1
    }
    
    predicted_user, probabilities = predictor.predict_user(example_input)
    
    print(f"\nPredicted User: {predicted_user}")
    print("\nConfidence scores for all users:")
    for user, prob in probabilities:
        print(f"{user}: {prob:.4f}")
    
    return predictor

if __name__ == "__main__":
    predictor = main()