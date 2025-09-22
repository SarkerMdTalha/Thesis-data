import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ImprovedUserIDPredictor:
    def __init__(self):
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.model = None
        self.feature_names = None
        self.trained = False
        self.min_samples_per_user = 3  # Minimum samples required per user
        
    def load_and_preprocess_data(self, file_path=None, df=None):
        """Load and preprocess the dataset with improved filtering"""
        if df is not None:
            self.df = df.copy()
        else:
            self.df = pd.read_csv(file_path)
        
        print("Dataset Overview:")
        print(f"Original Shape: {self.df.shape}")
        print(f"Original Users: {self.df['user_id'].nunique()}")
        print(f"Problems: {self.df['problem_id'].nunique()}")
        
        # Filter users with sufficient samples
        user_counts = self.df['user_id'].value_counts()
        users_with_enough_samples = user_counts[user_counts >= self.min_samples_per_user].index
        
        print(f"\nFiltering users with at least {self.min_samples_per_user} samples:")
        print(f"Users before filtering: {len(user_counts)}")
        print(f"Users after filtering: {len(users_with_enough_samples)}")
        
        self.df = self.df[self.df['user_id'].isin(users_with_enough_samples)]
        
        print(f"Filtered Shape: {self.df.shape}")
        print(f"Remaining Users: {self.df['user_id'].nunique()}")
        
        # Check if we have enough data
        if self.df['user_id'].nunique() < 2:
            raise ValueError("Not enough users with sufficient samples for training")
        
        print("\nUser distribution after filtering:")
        print(self.df['user_id'].value_counts().head(10))
        
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
        """Create additional features with better handling of edge cases"""
        df = df.copy()
        
        # Basic ratios and rates (with safe division)
        df['success_rate'] = np.where(df['submission_count'] > 0, 
                                    df['Accepted'] / df['submission_count'], 0)
        df['error_rate'] = np.where(df['submission_count'] > 0,
                                  (df['Runtime Error'] + df['Time Limit Exceeded'] + df['Wrong Answer']) / df['submission_count'], 0)
        df['first_attempt_success'] = (df['submission_count'] == 1) & (df['Accepted'] == 1)
        
        # Performance and timing features
        df['cpu_time_range'] = df['cpu_time_max'] - df['cpu_time_min']
        df['cpu_time_consistency'] = np.where(df['cpu_time_max'] > 0, 
                                            1 - (df['cpu_time_range'] / df['cpu_time_max']), 1)
        df['cpu_time_avg'] = (df['cpu_time_min'] + df['cpu_time_max']) / 2
        
        # Code style features (with safe division)
        df['code_density'] = np.where(df['total_lines_mean'] > 0, 
                                    df['code_size_mean'] / df['total_lines_mean'], 0)
        df['comment_ratio'] = np.where(df['total_lines_mean'] > 0,
                                     df['total_comments_mean'] / df['total_lines_mean'], 0)
        df['variable_density'] = np.where(df['total_lines_mean'] > 0,
                                        df['total_variables_mean'] / df['total_lines_mean'], 0)
        
        # Programming patterns
        df['uses_loops'] = (df['loop_count_mean'] > 0).astype(int)
        df['uses_conditionals'] = (df['if_else_count_mean'] > 0).astype(int)
        df['uses_comments'] = (df['total_comments_mean'] > 0).astype(int)
        df['high_variable_naming'] = (df['capitalized_variable_names_mean'] > 
                                    df['total_variables_mean'] * 0.5).astype(int)
        
        # Advanced features
        df['complexity_score'] = (df['if_else_count_mean'] + df['loop_count_mean'] + 
                                df['total_variables_mean']) / np.maximum(df['total_lines_mean'], 1)
        
        df['coding_efficiency'] = np.where(df['code_size_mean'] > 0,
                                         df['total_lines_mean'] / df['code_size_mean'], 0)
        
        # Memory efficiency
        df['memory_per_line'] = np.where(df['total_lines_mean'] > 0,
                                       df['memory_mean'] / df['total_lines_mean'], 0)
        
        # Submission patterns
        df['multiple_attempts'] = (df['submission_count'] > 1).astype(int)
        df['perfect_submission'] = ((df['submission_count'] == df['Accepted']) & 
                                  (df['submission_count'] > 0)).astype(int)
        
        # Handle NaN and infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with more intelligent defaults
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['user_id', 'problem_id']:
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def select_features(self, X, y, method='mutual_info', k=20):
        """Feature selection to reduce noise and improve performance"""
        print(f"\nPerforming feature selection using {method}...")
        print(f"Original features: {X.shape[1]}")
        
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
        else:
            selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        
        X_selected = selector.fit_transform(X, y)
        selected_features = [self.feature_names[i] for i in selector.get_support(indices=True)]
        
        print(f"Selected features: {len(selected_features)}")
        print("Top selected features:", selected_features[:10])
        
        self.feature_selector = selector
        self.selected_feature_names = selected_features
        
        return X_selected, selected_features
    
    def analyze_data_distribution(self, X, y):
        """Analyze data distribution and identify potential issues"""
        print("\n" + "="*50)
        print("DATA DISTRIBUTION ANALYSIS")
        print("="*50)
        
        # Class distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"Number of classes: {len(unique)}")
        print(f"Samples per class - Min: {counts.min()}, Max: {counts.max()}, Mean: {counts.mean():.2f}")
        
        # Feature statistics
        print(f"\nFeature statistics:")
        print(f"Number of features: {X.shape[1]}")
        print(f"Features with zero variance: {(X.var(axis=0) == 0).sum()}")
        print(f"Features with high correlation (>0.9): {self._count_high_correlations(X)}")
        
        # Check for class imbalance
        class_distribution = np.bincount(y) / len(y)
        imbalance_ratio = class_distribution.max() / class_distribution.min()
        print(f"Class imbalance ratio: {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 10:
            print("WARNING: High class imbalance detected. Consider using balanced models.")
        
        return imbalance_ratio
    
    def _count_high_correlations(self, X):
        """Count features with high correlation"""
        try:
            corr_matrix = np.corrcoef(X.T)
            # Set diagonal to 0 to ignore self-correlation
            np.fill_diagonal(corr_matrix, 0)
            high_corr_count = (np.abs(corr_matrix) > 0.9).sum() // 2  # Divide by 2 to avoid double counting
            return high_corr_count
        except:
            return 0
    
    def train_models(self, X, y):
        """Train multiple models with hyperparameter tuning"""
        
        # Analyze data distribution
        imbalance_ratio = self.analyze_data_distribution(X, y)
        
        # Feature selection
        X_selected, selected_features = self.select_features(X, y, method='mutual_info', k=15)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.25, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models with hyperparameter tuning
        models = {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },
                'use_scaled': False
            },
            'Extra Trees': {
                'model': ExtraTreesClassifier(random_state=42, class_weight='balanced'),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20],
                    'min_samples_split': [2, 5]
                },
                'use_scaled': False
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 150],
                    'learning_rate': [0.1, 0.05],
                    'max_depth': [5, 7]
                },
                'use_scaled': False
            },
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, class_weight='balanced', max_iter=2000),
                'params': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                },
                'use_scaled': True
            },
            'SVM': {
                'model': SVC(random_state=42, class_weight='balanced', probability=True),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                },
                'use_scaled': True
            }
        }
        
        results = {}
        
        print("\n" + "="*60)
        print("MODEL TRAINING WITH HYPERPARAMETER TUNING")
        print("="*60)
        
        for name, config in models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Prepare data
                if config['use_scaled']:
                    X_train_model = X_train_scaled
                    X_test_model = X_test_scaled
                else:
                    X_train_model = X_train
                    X_test_model = X_test
                
                # Grid search with cross-validation
                cv_folds = min(5, np.bincount(y_train).min())  # Ensure each fold has all classes
                grid_search = GridSearchCV(
                    config['model'], 
                    config['params'], 
                    cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_train_model, y_train)
                
                # Best model predictions
                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(X_test_model)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                results[name] = {
                    'model': best_model,
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'best_params': grid_search.best_params_,
                    'cv_score': grid_search.best_score_,
                    'predictions': y_pred
                }
                
                print(f"Best parameters: {grid_search.best_params_}")
                print(f"CV Score: {grid_search.best_score_:.4f}")
                print(f"Test Accuracy: {accuracy:.4f}")
                print(f"F1 Score: {f1:.4f}")
                
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
        
        if not results:
            raise ValueError("No models could be trained successfully.")
        
        # Select best model based on CV score
        best_model_name = max(results.keys(), key=lambda k: results[k]['cv_score'])
        self.model = results[best_model_name]['model']
        self.trained = True
        
        print(f"\nBest Model: {best_model_name}")
        print(f"CV Score: {results[best_model_name]['cv_score']:.4f}")
        print(f"Test Accuracy: {results[best_model_name]['accuracy']:.4f}")
        
        # Detailed analysis
        self._analyze_best_model(X_test, y_test, results[best_model_name], best_model_name)
        
        return results, X_test, y_test
    
    def _analyze_best_model(self, X_test, y_test, best_result, model_name):
        """Enhanced analysis of the best model"""
        print(f"\n{'='*50}")
        print(f"DETAILED ANALYSIS - {model_name}")
        print(f"{'='*50}")
        
        y_pred = best_result['predictions']
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Weighted F1 Score: {f1:.4f}")
        
        # Per-class analysis
        unique_classes = np.unique(y_test)
        user_names = self.label_encoder.inverse_transform(unique_classes)
        
        print(f"\nClasses in test set: {len(unique_classes)}")
        print("Sample of user names:", user_names[:10] if len(user_names) > 10 else user_names)
        
        # Classification report with error handling
        try:
            report = classification_report(y_test, y_pred, 
                                         labels=unique_classes,
                                         target_names=user_names,
                                         output_dict=True,
                                         zero_division=0)
            
            # Show only macro and weighted averages for clarity
            print(f"\nMacro Average F1: {report['macro avg']['f1-score']:.4f}")
            print(f"Weighted Average F1: {report['weighted avg']['f1-score']:.4f}")
            
        except Exception as e:
            print(f"Could not generate detailed classification report: {e}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.selected_feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 10 Most Important Features:")
            for idx, row in feature_importance.head(10).iterrows():
                print(f"{row['feature']}: {row['importance']:.4f}")
    
    def predict_user(self, features_dict):
        """Predict user ID for new data"""
        if not self.trained:
            raise ValueError("Model must be trained first!")
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([features_dict])
        
        # Apply same feature engineering
        input_df = self._engineer_features(input_df)
        
        # Select only the training features
        X_input = input_df[self.feature_names]
        
        # Apply feature selection
        X_input_selected = self.feature_selector.transform(X_input)
        
        # Scale if needed
        if isinstance(self.model, (LogisticRegression, SVC, MLPClassifier)):
            X_input_selected = self.scaler.transform(X_input_selected)
        
        # Make prediction
        prediction_encoded = self.model.predict(X_input_selected)[0]
        prediction_proba = self.model.predict_proba(X_input_selected)[0]
        
        # Get prediction details
        predicted_user = self.label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Get confidence scores
        trained_classes = self.model.classes_
        user_probabilities = {}
        
        for i, class_encoded in enumerate(trained_classes):
            if i < len(prediction_proba):
                user_name = self.label_encoder.inverse_transform([class_encoded])[0]
                user_probabilities[user_name] = prediction_proba[i]
        
        sorted_predictions = sorted(user_probabilities.items(), key=lambda x: x[1], reverse=True)
        
        return predicted_user, sorted_predictions

# Enhanced main function with better error handling
def main():
    predictor = ImprovedUserIDPredictor()
    
    # Configuration
    csv_file_path = "E:/thesis/trimmed_file6.csv"  # Update this path
    
    try:
        # Load and preprocess data
        X, y_encoded, y_original = predictor.load_and_preprocess_data(file_path=csv_file_path)
        
        # Train models
        results, X_test, y_test = predictor.train_models(X, y_encoded)
        
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        
        for name, result in results.items():
            print(f"{name}:")
            print(f"  CV Score: {result['cv_score']:.4f}")
            print(f"  Test Accuracy: {result['accuracy']:.4f}")
            print(f"  F1 Score: {result['f1_score']:.4f}")
            print()
        
        return predictor
        
    except FileNotFoundError:
        print(f"File not found: {csv_file_path}")
        print("Please update the file path in the code.")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    predictor = main()