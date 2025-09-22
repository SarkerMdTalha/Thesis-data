import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class DatasetAnalyzer:
    def __init__(self):
        self.df = None
        self.issues = []
        self.recommendations = []
    
    def load_and_analyze(self, file_path):
        """Load dataset and perform comprehensive analysis"""
        print("="*60)
        print("DATASET DIAGNOSIS")
        print("="*60)
        
        self.df = pd.read_csv(file_path)
        
        # Basic info
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        self._analyze_target_distribution()
        self._analyze_feature_quality()
        self._analyze_user_patterns()
        self._check_data_leakage()
        self._test_baseline_separability()
        
        return self._generate_recommendations()
    
    def _analyze_target_distribution(self):
        """Analyze user distribution"""
        print(f"\n1. TARGET DISTRIBUTION ANALYSIS")
        print("-" * 40)
        
        user_counts = self.df['user_id'].value_counts()
        
        print(f"Total users: {len(user_counts)}")
        print(f"Total samples: {len(self.df)}")
        print(f"Average samples per user: {len(self.df) / len(user_counts):.2f}")
        
        # Distribution breakdown
        single_sample = (user_counts == 1).sum()
        few_samples = ((user_counts >= 2) & (user_counts <= 5)).sum()
        many_samples = (user_counts > 5).sum()
        
        print(f"\nUser distribution:")
        print(f"  Users with 1 sample: {single_sample} ({single_sample/len(user_counts)*100:.1f}%)")
        print(f"  Users with 2-5 samples: {few_samples} ({few_samples/len(user_counts)*100:.1f}%)")
        print(f"  Users with 6+ samples: {many_samples} ({many_samples/len(user_counts)*100:.1f}%)")
        
        # Issue detection
        if single_sample > len(user_counts) * 0.5:
            self.issues.append("CRITICAL: Over 50% of users have only 1 sample")
            self.recommendations.append("Remove users with single samples or collect more data")
        
        if len(user_counts) > 100 and many_samples < 10:
            self.issues.append("HIGH: Too many users with too few samples each")
            self.recommendations.append("Focus on users with more samples (6+ submissions)")
    
    def _analyze_feature_quality(self):
        """Analyze feature quality and usefulness"""
        print(f"\n2. FEATURE QUALITY ANALYSIS")
        print("-" * 40)
        
        # Exclude non-feature columns
        feature_cols = [col for col in self.df.columns if col not in ['user_id', 'problem_id']]
        X = self.df[feature_cols]
        
        print(f"Number of features: {len(feature_cols)}")
        
        # Check for problematic features
        zero_variance = (X.var() == 0).sum()
        high_missing = (X.isnull().sum() > len(X) * 0.1).sum()
        
        print(f"Features with zero variance: {zero_variance}")
        print(f"Features with >10% missing: {high_missing}")
        
        # Feature correlation analysis
        try:
            # Sample users for correlation analysis
            sample_users = self.df['user_id'].value_counts().head(20).index
            sample_df = self.df[self.df['user_id'].isin(sample_users)]
            
            le = LabelEncoder()
            y_sample = le.fit_transform(sample_df['user_id'])
            X_sample = sample_df[feature_cols]
            
            # Calculate feature-target correlations
            correlations = []
            for col in feature_cols:
                try:
                    corr = np.corrcoef(X_sample[col], y_sample)[0, 1]
                    if not np.isnan(corr):
                        correlations.append((col, abs(corr)))
                except:
                    pass
            
            correlations.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\nTop 5 features by correlation with user_id:")
            for feat, corr in correlations[:5]:
                print(f"  {feat}: {corr:.4f}")
            
            if correlations and correlations[0][1] < 0.1:
                self.issues.append("CRITICAL: Highest feature correlation with users < 0.1")
                self.recommendations.append("Features may not be discriminative enough")
                
        except Exception as e:
            print(f"Could not compute correlations: {e}")
    
    def _analyze_user_patterns(self):
        """Check if users have distinct patterns"""
        print(f"\n3. USER PATTERN ANALYSIS")
        print("-" * 40)
        
        # Focus on users with multiple samples
        user_counts = self.df['user_id'].value_counts()
        multi_sample_users = user_counts[user_counts >= 3].head(10).index
        
        if len(multi_sample_users) < 3:
            self.issues.append("CRITICAL: Less than 3 users with 3+ samples")
            self.recommendations.append("Need more data per user for pattern detection")
            return
        
        print(f"Analyzing patterns for {len(multi_sample_users)} users with 3+ samples:")
        
        # Analyze key features for these users
        key_features = ['cpu_time_min', 'memory_mean', 'code_size_mean', 'total_lines_mean']
        available_features = [f for f in key_features if f in self.df.columns]
        
        user_stats = []
        for user in multi_sample_users:
            user_data = self.df[self.df['user_id'] == user][available_features]
            user_mean = user_data.mean()
            user_std = user_data.std()
            user_stats.append({
                'user': user,
                'samples': len(user_data),
                **{f'{feat}_mean': user_mean[feat] for feat in available_features},
                **{f'{feat}_std': user_std[feat] for feat in available_features}
            })
        
        stats_df = pd.DataFrame(user_stats)
        print(f"\nUser statistics (first 5 users):")
        print(stats_df.head().round(2))
        
        # Check variability between users vs within users
        for feat in available_features:
            if f'{feat}_mean' in stats_df.columns:
                between_user_var = stats_df[f'{feat}_mean'].var()
                avg_within_user_var = stats_df[f'{feat}_std'].fillna(0).mean()
                
                ratio = between_user_var / (avg_within_user_var + 0.001)  # Avoid division by zero
                print(f"{feat} - Between/Within user variance ratio: {ratio:.2f}")
                
                if ratio < 2:
                    self.issues.append(f"LOW: {feat} has low discrimination (ratio: {ratio:.2f})")
    
    def _check_data_leakage(self):
        """Check for potential data leakage issues"""
        print(f"\n4. DATA LEAKAGE CHECK")
        print("-" * 40)
        
        # Check if problem_id is related to user_id
        user_problem_counts = self.df.groupby('user_id')['problem_id'].nunique()
        
        print(f"Average problems per user: {user_problem_counts.mean():.2f}")
        print(f"Users solving only 1 problem: {(user_problem_counts == 1).sum()}")
        
        if (user_problem_counts == 1).sum() > len(user_problem_counts) * 0.8:
            self.issues.append("CRITICAL: Most users solve only 1 problem - potential data leakage")
            self.recommendations.append("This makes it problem identification, not user identification")
    
    def _test_baseline_separability(self):
        """Test if users are actually separable with current features"""
        print(f"\n5. BASELINE SEPARABILITY TEST")
        print("-" * 40)
        
        # Filter to users with multiple samples
        user_counts = self.df['user_id'].value_counts()
        good_users = user_counts[user_counts >= 3].index
        
        if len(good_users) < 5:
            print("Not enough users with multiple samples for testing")
            return
        
        # Take top users by sample count
        test_df = self.df[self.df['user_id'].isin(good_users[:min(10, len(good_users))])]
        
        feature_cols = [col for col in test_df.columns if col not in ['user_id', 'problem_id']]
        X = test_df[feature_cols].fillna(0)
        
        le = LabelEncoder()
        y = le.fit_transform(test_df['user_id'])
        
        print(f"Testing with {len(good_users[:10])} users, {len(test_df)} samples")
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
            
            # Simple Random Forest test
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            rf.fit(X_train, y_train)
            
            train_acc = rf.score(X_train, y_train)
            test_acc = rf.score(X_test, y_test)
            
            print(f"Quick RF test - Train: {train_acc:.3f}, Test: {test_acc:.3f}")
            
            if test_acc < 0.3:
                self.issues.append("CRITICAL: Even with filtered data, accuracy < 30%")
                self.recommendations.append("Features may not capture user coding style")
            elif test_acc > 0.7:
                print("Good news: Users are separable with proper filtering!")
            
        except Exception as e:
            print(f"Could not perform separability test: {e}")
    
    def _generate_recommendations(self):
        """Generate specific recommendations"""
        print(f"\n{'='*60}")
        print("ISSUES FOUND")
        print("="*60)
        
        for i, issue in enumerate(self.issues, 1):
            print(f"{i}. {issue}")
        
        print(f"\n{'='*60}")
        print("RECOMMENDATIONS")
        print("="*60)
        
        for i, rec in enumerate(self.recommendations, 1):
            print(f"{i}. {rec}")
        
        return self.issues, self.recommendations
    
    def create_improved_dataset(self, output_path, strategy='filter_users'):
        """Create an improved version of the dataset"""
        print(f"\n{'='*60}")
        print(f"CREATING IMPROVED DATASET - Strategy: {strategy}")
        print("="*60)
        
        if strategy == 'filter_users':
            return self._filter_users_strategy(output_path)
        elif strategy == 'aggregate_users':
            return self._aggregate_users_strategy(output_path)
        elif strategy == 'top_users_only':
            return self._top_users_strategy(output_path)
        else:
            print("Unknown strategy")
            return None
    
    def _filter_users_strategy(self, output_path):
        """Keep only users with sufficient samples"""
        print("Strategy: Filter users with insufficient samples")
        
        # Keep users with at least 3 samples
        user_counts = self.df['user_id'].value_counts()
        good_users = user_counts[user_counts >= 3].index
        
        filtered_df = self.df[self.df['user_id'].isin(good_users)]
        
        print(f"Original: {len(self.df)} samples, {self.df['user_id'].nunique()} users")
        print(f"Filtered: {len(filtered_df)} samples, {filtered_df['user_id'].nunique()} users")
        
        filtered_df.to_csv(output_path, index=False)
        print(f"Saved to: {output_path}")
        
        return filtered_df
    
    def _aggregate_users_strategy(self, output_path):
        """Aggregate multiple submissions per user-problem pair"""
        print("Strategy: Aggregate submissions per user-problem")
        
        # Group by user and problem, aggregate features
        agg_dict = {}
        
        # Numerical features - use mean
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['user_id', 'problem_id']:
                agg_dict[col] = 'mean'
        
        aggregated_df = self.df.groupby(['user_id', 'problem_id']).agg(agg_dict).reset_index()
        
        print(f"Original: {len(self.df)} samples")
        print(f"Aggregated: {len(aggregated_df)} samples")
        
        aggregated_df.to_csv(output_path, index=False)
        print(f"Saved to: {output_path}")
        
        return aggregated_df
    
    def _top_users_strategy(self, output_path):
        """Keep only top N users with most samples"""
        print("Strategy: Keep only top users by sample count")
        
        # Get top 20 users by sample count
        user_counts = self.df['user_id'].value_counts()
        top_users = user_counts.head(20).index
        
        top_df = self.df[self.df['user_id'].isin(top_users)]
        
        print(f"Original: {len(self.df)} samples, {self.df['user_id'].nunique()} users")
        print(f"Top users only: {len(top_df)} samples, {top_df['user_id'].nunique()} users")
        
        # Show distribution
        print("\nSample distribution for top users:")
        print(top_df['user_id'].value_counts())
        
        top_df.to_csv(output_path, index=False)
        print(f"Saved to: {output_path}")
        
        return top_df

def main():
    analyzer = DatasetAnalyzer()
    
    # Update this path to your dataset
    input_file = "E:/thesis/trimmed_file6.csv"
    
    try:
        # Analyze current dataset
        issues, recommendations = analyzer.load_and_analyze(input_file)
        
        # Ask user which strategy to use
        print(f"\n{'='*60}")
        print("DATASET IMPROVEMENT OPTIONS")
        print("="*60)
        print("1. filter_users - Remove users with < 3 samples")
        print("2. aggregate_users - Combine multiple submissions per user-problem") 
        print("3. top_users_only - Keep only top 20 users by sample count")
        
        strategy = input("\nWhich strategy do you want to try? (1/2/3): ").strip()
        
        strategy_map = {'1': 'filter_users', '2': 'aggregate_users', '3': 'top_users_only'}
        
        if strategy in strategy_map:
            output_file = f"improved_dataset_{strategy_map[strategy]}.csv"
            improved_df = analyzer.create_improved_dataset(output_file, strategy_map[strategy])
            
            print(f"\nImproved dataset created: {output_file}")
            print("Try training your model on this new dataset!")
            
        else:
            print("Invalid choice")
            
    except FileNotFoundError:
        print(f"File not found: {input_file}")
        print("Please update the file path in the code")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()