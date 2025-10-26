import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class MLPipeline:
    def __init__(self):
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), 
                                         max_iter=1000, random_state=42)
        }
        self.results = {}
    
    def load_data(self, data_file):
        """Load prepared data"""
        return joblib.load(data_file)
    
    def train_models(self, data_dict, use_poly_features=False):
        """Train all models on the data"""
        if use_poly_features:
            X_train = data_dict['X_train_poly']
            X_test = data_dict['X_test_poly']
        else:
            X_train = data_dict['X_train']
            X_test = data_dict['X_test']
        
        y_train = data_dict['y_train']
        y_test = data_dict['y_test']
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Metrics
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            self.results[name] = {
                'model': model,
                'y_pred_train': y_pred_train,
                'y_pred_test': y_pred_test,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse
            }
            
            print(f"{name} - Test R²: {test_r2:.3f}, Test MAE: {test_mae:.3f}")
    
    def evaluate_models(self):
        """Evaluate and compare all models"""
        metrics_df = pd.DataFrame({
            name: {
                'Train R²': result['train_r2'],
                'Test R²': result['test_r2'],
                'Train MAE': result['train_mae'],
                'Test MAE': result['test_mae'],
                'Train RMSE': result['train_rmse'],
                'Test RMSE': result['test_rmse']
            }
            for name, result in self.results.items()
        }).T
        
        return metrics_df
    
    def save_models(self, target_property, output_dir="ml_models"):
        """Save trained models"""
        Path(output_dir).mkdir(exist_ok=True)
        
        for name, result in self.results.items():
            filename = f"{output_dir}/{target_property}_{name.replace(' ', '_').lower()}.joblib"
            joblib.dump(result['model'], filename)
        
        # Save results summary
        metrics_df = self.evaluate_models()
        metrics_df.to_csv(f"{output_dir}/{target_property}_model_metrics.csv")
        
        return metrics_df
    
    def plot_feature_importance(self, data_dict, model_name='Random Forest'):
        """Plot feature importance (for tree-based models)"""
        if model_name not in self.models or not hasattr(self.results[model_name]['model'], 'feature_importances_'):
            print(f"Cannot plot feature importance for {model_name}")
            return
        
        model = self.results[model_name]['model']
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_names = data_dict['feature_names'] if len(importance) == len(data_dict['feature_names']) else data_dict['original_features']
            
            # Create feature importance dataframe
            fi_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=True)
            
            # Plot
            plt.figure(figsize=(10, 6))
            plt.barh(fi_df['feature'], fi_df['importance'])
            plt.xlabel('Feature Importance')
            plt.title(f'Feature Importance - {model_name}')
            plt.tight_layout()
            plt.savefig(f'visualization/{model_name.lower().replace(" ", "_")}_feature_importance.png', dpi=300)
            plt.close()

def run_ml_pipeline(target_property):
    """Complete ML pipeline for a target property"""
    print(f"\n{'='*50}")
    print(f"Training ML models for {target_property}")
    print(f"{'='*50}")
    
    # Load data
    pipeline = MLPipeline()
    data_dict = pipeline.load_data(f"data/processed/{target_property}_data.joblib")
    
    # Train models (with polynomial features)
    pipeline.train_models(data_dict, use_poly_features=True)
    
    # Evaluate
    metrics_df = pipeline.evaluate_models()
    print(f"\nModel Performance for {target_property}:")
    print(metrics_df.round(4))
    
    # Save models and results
    pipeline.save_models(target_property)
    
    # Plot feature importance
    pipeline.plot_feature_importance(data_dict)
    
    return pipeline, metrics_df

if __name__ == "__main__":
    # Train models for all mechanical properties
    targets = ['youngs_modulus', 'yield_strength', 'uts']
    
    all_results = {}
    for target in targets:
        try:
            pipeline, metrics = run_ml_pipeline(target)
            all_results[target] = metrics
        except Exception as e:
            print(f"Error processing {target}: {e}")
    
    # Create summary report
    summary_df = pd.concat(all_results, axis=1)
    summary_df.to_csv("ml_models/model_performance_summary.csv")
    print("\nModel training completed! Check ml_models/ directory for results.")