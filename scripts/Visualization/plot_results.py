import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib

matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['figure.figsize'] = (10, 6)

class ResultsVisualizer:
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        self.setup_style()
    
    def setup_style(self):
        """Setup plotting style"""
        sns.set_style("whitegrid")
        plt.rcParams['font.family'] = 'serif'
    
    def plot_stress_strain_curves(self, data_dir="data/raw"):
        """Plot multiple stress-strain curves"""
        data_dir = self.base_dir / data_dir
        curves = list(data_dir.glob("stress_strain_*.csv"))
        
        if not curves:
            print("No stress-strain curves found")
            return
        
        plt.figure(figsize=(12, 8))
        
        for curve_file in curves[:10]:  # Plot first 10 curves
            df = pd.read_csv(curve_file)
            if len(df) > 0:
                # Extract parameters from filename
                filename = curve_file.stem
                comp = float(filename.split('comp')[1].split('_')[0])
                temp = int(filename.split('temp')[1].split('_')[0])
                rate = float(filename.split('rate')[1])
                
                plt.plot(df['strain'], df['stress'], 
                        label=f'Ni{comp*100:.0f}Cr{(1-comp)*100:.0f}, {temp}K, {rate:.1e}/s',
                        alpha=0.7, linewidth=2)
        
        plt.xlabel('Strain')
        plt.ylabel('Stress (GPa)')
        plt.title('NiCr Alloy - Stress-Strain Curves from MD Simulations')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('visualization/stress_strain_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_predicted_vs_actual(self, target_property):
        """Plot predicted vs actual values for all models"""
        try:
            data_dict = joblib.load(f"data/processed/{target_property}_data.joblib")
            models_metrics = pd.read_csv(f"ml_models/{target_property}_model_metrics.csv", index_col=0)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.ravel()
            
            for idx, (model_name, metrics) in enumerate(models_metrics.iterrows()):
                model = joblib.load(f"ml_models/{target_property}_{model_name.replace(' ', '_').lower()}.joblib")
                
                # Predictions
                if hasattr(data_dict['X_test_poly'], 'shape'):
                    X_test = data_dict['X_test_poly']
                else:
                    X_test = data_dict['X_test']
                
                y_pred = model.predict(X_test)
                y_true = data_dict['y_test'].values
                
                # Plot
                axes[idx].scatter(y_true, y_pred, alpha=0.6, s=50)
                axes[idx].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
                axes[idx].set_xlabel('Actual')
                axes[idx].set_ylabel('Predicted')
                axes[idx].set_title(f'{model_name}\nR² = {metrics["Test R²"]:.3f}, MAE = {metrics["Test MAE"]:.3f}')
                axes[idx].grid(True, alpha=0.3)
            
            plt.suptitle(f'{target_property.replace("_", " ").title()} - Predicted vs Actual', fontsize=16)
            plt.tight_layout()
            plt.savefig(f'visualization/{target_property}_predicted_vs_actual.png', dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"Error plotting {target_property}: {e}")
    
    def plot_feature_correlations(self):
        """Plot correlation matrix of features and targets"""
        try:
            df = pd.read_csv("data/processed/mechanical_properties.csv")
            
            # Create feature set
            engineer = joblib.load("ml_models/preprocessor.joblib")
            features = engineer.create_features(df)
            
            # Combine with targets
            combined_df = pd.concat([features, df[['youngs_modulus', 'yield_strength', 'uts']]], axis=1)
            
            # Plot correlation matrix
            plt.figure(figsize=(12, 10))
            corr_matrix = combined_df.corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                       square=True, fmt='.2f', cbar_kws={"shrink": .8})
            plt.title('Feature and Target Correlation Matrix')
            plt.tight_layout()
            plt.savefig('visualization/correlation_matrix.png', dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"Error plotting correlations: {e}")
    
    def create_summary_dashboard(self):
        """Create a comprehensive summary dashboard"""
        self.plot_stress_strain_curves()
        self.plot_feature_correlations()
        
        # Plot predicted vs actual for all targets
        for target in ['youngs_modulus', 'yield_strength', 'uts']:
            try:
                self.plot_predicted_vs_actual(target)
            except:
                continue
        
        print("Visualization completed! Check visualization/ directory for plots.")

if __name__ == "__main__":
    visualizer = ResultsVisualizer()
    visualizer.create_summary_dashboard()