import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
import joblib

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
    
    def load_data(self, filepath):
        """Load mechanical properties data"""
        self.df = pd.read_csv(filepath)
        print(f"Loaded dataset with {len(self.df)} samples")
        return self.df
    
    def create_features(self, df):
        """Create feature set from raw parameters"""
        # Basic features
        features = df[['composition', 'temperature', 'strain_rate']].copy()
        
        # Feature engineering
        features['cr_content'] = 1 - features['composition']
        features['log_strain_rate'] = np.log10(features['strain_rate'])
        features['temp_composition_interaction'] = features['temperature'] * features['cr_content']
        features['rate_temp_interaction'] = features['strain_rate'] * features['temperature']
        
        # Physical properties approximations
        features['melting_point_approx'] = 1728 - 500 * features['cr_content']  # Approximate melting point
        features['lattice_parameter'] = 3.52 + 0.1 * features['cr_content']  # Approximate lattice parameter
        
        return features
    
    def prepare_ml_data(self, target_property='youngs_modulus'):
        """Prepare data for machine learning"""
        if target_property not in self.df.columns:
            raise ValueError(f"Target property {target_property} not in dataset")
        
        # Create features
        X = self.create_features(self.df)
        y = self.df[target_property]
        
        # Remove rows with NaN values
        valid_mask = ~y.isna() & ~X.isna().any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"Valid samples for {target_property}: {len(X)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create polynomial features
        X_train_poly = self.poly.fit_transform(X_train_scaled)
        X_test_poly = self.poly.transform(X_test_scaled)
        
        feature_names = self.poly.get_feature_names_out(X.columns)
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'X_train_poly': X_train_poly,
            'X_test_poly': X_test_poly,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names,
            'original_features': X.columns
        }
    
    def save_preprocessor(self, filepath):
        """Save scaler and polynomial transformer"""
        joblib.dump({
            'scaler': self.scaler,
            'poly': self.poly
        }, filepath)

if __name__ == "__main__":
    # Example usage
    engineer = FeatureEngineer()
    df = engineer.load_data("data/processed/mechanical_properties.csv")
    
    # Prepare data for different targets
    targets = ['youngs_modulus', 'yield_strength', 'uts']
    
    for target in targets:
        print(f"\nPreparing data for {target}")
        data_dict = engineer.prepare_ml_data(target)
        
        # Save prepared data
        joblib.dump(data_dict, f"data/processed/{target}_data.joblib")
    
    # Save preprocessor
    engineer.save_preprocessor("ml_models/preprocessor.joblib")