#!/usr/bin/env python3
"""
Main execution script for NiCr Alloy Mechanical Properties Prediction Project
"""

import sys
from pathlib import Path
import subprocess
import argparse
import os

def create_directories_safe():
    """Safely create directories without throwing errors if they exist"""
    directories = [
        'data/raw', 'data/processed', 
        'lammps/outputs', 'lammps/inputs', 'lammps/potentials',
        'visualization', 'ml_models'
    ]
    
    for dir_path in directories:
        try:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"✓ Directory ready: {dir_path}")
        except FileExistsError:
            print(f"✓ Directory already exists: {dir_path}")
        except Exception as e:
            print(f"✗ Error creating {dir_path}: {e}")

def check_prerequisites():
    """Check if all prerequisites are met"""
    try:
        from scripts.automation.run_simulations import LAMMPSRunner
    except ImportError as e:
        print(f"ERROR: Could not import required modules: {e}")
        return False
    
    runner = LAMMPSRunner()
    if runner.lammps_executable is None:
        print("ERROR: LAMMPS not found. Please install LAMMPS first.")
        print("Run: conda install -c conda-forge lammps")
        return False
    
    # Check potential files
    potential_files = ['lammps/potentials/library.meam', 'lammps/potentials/CrNi.meam']
    missing_files = []
    
    for file in potential_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("ERROR: Missing potential files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease download from NIST Interatomic Potentials Repository:")
        print("https://www.ctcms.nist.gov/potentials/")
        print("Search for: 2025--Sharifi-H--Cr-Ni--LAMMPS--ipr1")
        print("Download: library.meam and CrNi.meam")
        return False
    
    print("✓ All prerequisites met!")
    return True

def main():
    parser = argparse.ArgumentParser(description='NiCr Alloy ML Project Pipeline')
    parser.add_argument('--run-simulations', action='store_true', 
                       help='Run LAMMPS simulations')
    parser.add_argument('--parse-outputs', action='store_true',
                       help='Parse LAMMPS outputs')
    parser.add_argument('--train-models', action='store_true',
                       help='Train ML models')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations')
    parser.add_argument('--all', action='store_true',
                       help='Run complete pipeline')
    parser.add_argument('--check-setup', action='store_true',
                       help='Check system setup')
    parser.add_argument('--test-only', action='store_true',
                       help='Run only a single test simulation')
    
    args = parser.parse_args()
    
    if args.check_setup:
        try:
            subprocess.run([sys.executable, "check_setup.py"])
        except FileNotFoundError:
            print("check_setup.py not found. Running internal check...")
            create_directories_safe()
            check_prerequisites()
        return
    
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    print("NiCr Alloy ML Project Pipeline")
    print("=" * 50)
    
    # Create directories safely
    create_directories_safe()
    
    if args.run_simulations or args.all or args.test_only:
        print("\nStep 1: Running LAMMPS simulations...")
        if not check_prerequisites():
            print("Cannot run simulations due to missing prerequisites.")
            if not args.all:
                return
        
        from scripts.automation.run_simulations import LAMMPSRunner
        runner = LAMMPSRunner()
        
        if args.test_only:
            # Run only test simulation
            print("Running test simulation only...")
            success = runner.run_test_simulation()
            if success:
                print("✓ Test simulation completed successfully!")
            else:
                print("✗ Test simulation failed!")
            return
        
        # Run test first, then full sweep
        print("Running test simulation...")
        if runner.run_test_simulation():
            print("✓ Test successful! Running parameter sweep...")
            df_results = runner.run_parameter_sweep(
                compositions=[0.7, 0.8, 0.9],
                temperatures=[300, 600, 900],
                strain_rates=[1e8, 1e9, 1e10]
            )
            df_results.to_csv('data/raw/simulation_log.csv', index=False)
        else:
            print("✗ Test simulation failed. Please check the setup.")
            if not args.all:
                return
    
    if args.parse_outputs or args.all:
        print("\nStep 2: Parsing simulation outputs...")
        try:
            from scripts.automation.parse_outputs import LAMMPSOutputParser
            parser = LAMMPSOutputParser()
            df = parser.process_all_simulations()
            
            if df is None:
                print("No simulation outputs found to parse.")
                # Create a dummy dataset for testing if no simulations ran
                if args.all:
                    print("Creating sample dataset for testing...")
                    create_sample_dataset()
            else:
                print(f"✓ Successfully parsed {len(df)} simulations")
                
        except Exception as e:
            print(f"Error parsing outputs: {e}")
            if args.all:
                print("Creating sample dataset for testing...")
                create_sample_dataset()
    
    if args.train_models or args.all:
        print("\nStep 3: Training machine learning models...")
        # Check if data exists
        data_file = Path("data/processed/mechanical_properties.csv")
        if not data_file.exists():
            print("No data found for training. Using sample data for demonstration.")
            create_sample_dataset()
        
        try:
            # Feature engineering
            from scripts.ml_pipeline.make_features import FeatureEngineer
            engineer = FeatureEngineer()
            df = engineer.load_data("data/processed/mechanical_properties.csv")
            
            targets = ['youngs_modulus', 'yield_strength', 'uts']
            available_targets = [target for target in targets if target in df.columns]
            
            if not available_targets:
                print("No target columns found in dataset.")
                return
            
            for target in available_targets:
                print(f"Preparing data for {target}...")
                data_dict = engineer.prepare_ml_data(target)
                import joblib
                joblib.dump(data_dict, f"data/processed/{target}_data.joblib")
            
            engineer.save_preprocessor("ml_models/preprocessor.joblib")
            
            # Model training
            from scripts.ml_pipeline.train_models import run_ml_pipeline
            for target in available_targets:
                print(f"Training models for {target}...")
                run_ml_pipeline(target)
                
        except Exception as e:
            print(f"Error in training pipeline: {e}")
    
    if args.visualize or args.all:
        print("\nStep 4: Generating visualizations...")
        try:
            from scripts.visualization.plot_results import ResultsVisualizer
            visualizer = ResultsVisualizer()
            visualizer.create_summary_dashboard()
            print("✓ Visualizations created successfully!")
        except Exception as e:
            print(f"Error creating visualizations: {e}")
    
    print("\n" + "=" * 50)
    print("Project pipeline completed!")

def create_sample_dataset():
    """Create a sample dataset for testing when no simulations are available"""
    import pandas as pd
    import numpy as np
    
    print("Creating sample dataset for testing...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 50
    
    data = []
    for i in range(n_samples):
        composition = np.random.uniform(0.7, 0.9)
        temperature = np.random.choice([300, 600, 900])
        strain_rate = np.random.choice([1e8, 1e9, 1e10])
        
        # Simulate mechanical properties based on known trends
        base_youngs = 200 - 50 * (1 - composition)  # Young's modulus decreases with Cr
        youngs_modulus = base_youngs + np.random.normal(0, 5)
        
        base_yield = 0.5 - 0.2 * (1 - composition)  # Yield strength
        yield_strength = base_yield + np.random.normal(0, 0.05)
        
        base_uts = 0.8 - 0.3 * (1 - composition)  # UTS
        uts = base_uts + np.random.normal(0, 0.05)
        
        data.append({
            'composition': composition,
            'temperature': temperature,
            'strain_rate': strain_rate,
            'youngs_modulus': youngs_modulus,
            'yield_strength': yield_strength,
            'uts': uts
        })
    
    df = pd.DataFrame(data)
    df.to_csv("data/processed/mechanical_properties.csv", index=False)
    print("✓ Sample dataset created: data/processed/mechanical_properties.csv")
    
    # Also create some sample stress-strain curves
    import os
    os.makedirs("data/raw", exist_ok=True)
    
    for i in range(5):
        strains = np.linspace(0, 0.2, 100)
        stresses = 0.5 * strains + 0.1 * np.sin(10 * strains) + np.random.normal(0, 0.01, 100)
        
        curve_df = pd.DataFrame({
            'strain': strains,
            'stress': stresses
        })
        curve_df.to_csv(f"data/raw/sample_stress_strain_{i}.csv", index=False)
    
    print("✓ Sample stress-strain curves created")

if __name__ == "__main__":
    main()