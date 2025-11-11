# scripts/visualization.py
"""
Visualization utilities for NiCr Alloy ML Project
Handles stress-strain curves, model performance, and property comparisons.
"""

import matplotlib.pyplot as plt
import pandas as pd
import os
from scripts.visualization.plot_results import plot_model_results

def plot_stress_strain(csv_path, save_dir="visualization"):
    """Plot stress-strain curves from CSV"""
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
    os.makedirs(save_dir, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    if "strain" not in df.columns or "stress" not in df.columns:
        print("CSV must contain 'strain' and 'stress' columns.")
        return
    
    plt.figure(figsize=(6,4))
    plt.plot(df["strain"], df["stress"], label="Stress-Strain Curve", color="blue")
    plt.xlabel("Strain")
    plt.ylabel("Stress (GPa)")
    plt.title("NiCr Alloy Stress-Strain Curve")
    plt.legend()
    plt.tight_layout()
    
    out_file = os.path.join(save_dir, "stress_strain_curve.png")
    plt.savefig(out_file)
    plt.close()
    print(f"✓ Saved plot to {out_file}")

def plot_model_performance(results_df, save_dir="visualization"):
    """Plot comparison of ML model performances"""
    os.makedirs(save_dir, exist_ok=True)
    metrics = ["Test R²", "Test MAE", "Test RMSE"]
    for metric in metrics:
        plt.figure(figsize=(6,4))
        plt.bar(results_df.index, results_df[metric])
        plt.ylabel(metric)
        plt.title(f"Model Comparison: {metric}")
        plt.tight_layout()
        path = os.path.join(save_dir, f"{metric.replace(' ', '_')}.png")
        plt.savefig(path)
        plt.close()
        print(f"✓ Saved plot to {path}")
