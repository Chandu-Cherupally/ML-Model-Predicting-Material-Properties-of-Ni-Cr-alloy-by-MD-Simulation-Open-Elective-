# 🧩 Machine Learning-Based Prediction of Mechanical Properties of Ni–Cr Alloy

### Author: **Chandu Cherupally (22CSB0C20, CSE-A)**
### Project Area: **Machine Learning for Material Informatics**

---

## 🧠 Project Overview

This project integrates **Molecular Dynamics (MD) simulations** with **Machine Learning (ML)** models to predict the **mechanical properties** of Nickel-Chromium (Ni–Cr) alloys.  

The aim is to accelerate materials design by using ML to learn from simulation data, reducing the need for repeated computationally expensive molecular dynamics experiments.

The properties predicted include:
- **Young’s Modulus**
- **Yield Strength**
- **Ultimate Tensile Strength (UTS)**

---

## ⚙️ Workflow Summary

The project follows a 5-stage workflow:

1. **Data Generation (LAMMPS Simulations)**  
   - Molecular dynamics simulations of Ni–Cr alloys are automated using Python scripts (`run_simulations.py`).  
   - Each simulation outputs stress–strain data as CSV files stored in `Data/raw/`.

2. **Data Processing & Feature Extraction**  
   - Raw stress–strain data are parsed using `parse_outputs.py` and processed into numerical features using `make_features.py`.  
   - Derived quantities include UTS, yield point, and elastic modulus.

3. **Dataset Preparation**  
   - All processed data are saved in `Data/processed/mechanical_properties.csv` and split into individual `.joblib` datasets for each target variable.

4. **Model Training and Evaluation**  
   - Multiple regression models are trained using `train_models.py`:
     - Linear Regression  
     - Ridge Regression  
     - Random Forest Regressor  
     - Multi-Layer Perceptron (Neural Network)
   - Performance is measured using **MAE**, **RMSE**, and **R² score**.

5. **Visualization & Result Analysis**  
   - Feature importance and performance plots are generated (`plot_results.py`), showing the predictive capability of each model.

---

## 🧩 Project Structure
MLMI/
├── run_project.py # Main entry script to run the full pipeline
├── check_setup.py # Environment and dependency checker
│
├── Data/
│ ├── raw/ # Raw MD stress–strain CSVs
│ ├── processed/ # Cleaned datasets and .joblib files
│
├── Lammps/
│ ├── inputs/ # LAMMPS input scripts (minimize, equilibrate, tensile)
│ └── potentials.txt # Potential files used for Ni–Cr
│
├── scripts/
│ ├── automation/ # MD automation and parsing
│ │ ├── run_simulations.py
│ │ └── parse_outputs.py
│ ├── ml_pipeline/ # Feature extraction and model training
│ │ ├── make_features.py
│ │ └── train_models.py
│ └── Visualization/ # Plotting utilities
│ └── plot_results.py
│
└── ML_models/
├── model.joblib # Saved trained models
├── model_metrics.csv # Evaluation metrics for each property
└── preprocessor.joblib # Data scaler used during training


---

## 📊 Machine Learning Models

| Model | Description | Strengths |
|-------|--------------|------------|
| **Linear Regression** | Simple baseline regression model | Interpretable coefficients |
| **Ridge Regression** | Linear model with regularization | Reduces overfitting |
| **Random Forest** | Ensemble of decision trees | High accuracy & feature importance |
| **MLP Regressor** | Neural network-based model | Captures nonlinear relationships |

Each property (Young’s modulus, yield strength, UTS) is modeled independently.

---

## 📈 Evaluation Metrics

Models are evaluated using:

- **MAE (Mean Absolute Error)**  
- **MSE / RMSE (Mean Squared / Root Mean Squared Error)**  
- **R² Score (Coefficient of Determination)**  

These metrics are stored in:
ML_models/
├── youngs_modulus_model_metrics.csv
├── yield_strength_model_metrics.csv
└── uts_model_metrics.csv

---

## 🧪 How to Run the Project

### 1️⃣ Set Up Environment
Install dependencies (Python 3.10+):
```bash
pip install -r requirements.txt
2️⃣ Run Complete Pipeline
python run_project.py --all

This will:
Run MD simulations (if enabled)
Process data
Train all ML models
Save metrics and visualizations

3️⃣ Train ML Models Only
python scripts/ml_pipeline/train_models.py

📊 Results Summary (example layout)
| Property        | Best Model       | R² Score | MAE | RMSE |
| --------------- | ---------------- | -------- | --- | ---- |
| Young’s Modulus | Random Forest    | 0.95     | 3.2 | 4.1  |
| Yield Strength  | MLP              | 0.92     | 5.4 | 6.3  |
| UTS             | Ridge Regression | 0.89     | 7.1 | 8.0  |

