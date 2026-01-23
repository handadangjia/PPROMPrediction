# PPROM Risk Prediction Tool

## Overview

This project provides a **GUI application** for predicting **Preterm Premature Rupture of Membranes (PPROM)** risk using a **pre-trained XGBoost model**. The tool allows users to:

- Upload clinical feature data in CSV or Excel format.
- Predict PPROM risk for each sample.
- Visualize **SHAP explanations** for individual samples, highlighting feature contributions to the prediction.
- Display results directly in the GUI without exposing raw model thresholds.

This tool is designed for research purposes and does **not** use or expose real patient data.

---

## Features

1. **Prediction**
   - Loads a fixed XGBoost model (`xgboost_model.json`).
   - Supports batch prediction of multiple samples.
   - Displays probability and classification (`PPROM` / `non-PPROM`) in a table.

2. **SHAP Explainability**
   - Select a sample in the result table to view its SHAP feature contribution.
   - SHAP plots are embedded in the GUI, showing features with positive (red) and negative (blue) contributions.
   - Uses `shap.TreeExplainer` for interpretability.

3. **User-Friendly GUI**
   - Built with **Tkinter** for cross-platform compatibility.
   - Supports CSV and Excel file inputs.
   - No need for manual command-line operations.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/handadangjia/PPROMPrediction
cd PPROMPrediction

2. Install dependencies:

```bash
pip install -r requirements.txt

## Usage
Run the GUI:

```bash
python pprom_prediction_gui_shap.py

Upload feature data via the "Browse Data" button.
Click "Run Prediction" to populate the table with predicted probabilities.
Select a sample in the table and click "Show SHAP Explanation" to view the embedded SHAP force/bar plot for that sample.

Requirements：
Python 3.9+
XGBoost
pandas
numpy
shap
matplotlib
openpyxl (for Excel support)
tkinter (usually included in Python standard library)

Notes:
This application is designed for simulated or de-identified clinical datasets to ensure patient privacy.
The tool supports binary classification (PPROM vs. non-PPROM).
For reproducibility, use the exact feature order as used in the demo data.

