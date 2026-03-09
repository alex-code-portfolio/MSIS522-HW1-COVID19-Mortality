# MSIS 522 HW1 — COVID-19 Mortality Prediction

**Author:** Alex Salcedo | University of Washington, Foster School of Business

## Overview

End-to-end machine learning pipeline predicting COVID-19 patient mortality using Logistic Regression, Decision Trees, Random Forest, LightGBM, and a Neural Network (MLP). Includes SHAP explainability analysis and a deployed Streamlit web application.

## Live App

https://msis522-hw1-covid19-mortality-alexsalcedo.streamlit.app/

## Project Structure

```
MSIS522_HW1/
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── README.md
├── .gitignore
├── models/
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── lightgbm_model.pkl
│   ├── neural_network.keras
│   └── scaler.pkl
└── notebooks/
    └── MSIS522_HW1_AlexSalcedo_GoldCohort.ipynb
```

## How to Run Locally

1. Clone the repo:
   ```bash
   git clone <repo-url>
   cd MSIS522_HW1
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```
   The dataset is downloaded automatically from Google Drive on first launch.

## Dataset

~1M anonymized COVID-19 patient records, balanced to 10,000 (5,000 died / 5,000 survived) for training. 16 features including age, sex, hospitalization status, and pre-existing comorbidities. Source: instructor-provided via Google Drive.

## Models and Best Hyperparameters

| Model | Key Hyperparameters |
|---|---|
| Logistic Regression | max_iter=1000 (baseline) |
| Decision Tree | max_depth=4, min_samples_leaf=50 |
| Random Forest | max_depth=8, n_estimators=200 |
| LightGBM | learning_rate=0.05, max_depth=4, n_estimators=50 |
| Neural Network | 2 hidden layers (128 units), ReLU, Adam, 5 epochs |

## Key Findings

- Random Forest achieved the best AUC-ROC (0.9505)
- Neural Network achieved the highest recall (0.9460) and F1 (0.9045)
- HOSPITALIZED is the single strongest predictor of mortality per SHAP analysis
- All models performed competitively (AUC range: 0.9448–0.9505)
