# MSIS 522 HW1 — Cursor Agent Context
## COVID-19 Mortality Prediction: Streamlit App + GitHub Repo

---

## WHO YOU ARE HELPING

This project is for Alex Salcedo, a graduate student in the UW Foster School of Business
MSIS program, completing HW1 for MSIS 522 (Advanced Machine Learning). Alex has already
completed all analysis in a Google Colab notebook. Your job is to help build the Streamlit
app and GitHub repo that wrap and present that work as final deliverables.

---

## WHAT HAS ALREADY BEEN DONE (in Google Colab)

All models have been trained and evaluated. Here is a summary of everything completed:

### Dataset
- COVID-19 patient mortality dataset (~1M records, balanced to 10,000: 5,000 died / 5,000 survived)
- 16 features: SEX, HOSPITALIZED, PNEUMONIA, AGE, PREGNANT, DIABETES, COPD, ASTHMA,
  IMMUNOSUPPRESSION, HYPERTENSION, OTHER_DISEASE, CARDIOVASCULAR, OBESITY, RENAL_CHRONIC,
  TOBACCO, COVID_POSITIVE
- Target variable: DEATH (binary: 0 = survived, 1 = died)
- Train/test split: 70/30, random_state=42

### Models Trained and Their Test Set Results

| Model            | Accuracy | Precision | Recall | F1     | AUC-ROC |
|------------------|----------|-----------|--------|--------|---------|
| Decision Tree    | 0.8977   | 0.8736    | 0.9356 | 0.9036 | 0.9456  |
| Random Forest    | 0.8990   | 0.8712    | 0.9421 | 0.9053 | 0.9505  |
| LightGBM         | 0.8963   | 0.8720    | 0.9349 | 0.9024 | 0.9501  |
| Neural Network   | TBD      | TBD       | TBD    | TBD    | TBD     |

**Best model: Random Forest** (max_depth=8, n_estimators=200)
**Best Decision Tree params:** max_depth=4, min_samples_leaf=50
**Best LightGBM params:** learning_rate=0.05, max_depth=4, n_estimators=50
**Optimal LightGBM threshold:** 0.39 (maximizes F1, reduces false negatives)

### SHAP Analysis (on best LightGBM model)
Top features by importance:
1. HOSPITALIZED (~1.6 mean |SHAP|) — strongest predictor by far
2. AGE (~0.45) — older patients = higher mortality risk
3. PNEUMONIA (~0.40) — presence strongly increases mortality risk
4. COVID_POSITIVE (~0.20)
5. SEX, RENAL_CHRONIC, DIABETES — lower but present impact

Waterfall plot high-risk patient: 83-year-old, hospitalized, pneumonia, COVID positive,
diabetic. Final model output: 2.497 (very high mortality probability).

---

## PROJECT FOLDER STRUCTURE

Create the following structure locally:

```
msis522-hw1/
├── app.py                        # Main Streamlit application
├── requirements.txt              # All dependencies
├── README.md                     # Project documentation
├── data/
│   └── covid.csv                 # Raw dataset (download separately — see README)
├── models/
│   ├── decision_tree.pkl         # Saved best Decision Tree model
│   ├── random_forest.pkl         # Saved best Random Forest model
│   ├── lightgbm_model.pkl        # Saved best LightGBM model
│   ├── neural_network.keras      # Saved Keras model
│   └── scaler.pkl                # Saved StandardScaler (for Neural Network)
├── notebooks/
│   └── MSIS522_HW1_AlexSalcedo.ipynb   # Completed Colab notebook (exported)
└── assets/
    └── plots/                    # Optional: pre-saved plot images if needed
```

---

## HOW TO SAVE MODELS FROM COLAB

Add this code at the end of the Colab notebook to save all models before building the app:

```python
import joblib

# Save sklearn models
joblib.dump(grid_search.best_estimator_, 'decision_tree.pkl')
joblib.dump(grid_search_rf.best_estimator_, 'random_forest.pkl')
joblib.dump(grid_search_lgb.best_estimator_, 'lightgbm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Save Keras model
model.save('neural_network.keras')

print("All models saved.")
```

Download all .pkl and .keras files from Colab and place them in the models/ folder.

---

## STREAMLIT APP REQUIREMENTS

The app must have exactly 4 tabs using st.tabs(). Here is what each tab needs:

### Tab 1 — Executive Summary
- A written description of the dataset and prediction task (at least 1 paragraph)
- Why this problem matters clinically — the "so what" (at least 1 paragraph)
- Brief overview of approach and key findings (1-2 paragraphs)
- Must be written for a non-technical stakeholder — not just metrics
- Do NOT make this AI-generated sounding. Write it as a professional summary.

### Tab 2 — Descriptive Analytics
Display all Part 1 visualizations with captions. Recreate them using the same logic
from the notebook:
- Age distribution histogram (sns.histplot)
- Age by outcome boxplot (sns.boxplot, x=DEATH, y=AGE)
- Mortality rate by comorbidity bar chart
- Comorbidity co-occurrence heatmap (among deceased patients)
- Feature correlation matrix heatmap

Each plot needs a 2-sentence caption explaining the insight.

### Tab 3 — Model Performance
- Model comparison summary table (all 4 models, all 5 metrics)
- Bar chart comparing F1 scores across models
- Bar chart comparing AUC-ROC scores across models
- ROC curves for each model (can be combined on one plot or individual)
- Best hyperparameters for each model displayed clearly
- Confusion matrices for each model

### Tab 4 — Explainability and Interactive Prediction
This is the most important tab — it has 8 of the 20 Streamlit points.

**SHAP plots section:**
- SHAP summary plot (beeswarm)
- SHAP bar plot (mean absolute values)

**Interactive prediction section:**
- User selects which model to use (selectbox: Decision Tree, Random Forest, LightGBM, Neural Network)
- Sliders/inputs for the following key features:
  - AGE (slider, 0-100)
  - HOSPITALIZED (selectbox: Yes/No)
  - PNEUMONIA (selectbox: Yes/No)
  - COVID_POSITIVE (selectbox: Yes/No)
  - DIABETES (selectbox: Yes/No)
  - HYPERTENSION (selectbox: Yes/No)
  - RENAL_CHRONIC (selectbox: Yes/No)
  - Use mean values for all other features automatically
- Display: predicted class (Survived / Died) and probability
- Display: SHAP waterfall plot for the user's custom input (LightGBM only)

---

## STREAMLIT APP CODE STRUCTURE (app.py)

Structure the app.py file like this:

```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import lightgbm as lgb
from tensorflow import keras
from sklearn.metrics import (roc_curve, roc_auc_score, confusion_matrix,
                             accuracy_score, precision_score, recall_score, f1_score)

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="COVID-19 Mortality Prediction", layout="wide")

# ── Load models ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    dt = joblib.load('models/decision_tree.pkl')
    rf = joblib.load('models/random_forest.pkl')
    lgbm = joblib.load('models/lightgbm_model.pkl')
    nn = keras.models.load_model('models/neural_network.keras')
    scaler = joblib.load('models/scaler.pkl')
    return dt, rf, lgbm, nn, scaler

# ── Load data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    data = pd.read_csv('data/covid.csv', usecols=lambda col: col not in ['Unnamed: 0'])
    death_1 = data[data['DEATH'] == 1].sample(n=5000, random_state=42)
    death_0 = data[data['DEATH'] == 0].sample(n=5000, random_state=42)
    df = pd.concat([death_1, death_0])
    return df

dt, rf, lgbm, nn, scaler = load_models()
df = load_data()

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Summary",
    "Descriptive Analytics",
    "Model Performance",
    "Explainability & Interactive Prediction"
])

with tab1:
    # Executive summary content here
    pass

with tab2:
    # Recreate all Part 1 visualizations here
    pass

with tab3:
    # Model comparison table, ROC curves, confusion matrices here
    pass

with tab4:
    # SHAP plots + interactive prediction widget here
    pass
```

---

## INTERACTIVE PREDICTION LOGIC

When the user inputs feature values, build the input row like this:

```python
# Get mean values for all features
feature_means = df.drop(columns='DEATH').mean()

# Override with user inputs
input_data = feature_means.copy()
input_data['AGE'] = age_input
input_data['HOSPITALIZED'] = 1 if hospitalized == 'Yes' else 0
input_data['PNEUMONIA'] = 1 if pneumonia == 'Yes' else 0
input_data['COVID_POSITIVE'] = 1 if covid_positive == 'Yes' else 0
input_data['DIABETES'] = 1 if diabetes == 'Yes' else 0
input_data['HYPERTENSION'] = 1 if hypertension == 'Yes' else 0
input_data['RENAL_CHRONIC'] = 1 if renal_chronic == 'Yes' else 0

input_df = pd.DataFrame([input_data])

# For Neural Network, scale the input
input_scaled = scaler.transform(input_df)

# Get prediction based on selected model
if selected_model == 'Random Forest':
    prob = rf.predict_proba(input_df)[0][1]
elif selected_model == 'Decision Tree':
    prob = dt.predict_proba(input_df)[0][1]
elif selected_model == 'LightGBM':
    prob = lgbm.predict_proba(input_df)[0][1]
elif selected_model == 'Neural Network':
    prob = nn.predict(input_scaled).flatten()[0]

prediction = 'Died' if prob >= 0.5 else 'Survived'
```

---

## SHAP WATERFALL PLOT IN STREAMLIT

SHAP plots don't render natively in Streamlit — use this pattern:

```python
explainer = shap.TreeExplainer(lgbm)
shap_values = explainer.shap_values(input_df)
shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values

fig, ax = plt.subplots()
shap.plots.waterfall(
    shap.Explanation(
        values=shap_vals[0],
        base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
        data=input_df.iloc[0],
        feature_names=input_df.columns.tolist()
    ),
    show=False
)
st.pyplot(fig)
plt.clf()
```

---

## REQUIREMENTS.TXT

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
shap>=0.44.0
tensorflow>=2.15.0
joblib>=1.3.0
```

---

## README.MD CONTENT

The README should include:

```markdown
# MSIS 522 HW1 — COVID-19 Mortality Prediction

## Overview
End-to-end ML pipeline predicting COVID-19 patient mortality using Decision Trees,
Random Forest, LightGBM, and a Neural Network. Includes SHAP explainability and
a deployed Streamlit app.

## Live App
[Link to deployed Streamlit app — add after deployment]

## Project Structure
[Paste folder structure from above]

## How to Run Locally
1. Clone the repo
2. Install dependencies: pip install -r requirements.txt
3. Download the dataset:
   - Run the Colab notebook OR
   - Download manually from: https://drive.google.com/uc?id=1R-GDTtX0l38JYlPaG7f8eKx3D6pN-CKE
   - Place as data/covid.csv
4. Ensure model files are in models/ folder
5. Run: streamlit run app.py

## Dataset
~1M anonymized COVID-19 patient records. Balanced to 10,000 for training.
Source: Instructor-provided via Google Drive.

## Models
- Decision Tree (best: max_depth=4, min_samples_leaf=50)
- Random Forest (best: max_depth=8, n_estimators=200) ← Best overall
- LightGBM (best: learning_rate=0.05, max_depth=4, n_estimators=50)
- Neural Network (2 hidden layers, 128 units each, ReLU, 5 epochs)

## Key Findings
- Random Forest achieved the best AUC-ROC (0.9505) and F1 (0.9053)
- HOSPITALIZED is the single strongest predictor of mortality per SHAP analysis
- Optimal classification threshold for LightGBM is 0.39, not the default 0.50
```

---

## DEPLOYMENT TO STREAMLIT COMMUNITY CLOUD

1. Push the completed project to a public GitHub repo
2. Go to share.streamlit.io and sign in with GitHub
3. Click "New app" and select your repo
4. Set main file path to: app.py
5. Click Deploy — the app will be live at a public URL within a few minutes
6. Test by opening the URL in an incognito window before submitting

NOTE: The data/covid.csv file is large (~100MB). Options:
- Add to .gitignore and load from Google Drive URL directly in app.py
- Use st.cache_data with gdown to download on first load:

```python
import gdown
import os

@st.cache_data
def load_data():
    if not os.path.exists('data/covid.csv'):
        os.makedirs('data', exist_ok=True)
        gdown.download(
            "https://drive.google.com/uc?id=1R-GDTtX0l38JYlPaG7f8eKx3D6pN-CKE",
            "data/covid.csv",
            quiet=False
        )
    data = pd.read_csv('data/covid.csv', usecols=lambda col: col not in ['Unnamed: 0'])
    death_1 = data[data['DEATH'] == 1].sample(n=5000, random_state=42)
    death_0 = data[data['DEATH'] == 0].sample(n=5000, random_state=42)
    return pd.concat([death_1, death_0])
```

---

## IMPORTANT NOTES FOR CURSOR

1. All models use random_state=42 — keep this consistent everywhere
2. Neural Network requires StandardScaler — always scale input before predicting
3. SHAP plots must use matplotlib figures wrapped in st.pyplot() — they do not render natively
4. The app loads pre-trained models — it does NOT retrain on the fly
5. For Tab 4 interactive prediction, use average values for the 9 features not shown as sliders
6. Feature columns (in order): SEX, HOSPITALIZED, PNEUMONIA, AGE, PREGNANT, DIABETES,
   COPD, ASTHMA, IMMUNOSUPPRESSION, HYPERTENSION, OTHER_DISEASE, CARDIOVASCULAR,
   OBESITY, RENAL_CHRONIC, TOBACCO, COVID_POSITIVE
7. DEATH column is the target — never include it in X
8. The professor will check that the Streamlit app is publicly accessible — do not submit localhost
9. Tab 4 is worth 8/20 points — prioritize making the interactive prediction clean and functional
10. Executive summary in Tab 1 must not read as AI-generated — write it in a professional but
    natural voice suitable for healthcare executives
