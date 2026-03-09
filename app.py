import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import gdown
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

st.set_page_config(page_title="COVID-19 Mortality Prediction", layout="wide")
st.title("COVID-19 Mortality Prediction — MSIS 522 HW1")
st.caption("Alex Salcedo | University of Washington, Foster School of Business")

COMORBIDITY_LABELS = {
    "DIABETES": "Diabetes",
    "HYPERTENSION": "Hypertension",
    "OBESITY": "Obesity",
    "COPD": "COPD",
    "CARDIOVASCULAR": "Cardiovascular",
    "RENAL_CHRONIC": "Chronic Kidney Disease",
}


# ── Data loading ──────────────────────────────────────────────
@st.cache_data
def load_data():
    path = "data/covid.csv"
    if not os.path.exists(path):
        os.makedirs("data", exist_ok=True)
        gdown.download(
            "https://drive.google.com/uc?id=1R-GDTtX0l38JYlPaG7f8eKx3D6pN-CKE",
            path, quiet=False,
        )
    data = pd.read_csv(path, usecols=lambda c: c != "Unnamed: 0")
    death_1 = data[data["DEATH"] == 1].sample(n=5000, random_state=42)
    death_0 = data[data["DEATH"] == 0].sample(n=5000, random_state=42)
    return pd.concat([death_1, death_0])


@st.cache_resource
def load_models():
    dt = joblib.load("models/decision_tree.pkl")
    rf = joblib.load("models/random_forest.pkl")
    lgbm = joblib.load("models/lightgbm_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return dt, rf, lgbm, scaler


@st.cache_resource
def train_logistic(_train_x_scaled, _train_y):
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(_train_x_scaled, _train_y)
    return lr


df = load_data()
dt_model, rf_model, lgbm_model, scaler = load_models()

X = df.drop(columns="DEATH")
y = df["DEATH"]
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)
train_x_scaled = scaler.transform(train_x)
test_x_scaled = scaler.transform(test_x)

lr_model = train_logistic(train_x_scaled, train_y)

FEATURES = X.columns.tolist()


def get_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "AUC-ROC": roc_auc_score(y_true, y_proba),
    }


# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Summary",
    "Descriptive Analytics",
    "Model Performance",
    "Explainability & Interactive Prediction",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — Executive Summary
# ══════════════════════════════════════════════════════════════
with tab1:
    st.header("Executive Summary")

    st.subheader("Dataset & Prediction Task")
    st.markdown("""
This analysis uses a dataset of approximately **one million anonymized COVID-19 patient
records** originally compiled by Mexico's Secretariat of Health (*Secretaría de Salud /
Dirección General de Epidemiología*) as part of its national epidemiological surveillance
system. The dataset was provided by the course instructor via Google Drive for use in
this assignment.

Each patient record contains **16 features** spanning three categories:

- **Demographics:** age and sex.
- **Clinical status:** whether the patient was hospitalized, diagnosed with pneumonia,
  and whether they tested positive for COVID-19.
- **Pre-existing conditions:** diabetes, hypertension, obesity, COPD (chronic
  obstructive pulmonary disease), asthma, immunosuppression, cardiovascular disease,
  chronic kidney disease, tobacco use, pregnancy, and other diseases.

The **prediction target** is patient mortality — specifically, a binary variable
indicating whether the patient **survived or died**. Because the raw dataset is heavily
imbalanced (far more survivors than deaths), the data was balanced to 10,000 records
(5,000 deaths and 5,000 survivors) before modeling to ensure the models learn from
both outcomes equally.
    """)

    st.subheader("Why This Matters")
    st.markdown("""
In a clinical setting, the ability to accurately identify which patients are most
likely to die is not just a technical exercise — it is a tool for **resource allocation
and triage**. During pandemic surges, hospitals must decide which patients receive ICU
beds, ventilators, and specialist attention. A reliable risk model can help clinicians
identify the most vulnerable patients early, enabling proactive interventions that
improve outcomes and reduce strain on overwhelmed healthcare systems.

Beyond individual patient care, understanding *which* pre-existing conditions drive
mortality risk informs **public health messaging and vaccination prioritization
strategies**. If chronic kidney disease and COPD are shown to dramatically increase
mortality risk, public health agencies can target those populations for early
intervention.
    """)

    st.subheader("Approach & Key Findings")
    st.markdown("""
Five different machine learning models were trained and compared: **Logistic Regression**
(a simple baseline), **Decision Tree**, **Random Forest**, **LightGBM** (a gradient
boosting method), and a **Neural Network**. Each model was tuned using cross-validation
(testing multiple configurations to find the best settings) and then evaluated on a
held-out 30% portion of the data that the models never saw during training.

All five models performed strongly. The key metrics used to evaluate them were:

- **AUC-ROC** (how well the model distinguishes survivors from non-survivors overall):
  ranged from 0.94 to 0.95 across all models.
- **F1 Score** (a balanced measure of how accurately the model identifies deaths without
  too many false alarms): ranged from 0.90 to 0.91.

**Random Forest** achieved the highest overall discrimination (AUC-ROC of 0.9505), while
the **Neural Network** was best at catching actual deaths (recall of 0.9460). The small
performance gap across models suggests that the key patient characteristics —
particularly **hospitalization status, age, and pneumonia** — carry most of the
predictive signal regardless of which algorithm is used.

A feature importance analysis (SHAP) confirmed that **hospitalization status** is by far
the strongest predictor of mortality, followed by **age** and **pneumonia**. The
recommended model for deployment is **Random Forest**, which balances strong predictive
performance with the ability to explain individual predictions to clinicians.
    """)


# ══════════════════════════════════════════════════════════════
# TAB 2 — Descriptive Analytics
# ══════════════════════════════════════════════════════════════
with tab2:
    st.header("Descriptive Analytics")
    st.markdown("Visual exploration of the dataset to understand patient demographics, "
                "health conditions, and their relationship to mortality before building "
                "predictive models.")

    # 0) Target distribution
    st.subheader("Target Distribution: Survived vs. Died")
    fig0, ax0 = plt.subplots(figsize=(7, 5))
    counts = df["DEATH"].value_counts().sort_index()
    bars = ax0.bar(["Survived", "Died"], [counts[0], counts[1]],
                   color=["steelblue", "indianred"])
    for bar, val in zip(bars, [counts[0], counts[1]]):
        ax0.text(bar.get_x() + bar.get_width() / 2, val + 50,
                 str(val), ha="center", fontsize=12, fontweight="bold")
    ax0.set_title("Patient Outcome Distribution", fontsize=16)
    ax0.set_ylabel("Number of Patients")
    fig0.tight_layout()
    st.pyplot(fig0)
    plt.close(fig0)
    st.markdown("""
The dataset was intentionally balanced to contain an equal number of patients who
survived (5,000) and patients who died (5,000). This balancing step prevents the models
from being biased toward simply predicting "survived" for every patient, which would
happen if the original imbalanced distribution were used.
    """)

    st.divider()

    # 1) Age histogram
    st.subheader("Age Distribution of Patients")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.histplot(df["AGE"], bins=30, kde=True, color="steelblue", ax=ax1)
    ax1.set_title("Age Distribution of Patients", fontsize=16)
    ax1.set_xlabel("Age")
    ax1.set_ylabel("Count")
    fig1.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)
    st.markdown("""
The age distribution is roughly bell-shaped and centered around 45-65 years old, with
relatively few patients under 20. The smooth curve confirms a slight right skew, meaning
older patients are well-represented — an important consideration given that age is one
of the strongest known risk factors for COVID-19 mortality.
    """)

    st.divider()

    # 2) Boxplot
    st.subheader("Age Distribution by Patient Outcome")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.boxplot(x="DEATH", y="AGE", data=df, palette="Set2", ax=ax2)
    ax2.set_xticklabels(["Survived", "Died"])
    ax2.set_title("Age Distribution by Patient Outcome", fontsize=16)
    ax2.set_xlabel("Outcome")
    ax2.set_ylabel("Age")
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)
    st.markdown("""
Patients who died show a noticeably higher median age and a wider spread compared to
survivors, confirming that older patients faced significantly greater mortality risk.
The overlap between the two groups also suggests age alone is not sufficient for
prediction, which motivates the use of additional clinical features in the models.
    """)

    st.divider()

    # 3) Mortality rate by comorbidity
    st.subheader("Mortality Rate by Pre-existing Condition")
    comorbidities = list(COMORBIDITY_LABELS.keys())
    comorb_labels = list(COMORBIDITY_LABELS.values())
    mortality_rates = [df[df[c] == 1]["DEATH"].mean() * 100 for c in comorbidities]
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=comorb_labels, y=mortality_rates, palette="Reds_d", ax=ax3)
    ax3.set_title("Mortality Rate by Pre-existing Condition (%)", fontsize=16)
    ax3.set_xlabel("Condition")
    ax3.set_ylabel("Mortality Rate (%)")
    ax3.tick_params(axis="x", rotation=15)
    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)
    st.markdown("""
COPD and chronic kidney disease are associated with the highest mortality rates among
all pre-existing conditions examined, suggesting these conditions significantly
compromise a patient's ability to recover. Clinicians may want to flag patients
presenting with either condition for early intervention and closer monitoring.
    """)

    st.divider()

    # 4) Comorbidity co-occurrence heatmap
    st.subheader("Pre-existing Condition Co-occurrence Among Deceased Patients")
    deceased = df[df["DEATH"] == 1][comorbidities]
    co_occurrence = deceased.T.dot(deceased)
    co_occurrence.index = comorb_labels
    co_occurrence.columns = comorb_labels
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    sns.heatmap(co_occurrence, annot=True, fmt="d", cmap="YlOrRd", linewidths=0.5,
                ax=ax4, annot_kws={"size": 11})
    ax4.set_title("Condition Co-occurrence Among Deceased Patients", fontsize=16)
    ax4.tick_params(axis="x", rotation=45, labelsize=11)
    ax4.tick_params(axis="y", rotation=0, labelsize=11)
    fig4.tight_layout(pad=2.0)
    st.pyplot(fig4)
    plt.close(fig4)
    st.markdown("""
Among deceased patients, hypertension and diabetes show the highest co-occurrence,
indicating that patients carrying both conditions represent a particularly high-risk
group. This pattern suggests that having multiple pre-existing conditions together may
be more predictive of mortality than any single condition in isolation.
    """)

    st.divider()

    # 5) Correlation heatmap
    st.subheader("Feature Correlation Matrix")
    fig5, ax5 = plt.subplots(figsize=(12, 9))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5,
                annot_kws={"size": 8}, ax=ax5)
    ax5.set_title("Feature Correlation Matrix", fontsize=16)
    fig5.tight_layout()
    st.pyplot(fig5)
    plt.close(fig5)
    st.markdown("""
Hospitalization status shows the strongest positive correlation with mortality (0.78),
followed by pneumonia (0.62) and age (0.56). Most pre-existing condition features show
weak to moderate correlations with each other, which means each feature contributes
relatively independent information to the predictive models.
    """)


# ══════════════════════════════════════════════════════════════
# TAB 3 — Model Performance
# ══════════════════════════════════════════════════════════════
with tab3:
    st.header("Model Performance Comparison")
    st.markdown("All models were trained on 70% of the data and evaluated on a "
                "held-out 30% test set that the models never saw during training.")

    model_dict = {
        "Logistic Regression": (lr_model, test_x_scaled),
        "Decision Tree": (dt_model, test_x),
        "Random Forest": (rf_model, test_x),
        "LightGBM": (lgbm_model, test_x),
    }

    all_metrics = {}
    roc_data = {}
    cm_data = {}

    for name, (mdl, x_input) in model_dict.items():
        pred = mdl.predict(x_input)
        proba = mdl.predict_proba(x_input)[:, 1]
        all_metrics[name] = get_metrics(test_y, pred, proba)
        fpr, tpr, _ = roc_curve(test_y, proba)
        roc_data[name] = (fpr, tpr, all_metrics[name]["AUC-ROC"])
        cm_data[name] = confusion_matrix(test_y, pred)

    all_metrics["Neural Network"] = {
        "Accuracy": 0.8977, "Precision": 0.8665, "Recall": 0.9460,
        "F1 Score": 0.9045, "AUC-ROC": 0.9448,
    }

    # Metrics table
    st.subheader("Test Set Metrics")
    st.markdown("Each metric captures a different aspect of model quality. "
                "**Accuracy** is the overall correctness rate. **Precision** measures "
                "how many predicted deaths were actual deaths. **Recall** measures how "
                "many actual deaths the model caught. **F1 Score** balances precision "
                "and recall. **AUC-ROC** measures overall ability to distinguish "
                "survivors from non-survivors.")
    metrics_df = pd.DataFrame(all_metrics).T.round(4)
    st.dataframe(metrics_df, use_container_width=True)

    st.divider()

    # Bar charts
    st.subheader("F1 Score and AUC-ROC Comparison")
    colors = ["crimson", "steelblue", "forestgreen", "darkorange", "purple"]
    model_names = list(all_metrics.keys())

    fig_bar, (ax_f1, ax_auc) = plt.subplots(1, 2, figsize=(13, 5))

    f1_vals = [all_metrics[m]["F1 Score"] for m in model_names]
    ax_f1.bar(model_names, f1_vals, color=colors[:len(model_names)])
    ax_f1.set_title("F1 Score by Model", fontsize=14)
    ax_f1.set_ylabel("F1 Score")
    ax_f1.set_ylim(min(f1_vals) - 0.01, max(f1_vals) + 0.01)
    for i, v in enumerate(f1_vals):
        ax_f1.text(i, v + 0.0003, f"{v:.4f}", ha="center", fontsize=9)
    ax_f1.tick_params(axis="x", rotation=20)

    auc_vals = [all_metrics[m]["AUC-ROC"] for m in model_names]
    ax_auc.bar(model_names, auc_vals, color=colors[:len(model_names)])
    ax_auc.set_title("AUC-ROC by Model", fontsize=14)
    ax_auc.set_ylabel("AUC-ROC")
    ax_auc.set_ylim(min(auc_vals) - 0.01, max(auc_vals) + 0.01)
    for i, v in enumerate(auc_vals):
        ax_auc.text(i, v + 0.0003, f"{v:.4f}", ha="center", fontsize=9)
    ax_auc.tick_params(axis="x", rotation=20)

    fig_bar.tight_layout()
    st.pyplot(fig_bar)
    plt.close(fig_bar)

    st.divider()

    # ROC curves
    st.subheader("ROC Curves")
    st.markdown("The ROC curve shows how well each model separates survivors from "
                "non-survivors across all possible decision thresholds. A curve closer "
                "to the top-left corner indicates better performance.")
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    line_colors = {"Logistic Regression": "crimson", "Decision Tree": "steelblue",
                   "Random Forest": "forestgreen", "LightGBM": "darkorange",
                   "Neural Network": "purple"}
    for name, (fpr, tpr, auc_val) in roc_data.items():
        ax_roc.plot(fpr, tpr, lw=2, color=line_colors.get(name, "black"),
                    label=f"{name} (AUC = {auc_val:.4f})")
    ax_roc.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Baseline")
    ax_roc.set_title("ROC Curves — All Models", fontsize=14)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right")
    fig_roc.tight_layout()
    st.pyplot(fig_roc)
    plt.close(fig_roc)

    st.divider()

    # Best hyperparameters
    st.subheader("Best Hyperparameters")
    st.markdown("Each model was tuned by testing different configurations and selecting "
                "the one that performed best during cross-validation.")
    hp_data = {
        "Logistic Regression": "max_iter=1000 (baseline)",
        "Decision Tree": "max_depth=4, min_samples_leaf=50",
        "Random Forest": "max_depth=8, n_estimators=200",
        "LightGBM": "learning_rate=0.05, max_depth=4, n_estimators=50",
        "Neural Network": "2 hidden layers (128 units each), ReLU, Adam, 5 epochs",
    }
    hp_df = pd.DataFrame(
        {"Model": hp_data.keys(), "Best Hyperparameters": hp_data.values()}
    )
    st.table(hp_df)

    st.divider()

    # Confusion matrices
    st.subheader("Confusion Matrices")
    st.markdown("Each matrix shows how many patients the model correctly and incorrectly "
                "classified. The top-left and bottom-right cells are correct predictions; "
                "the other two cells are errors.")
    n_models = len(cm_data)
    fig_cm, axes_cm = plt.subplots(1, n_models, figsize=(4 * n_models, 4))
    if n_models == 1:
        axes_cm = [axes_cm]
    for ax_cm, (name, cm) in zip(axes_cm, cm_data.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
                    xticklabels=["Survived", "Died"], yticklabels=["Survived", "Died"])
        ax_cm.set_title(name, fontsize=11)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
    fig_cm.tight_layout()
    st.pyplot(fig_cm)
    plt.close(fig_cm)


# ══════════════════════════════════════════════════════════════
# TAB 4 — Explainability & Interactive Prediction
# ══════════════════════════════════════════════════════════════
with tab4:
    st.header("Explainability & Interactive Prediction")

    # ── SHAP explainer (cached, needed by both sections) ──────
    @st.cache_resource
    def get_shap_explainer(_model, _test_x):
        explainer = shap.TreeExplainer(_model)
        sv = explainer.shap_values(_test_x)
        vals = sv[1] if isinstance(sv, list) else sv
        return explainer, vals

    explainer, shap_vals = get_shap_explainer(lgbm_model, test_x)

    # ── Interactive prediction (FIRST) ────────────────────────
    st.subheader("Try It Yourself: Predict Patient Mortality Risk")
    st.markdown("""
Use the controls below to create a patient profile and see what the model predicts in
real time. Adjust the patient's age, hospitalization status, and pre-existing conditions
to explore how different risk factors affect the predicted outcome. Features not shown
below are held at their dataset average values.
    """)

    available_models = ["Decision Tree", "Random Forest", "LightGBM"]
    selected_model = st.selectbox("Choose a prediction model:", available_models)

    col1, col2, col3 = st.columns(3)
    with col1:
        age_input = st.slider("Patient Age", 0, 100, 50)
        hospitalized = st.selectbox("Hospitalized?", ["No", "Yes"])
        diabetes = st.selectbox("Diabetes?", ["No", "Yes"])
    with col2:
        pneumonia = st.selectbox("Pneumonia?", ["No", "Yes"])
        covid_positive = st.selectbox("COVID-19 Positive?", ["No", "Yes"])
        hypertension = st.selectbox("Hypertension?", ["No", "Yes"])
    with col3:
        renal_chronic = st.selectbox("Chronic Kidney Disease?", ["No", "Yes"])

    feature_means = X.mean()
    input_data = feature_means.copy()
    input_data["AGE"] = age_input
    input_data["HOSPITALIZED"] = 1 if hospitalized == "Yes" else 0
    input_data["PNEUMONIA"] = 1 if pneumonia == "Yes" else 0
    input_data["COVID_POSITIVE"] = 1 if covid_positive == "Yes" else 0
    input_data["DIABETES"] = 1 if diabetes == "Yes" else 0
    input_data["HYPERTENSION"] = 1 if hypertension == "Yes" else 0
    input_data["RENAL_CHRONIC"] = 1 if renal_chronic == "Yes" else 0

    input_df = pd.DataFrame([input_data])[FEATURES]
    input_scaled = scaler.transform(input_df)

    if selected_model == "Decision Tree":
        prob = dt_model.predict_proba(input_df)[0][1]
    elif selected_model == "Random Forest":
        prob = rf_model.predict_proba(input_df)[0][1]
    elif selected_model == "LightGBM":
        prob = lgbm_model.predict_proba(input_df)[0][1]

    prediction = "Died" if prob >= 0.5 else "Survived"

    st.divider()

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric("Predicted Outcome", prediction)
    with res_col2:
        st.metric("Mortality Probability", f"{prob:.2%}")

    # SHAP waterfall for the custom input (LightGBM only)
    if selected_model == "LightGBM":
        st.markdown("**What is driving this prediction?** The waterfall plot below "
                    "shows how each patient feature pushes the prediction toward "
                    "'Survived' (blue/left) or 'Died' (red/right).")
        sv_input = explainer.shap_values(input_df)
        sv_input_vals = sv_input[1] if isinstance(sv_input, list) else sv_input
        base_val = (explainer.expected_value[1]
                    if isinstance(explainer.expected_value, list)
                    else explainer.expected_value)

        fig_wf, ax_wf = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(
            shap.Explanation(
                values=sv_input_vals[0],
                base_values=base_val,
                data=input_df.iloc[0],
                feature_names=input_df.columns.tolist(),
            ),
            show=False,
        )
        st.pyplot(fig_wf)
        plt.close("all")
    else:
        st.info("Select **LightGBM** to see a detailed breakdown of what is driving "
                "the prediction for this patient.")

    st.divider()

    # ── SHAP global analysis (SECOND) ─────────────────────────
    st.subheader("Feature Importance: What Drives the Model's Predictions?")
    st.markdown("""
The charts below use **SHAP analysis** to show which patient characteristics have the
greatest influence on the model's predictions across all patients in the test set.
The left plot shows how each feature pushes individual predictions toward survival
(blue) or death (red). The right plot ranks features by their average importance.
    """)

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("**Impact on Individual Predictions (Beeswarm)**")
        fig_shap1, ax_shap1 = plt.subplots(figsize=(8, 6))
        shap.summary_plot(shap_vals, test_x, show=False)
        st.pyplot(fig_shap1)
        plt.close("all")

    with col_s2:
        st.markdown("**Average Feature Importance**")
        fig_shap2, ax_shap2 = plt.subplots(figsize=(8, 6))
        shap.summary_plot(shap_vals, test_x, plot_type="bar", show=False)
        st.pyplot(fig_shap2)
        plt.close("all")

    st.markdown("""
**Hospitalization status** is by far the most influential factor — patients who were
hospitalized show dramatically higher predicted mortality risk. **Age** is the second
most important feature, with older patients receiving higher risk scores. **Pneumonia**
ranks third, and its presence substantially increases predicted mortality risk. These
three features together dominate the model's decision-making, while pre-existing
conditions like diabetes, hypertension, and chronic kidney disease play smaller but
meaningful supporting roles.
    """)
