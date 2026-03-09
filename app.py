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


# ── Helper: compute metrics for a model ──────────────────────
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
This analysis uses a dataset of approximately one million anonymized COVID-19 patient
records provided by the course instructor. Each record contains demographic information
(age, sex) and binary indicators for pre-existing conditions such as diabetes,
hypertension, COPD, obesity, and chronic renal disease, along with hospitalization
status and COVID-19 test results. The prediction target is **DEATH** — a binary
variable indicating whether the patient survived or died. To handle the severe class
imbalance in the raw data, the dataset was balanced to 10,000 records (5,000 deaths
and 5,000 survivors) before modeling.
    """)

    st.subheader("Why This Matters")
    st.markdown("""
In a clinical setting, the ability to accurately stratify patient mortality risk is
not just a technical exercise — it is a tool for resource allocation and triage.
During pandemic surges, hospitals must decide which patients receive ICU beds,
ventilators, and specialist attention. A reliable risk model can help clinicians
identify the most vulnerable patients early, enabling proactive interventions that
improve outcomes and reduce strain on overwhelmed healthcare systems. Understanding
*which* comorbidities drive mortality risk also informs public health messaging and
vaccination prioritization strategies.
    """)

    st.subheader("Approach & Key Findings")
    st.markdown("""
Five models were trained and evaluated: Logistic Regression (baseline), Decision Tree,
Random Forest, LightGBM, and a Neural Network (MLP). All models were tuned via 5-fold
cross-validation with `random_state=42` and evaluated on a held-out 30% test set.

All models performed competitively, with AUC-ROC scores ranging from 0.9448 to 0.9505
and F1 scores between 0.9024 and 0.9045. **Random Forest** achieved the highest AUC-ROC
(0.9505), while the **Neural Network** led on recall (0.9460). The narrow performance
gap across models suggests that the feature set itself — particularly hospitalization
status, age, and pneumonia — carries most of the predictive signal.

SHAP analysis on the LightGBM model confirmed that **HOSPITALIZED** is by far the
strongest predictor, followed by **AGE** and **PNEUMONIA**. The recommended model for
deployment is Random Forest, which balances strong discrimination with SHAP-based
interpretability that clinicians can audit at the individual patient level.
    """)


# ══════════════════════════════════════════════════════════════
# TAB 2 — Descriptive Analytics
# ══════════════════════════════════════════════════════════════
with tab2:
    st.header("Descriptive Analytics")

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
The age distribution is roughly bell-shaped and centered around 45–65 years old, with
relatively few patients under 20. The KDE curve confirms a slight right skew, meaning
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
Patients who died show a noticeably higher median age and a wider interquartile range
compared to survivors, confirming that older patients faced significantly greater
mortality risk. The overlap between the two groups also suggests age alone is not
sufficient for prediction, which motivates the use of additional clinical features.
    """)

    st.divider()

    # 3) Mortality rate by comorbidity
    st.subheader("Mortality Rate by Comorbidity")
    comorbidities = ["DIABETES", "HYPERTENSION", "OBESITY", "COPD", "CARDIOVASCULAR", "RENAL_CHRONIC"]
    mortality_rates = [df[df[c] == 1]["DEATH"].mean() * 100 for c in comorbidities]
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=comorbidities, y=mortality_rates, palette="Reds_d", ax=ax3)
    ax3.set_title("Mortality Rate by Comorbidity (%)", fontsize=16)
    ax3.set_xlabel("Comorbidity")
    ax3.set_ylabel("Mortality Rate (%)")
    ax3.tick_params(axis="x", rotation=15)
    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)
    st.markdown("""
COPD and chronic renal disease are associated with the highest mortality rates among all
comorbidities, suggesting these conditions significantly compromise a patient's ability
to recover. Clinicians may want to flag patients presenting with either condition for
early intervention and closer monitoring.
    """)

    st.divider()

    # 4) Comorbidity co-occurrence heatmap
    st.subheader("Comorbidity Co-occurrence Among Deceased Patients")
    deceased = df[df["DEATH"] == 1][comorbidities]
    co_occurrence = deceased.T.dot(deceased)
    fig4, ax4 = plt.subplots(figsize=(9, 7))
    sns.heatmap(co_occurrence, annot=True, fmt="d", cmap="YlOrRd", linewidths=0.5, ax=ax4)
    ax4.set_title("Comorbidity Co-occurrence Among Deceased Patients", fontsize=16)
    fig4.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)
    st.markdown("""
Among deceased patients, hypertension and diabetes show the highest co-occurrence,
indicating that patients carrying both conditions represent a particularly high-risk
group. This pattern suggests that multi-comorbidity profiles may be more predictive of
mortality than any single condition in isolation.
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
followed by pneumonia (0.62) and age (0.56). Most comorbidity features show weak to
moderate correlations with each other, which reduces multicollinearity concerns and
suggests each feature contributes relatively independent information to the models.
    """)


# ══════════════════════════════════════════════════════════════
# TAB 3 — Model Performance
# ══════════════════════════════════════════════════════════════
with tab3:
    st.header("Model Performance Comparison")

    # Compute predictions for sklearn/lgbm models
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

    # NN metrics from notebook (TensorFlow too large for Streamlit Cloud)
    all_metrics["Neural Network"] = {
        "Accuracy": 0.8977, "Precision": 0.8665, "Recall": 0.9460,
        "F1 Score": 0.9045, "AUC-ROC": 0.9448,
    }

    # Metrics table
    st.subheader("Test Set Metrics")
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

    # ── SHAP plots ────────────────────────────────────────────
    st.subheader("SHAP Analysis (LightGBM)")

    @st.cache_resource
    def get_shap_explainer(_model, _test_x):
        explainer = shap.TreeExplainer(_model)
        sv = explainer.shap_values(_test_x)
        vals = sv[1] if isinstance(sv, list) else sv
        return explainer, vals

    explainer, shap_vals = get_shap_explainer(lgbm_model, test_x)

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("**Summary Plot (Beeswarm)**")
        fig_shap1, ax_shap1 = plt.subplots(figsize=(8, 6))
        shap.summary_plot(shap_vals, test_x, show=False)
        st.pyplot(fig_shap1)
        plt.close("all")

    with col_s2:
        st.markdown("**Feature Importance (Mean |SHAP|)**")
        fig_shap2, ax_shap2 = plt.subplots(figsize=(8, 6))
        shap.summary_plot(shap_vals, test_x, plot_type="bar", show=False)
        st.pyplot(fig_shap2)
        plt.close("all")

    st.markdown("""
**HOSPITALIZED** is by far the most influential feature — patients who were hospitalized
show large positive SHAP values, strongly pushing the prediction toward death. **AGE**
is the second most important feature, with older patients showing higher SHAP values.
**PNEUMONIA** ranks third, and its presence substantially increases predicted mortality
risk. These three features together dominate the model's decision-making.
    """)

    st.divider()

    # ── Interactive prediction ────────────────────────────────
    st.subheader("Interactive Prediction")

    available_models = ["Decision Tree", "Random Forest", "LightGBM"]
    selected_model = st.selectbox("Select a model for prediction:", available_models)

    col1, col2, col3 = st.columns(3)
    with col1:
        age_input = st.slider("AGE", 0, 100, 50)
        hospitalized = st.selectbox("HOSPITALIZED", ["No", "Yes"])
        diabetes = st.selectbox("DIABETES", ["No", "Yes"])
    with col2:
        pneumonia = st.selectbox("PNEUMONIA", ["No", "Yes"])
        covid_positive = st.selectbox("COVID_POSITIVE", ["No", "Yes"])
        hypertension = st.selectbox("HYPERTENSION", ["No", "Yes"])
    with col3:
        renal_chronic = st.selectbox("RENAL_CHRONIC", ["No", "Yes"])

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
        st.divider()
        st.markdown("**SHAP Waterfall Plot for This Prediction**")
        sv_input = explainer.shap_values(input_df)
        sv_input_vals = sv_input[1] if isinstance(sv_input, list) else sv_input
        base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value

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
        st.info("Select **LightGBM** to see a SHAP waterfall plot for this prediction.")
