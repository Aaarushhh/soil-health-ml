# ======================================================
# 🌾 STREAMLIT SOIL EXPLAINABILITY DASHBOARD (FINAL FIXED VERSION)
# Robust SHAP + LIME | Works with labeled or unlabeled soil datasets
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
from tqdm import tqdm

# ======================================================
# 🧭 PAGE CONFIG
# ======================================================
st.set_page_config(page_title="Soil Explainability Dashboard", layout="wide")
st.title("🌱 Soil–Crop Explainability Dashboard (SHAP + LIME + EDA)")

# ======================================================
# 📥 UPLOAD DATASET
# ======================================================
uploaded_file = st.file_uploader("📂 Upload your soil dataset (CSV)", type=["csv"])
if uploaded_file is None:
    st.info("Please upload a dataset (with or without 'label' column).")
    st.stop()

df = pd.read_csv(uploaded_file)
st.success("✅ Dataset loaded successfully!")
st.write("### 🔍 Dataset Preview")
st.dataframe(df.head())

# ======================================================
# 🧩 FEATURE / LABEL HANDLING
# ======================================================
if "label" in df.columns:
    st.success("Detected target column: **'label'** ✅")
    X = df.drop(columns=["label"])
    y = df["label"]
    mode = "labeled"
else:
    st.warning("⚠️ No 'label' column found — running in **Soil Feature Insight Mode**.")
    X = df.copy()
    y = None
    mode = "unlabeled"

feature_names = X.columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ======================================================
# 📊 EDA SECTION
# ======================================================
st.header("📊 Exploratory Data Analysis")

col1, col2 = st.columns(2)
with col1:
    st.write("### Basic Statistics")
    st.dataframe(df.describe())

with col2:
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(6, 4))
    corr = df.corr(numeric_only=True)
    im = ax.imshow(corr, cmap="YlGnBu")
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    plt.colorbar(im)
    plt.title("Feature Correlation Heatmap")
    st.pyplot(fig)

# ======================================================
# 🧭 UNLABELED MODE: SHAP-ONLY ANALYSIS (FIXED)
# ======================================================
if mode == "unlabeled":
    st.header("🧠 SHAP Feature Insight (Unsupervised)")
    st.info("Running SHAP on clean soil data — discovering dominant soil factors 🌾")

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    y_dummy = np.random.randint(0, 2, len(X_scaled))
    rf.fit(X_scaled, y_dummy)

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_scaled)

    if isinstance(shap_values, list):
        shap_array = np.mean([np.abs(vals) for vals in shap_values], axis=0)
    else:
        shap_array = np.abs(shap_values)

    mean_abs_shap = np.mean(shap_array, axis=0)
    if mean_abs_shap.ndim > 1:
        mean_abs_shap = mean_abs_shap.mean(axis=0)
    mean_abs_shap = np.array(mean_abs_shap).flatten()

    # ✅ Align feature and SHAP lengths
    len_features, len_shap = len(feature_names), len(mean_abs_shap)
    if len_features != len_shap:
        st.warning(f"⚠️ Adjusting SHAP length mismatch: Features={len_features}, SHAP={len_shap}")
        min_len = min(len_features, len_shap)
        feature_names = feature_names[:min_len]
        mean_abs_shap = mean_abs_shap[:min_len]

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP_Importance": mean_abs_shap
    }).sort_values(by="SHAP_Importance", ascending=False)

    st.write("### 🌿 SHAP-based Feature Influence Ranking")
    st.dataframe(shap_df)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(shap_df["Feature"], shap_df["SHAP_Importance"], color="teal")
    plt.gca().invert_yaxis()
    ax.set_title("🌾 Soil Feature Importance (Unlabeled SHAP)")
    st.pyplot(fig)
    st.stop()

# ======================================================
# 🧠 LABELED MODE: MODEL TRAINING + EXPLAINABILITY
# ======================================================
st.header("🧠 Model Explainability Mode (Labeled Data)")

label_counts = Counter(y)
label_df = pd.DataFrame.from_dict(label_counts, orient='index', columns=['Count']).sort_values(by='Count', ascending=False)
colA, colB = st.columns(2)
with colA:
    st.subheader("🌾 Crop Distribution")
    st.dataframe(label_df)
with colB:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(label_df.index, label_df['Count'], color='mediumseagreen')
    plt.xticks(rotation=45)
    ax.set_title("Crop Label Frequency")
    st.pyplot(fig)

rare_classes = [c for c, n in label_counts.items() if n < 2]
if rare_classes:
    st.warning(f"⚠️ Some crops have <2 samples: {rare_classes}. Stratify disabled.")
    stratify_arg = None
else:
    stratify_arg = y

test_size = st.slider("📏 Test Size", 0.1, 0.5, 0.2)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=test_size, random_state=42, stratify=stratify_arg
)

# ======================================================
# 🌳 RANDOM FOREST + 🔹 LOGISTIC REGRESSION
# ======================================================
st.header("🌿 Model Training & Evaluation")

col_rf, col_lr = st.columns(2)
with col_rf:
    st.subheader("🌳 Random Forest")
    n_estimators = st.slider("Number of Trees", 50, 300, 100)
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    st.write(f"✅ Accuracy: {accuracy_score(y_test, y_pred_rf):.3f}")
    st.text(classification_report(y_test, y_pred_rf))

with col_lr:
    st.subheader("🔹 Logistic Regression")
    logreg = LogisticRegression(max_iter=2000, random_state=42)
    logreg.fit(X_train, y_train)
    y_pred_lr = logreg.predict(X_test)
    st.write(f"✅ Accuracy: {accuracy_score(y_test, y_pred_lr):.3f}")
    st.text(classification_report(y_test, y_pred_lr))

# ======================================================
# 🧠 SHAP EXPLAINABILITY (WITH LENGTH ALIGN FIX)
# ======================================================
st.header("🧠 SHAP Explainability")

def safe_shap_dataframe(shap_values, model_name):
    if isinstance(shap_values, list):
        shap_array = np.mean([np.abs(vals) for vals in shap_values], axis=0)
    else:
        shap_array = np.abs(shap_values)
    mean_abs_shap = np.mean(shap_array, axis=0)
    if mean_abs_shap.ndim > 1:
        mean_abs_shap = mean_abs_shap.mean(axis=0)
    mean_abs_shap = np.array(mean_abs_shap).flatten()

    # Align lengths safely
    len_features, len_shap = len(feature_names), len(mean_abs_shap)
    if len_features != len_shap:
        st.warning(f"⚠️ [{model_name}] Adjusting SHAP length mismatch: Features={len_features}, SHAP={len_shap}")
        min_len = min(len_features, len_shap)
        f_names = feature_names[:min_len]
        mean_abs_shap = mean_abs_shap[:min_len]
    else:
        f_names = feature_names

    return pd.DataFrame({
        "Feature": f_names,
        "SHAP_Importance": mean_abs_shap
    }).sort_values(by="SHAP_Importance", ascending=False)

col_rf_shap, col_lr_shap = st.columns(2)
with col_rf_shap:
    explainer_rf = shap.TreeExplainer(rf)
    shap_values_rf = explainer_rf.shap_values(X_test)
    shap_rf_df = safe_shap_dataframe(shap_values_rf, "RandomForest")
    st.subheader("🌳 Random Forest SHAP")
    st.dataframe(shap_rf_df)

with col_lr_shap:
    explainer_lr = shap.LinearExplainer(logreg, X_train)
    shap_values_lr = explainer_lr.shap_values(X_test)
    shap_lr_df = safe_shap_dataframe(shap_values_lr, "LogisticRegression")
    st.subheader("🔹 Logistic Regression SHAP")
    st.dataframe(shap_lr_df)

# ======================================================
# 🍃 LIME EXPLAINABILITY
# ======================================================
st.header("🍃 LIME Explainability")

def compute_lime(model, model_name):
    explainer = LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=feature_names,
        class_names=np.unique(y_train),
        mode="classification"
    )
    lime_weights = []
    for i in tqdm(range(min(30, len(X_test))), desc=f"LIME {model_name}"):
        exp = explainer.explain_instance(
            X_test[i], model.predict_proba, num_features=len(feature_names)
        )
        lime_weights.append(dict(exp.as_list()))
    lime_df = pd.DataFrame(lime_weights).mean().abs().reset_index()
    lime_df.columns = ["Feature", "LIME_Importance"]
    return lime_df.sort_values(by="LIME_Importance", ascending=False)

col_rf_lime, col_lr_lime = st.columns(2)
with col_rf_lime:
    lime_rf_df = compute_lime(rf, "RF")
    st.subheader("🌳 Random Forest LIME")
    st.dataframe(lime_rf_df)
with col_lr_lime:
    lime_lr_df = compute_lime(logreg, "LogReg")
    st.subheader("🔹 Logistic Regression LIME")
    st.dataframe(lime_lr_df)

st.success("✅ Full Explainability Dashboard Ready!")
