# ===============================================================
# 🌾 Streamlit Dashboard — SHAP + LIME Interpretability Explorer
# ===============================================================

import os
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1️⃣ Page Setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="🌾 SHAP + LIME Interpretability Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌾 SHAP + LIME Interpretability Dashboard")
st.markdown("""
Explore **global** and **per-crop** model explanations using SHAP and LIME.  
Each visualization highlights how soil features influence crop classification.
---
""")

# ------------------------------------------------------------
# 2️⃣ File Paths
# ------------------------------------------------------------
BASE_DIR = "../images"
GLOBAL_DIR = os.path.join(BASE_DIR, "global_lime_shap_comparison")
CROP_DIR = os.path.join(BASE_DIR, "per_crop_shap_lime")

# ------------------------------------------------------------
# 3️⃣ Load Data
# ------------------------------------------------------------
@st.cache_data
def load_csvs():
    data = {}
    try:
        data["global_shap"] = pd.read_csv(os.path.join(GLOBAL_DIR, "global_shap_feature_importance.csv"))
        data["global_lime"] = pd.read_csv(os.path.join(GLOBAL_DIR, "global_lime_feature_importance.csv"))
        data["global_comparison"] = pd.read_csv(os.path.join(GLOBAL_DIR, "lime_shap_feature_comparison.csv"))
    except:
        st.warning("⚠️ Global SHAP/LIME data not found. Run your generation scripts first.")

    try:
        data["per_crop"] = pd.read_csv(os.path.join(CROP_DIR, "per_crop_shap_lime_comparison.csv"))
        data["top3"] = pd.read_csv(os.path.join(CROP_DIR, "top3_drivers_per_crop.csv"))
    except:
        st.warning("⚠️ Per-crop SHAP/LIME data not found. Run per-crop analysis first.")
    return data

data = load_csvs()

# ------------------------------------------------------------
# Sidebar Navigation
# ------------------------------------------------------------
st.sidebar.header("🧭 Navigation")
page = st.sidebar.radio(
    "Select Section:",
    ["Overview", "Global Comparison", "Per-Crop Analysis", "Top Drivers Summary"]
)

# ------------------------------------------------------------
# 4️⃣ Overview
# ------------------------------------------------------------
if page == "Overview":
    st.subheader("📘 Overview")
    st.markdown("""
This dashboard provides insights into **model interpretability** using:
- **SHAP (SHapley Additive Explanations):** Global contribution of each feature.
- **LIME (Local Interpretable Model-Agnostic Explanations):** Localized sensitivity for specific samples.

### What You’ll See:
1. Global feature comparison between SHAP and LIME.  
2. Per-crop interpretability showing feature influence directionality.  
3. Top drivers summary for all crops.
---
    """)

    st.image(os.path.join(GLOBAL_DIR, "global_lime_shap_heatmap.png"), caption="Global SHAP vs LIME Heatmap", use_container_width=True)
    st.image(os.path.join(GLOBAL_DIR, "lime_shap_ratio_bar.png"), caption="LIME/SHAP Ratio Comparison", use_container_width=True)

# ------------------------------------------------------------
# 5️⃣ Global Comparison
# ------------------------------------------------------------
elif page == "Global Comparison":
    st.subheader("🌍 Global SHAP + LIME Comparison")

    col1, col2 = st.columns(2)
    with col1:
        if "global_shap" in data:
            st.markdown("#### 🧠 SHAP Feature Importance")
            st.dataframe(data["global_shap"].round(4))
            st.image(os.path.join(GLOBAL_DIR, "global_lime_shap_heatmap.png"), use_container_width=True)
    with col2:
        if "global_lime" in data:
            st.markdown("#### 🍃 LIME Feature Importance")
            st.dataframe(data["global_lime"].round(4))
            st.image(os.path.join(GLOBAL_DIR, "lime_shap_ratio_bar.png"), use_container_width=True)

    if "global_comparison" in data:
        st.markdown("### 📈 Combined SHAP + LIME Comparison Table")
        st.dataframe(data["global_comparison"].round(4))

# ------------------------------------------------------------
# 6️⃣ Per-Crop Analysis
# ------------------------------------------------------------
elif page == "Per-Crop Analysis":
    st.subheader("🌾 Per-Crop Interpretability Analysis")

    if "per_crop" in data:
        df = data["per_crop"]
        crops = sorted(df["Crop"].unique().tolist())

        selected_crop = st.sidebar.selectbox("Select Crop:", crops)
        df_crop = df[df["Crop"] == selected_crop]

        st.markdown(f"### 🌱 {selected_crop} — SHAP + LIME Breakdown")
        st.write(df_crop[["Feature", "SHAP_Importance", "LIME_Importance", "LIME_to_SHAP_Ratio", "SHAP_Direction"]].round(4))

        # SHAP and LIME side-by-side barplots
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### SHAP Importance")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(df_crop, x="Feature", y="SHAP_Importance", color="cornflowerblue", ax=ax)
            ax.set_title(f"{selected_crop} — SHAP Importance")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)

        with col2:
            st.markdown("#### LIME Importance")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(df_crop, x="Feature", y="LIME_Importance", color="lightgreen", ax=ax)
            ax.set_title(f"{selected_crop} — LIME Importance")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)

        # Directionality and ratio heatmaps
        st.markdown("### 🌿 Directionality and Sensitivity Alignment")
        st.image(os.path.join(CROP_DIR, "per_crop_lime_shap_ratio_heatmap.png"), use_container_width=True)
        st.image(os.path.join(CROP_DIR, "per_crop_shap_directionality_heatmap.png"), use_container_width=True)

# ------------------------------------------------------------
# 7️⃣ Top Drivers Summary
# ------------------------------------------------------------
elif page == "Top Drivers Summary":
    st.subheader("🌿 Top-3 SHAP-Based Feature Drivers per Crop")

    if "top3" in data:
        st.dataframe(data["top3"].round(4))
        crops = sorted(data["top3"]["Crop"].unique().tolist())
        selected_crop = st.sidebar.selectbox("Select Crop:", crops)

        df_crop_top = data["top3"][data["top3"]["Crop"] == selected_crop]
        st.markdown(f"### 🌱 Top Features for {selected_crop}")

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(df_crop_top, x="Feature", y="SHAP_Importance", hue="Feature", ax=ax)
        plt.title(f"Top 3 SHAP Drivers for {selected_crop}")
        st.pyplot(fig)

    st.info("💾 All results and plots are stored under `../images/per_crop_shap_lime/`.")

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.markdown("""
---
👨‍🔬 *Built for explainable soil-crop modeling with SHAP + LIME.*  
Data & visualizations © 2025 — Interpretability Research Lab
""")
