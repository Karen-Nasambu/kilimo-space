import os
# Fix file paths for Streamlit Cloud deployment
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------
# Page Configuration
# ------------------------------------------------
st.set_page_config(
    page_title="Kilimo-Space",
    page_icon="🛰️",
    layout="wide"
)

# ------------------------------------------------
# Load Saved Model Files
# ------------------------------------------------
@st.cache_resource
def load_models():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open("feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return model, scaler, le, feature_cols

model, scaler, le, feature_cols = load_models()

# ------------------------------------------------
# Crop color mapping for charts
# ------------------------------------------------
CROP_COLORS = {
    "Cassava":                "#e67e22",
    "Cassava & Common Bean":  "#e74c3c",
    "Common Bean":            "#9b59b6",
    "Maize":                  "#f1c40f",
    "Maize & Common Bean":    "#2ecc71",
    "Maize & Cassava":        "#1abc9c",
    "Sugarcane":              "#3498db",
}

# ================================================
# SIDEBAR
# ================================================
st.sidebar.title("🛰️ Kilimo-Space")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Select Module:",
    ["🏠 Home", "🌍 Crop Type Mapping", "📂 Upload Your Data"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Coverage Area:**  
Western Kenya (Busia/Bungoma)

**Supported Crops:**
- 🌽 Maize
- 🌿 Cassava
- 🫘 Common Bean
- 🌽🌿 Maize & Cassava
- 🌽🫘 Maize & Common Bean
- 🌿🫘 Cassava & Common Bean
- 🌾 Sugarcane
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📊 Model Info
**Algorithm:** XGBoost + Class Weights  
**Accuracy:** 46%  
**F1 Score (weighted):** 0.54  
**Features:** 169  
**Crop Classes:** 7  
**Training Fields:** ~2,629  
""")

# ================================================
# HOME PAGE
# ================================================
if page == "🏠 Home":
    st.title("🛰️ Kilimo-Space")
    st.subheader("AI-Powered Crop Mapping for Western Kenya")
    st.markdown("""
    Kilimo-Space uses **Sentinel-2 satellite imagery** and machine learning
    to help farmers and the Ministry of Agriculture identify crop types
    across fields in Western Kenya — without visiting the farm!
    """)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🌍 Crop Type Mapping — Sample Demo
        See the model working on **real test fields** from the dataset.

        **How it works:**
        - Select a field from the test set
        - Model predicts the crop type
        - See confidence scores for all 7 classes

        **Accuracy: 46% | F1 Score: 0.54**  
        *(Weighted F1 is the fairer metric on this imbalanced dataset)*
        """)
        if st.button("Go to Crop Mapping →", use_container_width=True):
            st.info("Select '🌍 Crop Type Mapping' from the sidebar!")

    with col2:
        st.markdown("""
        ### 📂 Upload Your Own Data
        Have Sentinel-2 band data for your fields?
        Upload a CSV and get predictions instantly.

        **How it works:**
        - Download the CSV template with correct column names
        - Fill in your field band values
        - Upload and get crop predictions + confidence scores

        **Format:** 169 spectral band columns per field
        """)
        if st.button("Go to Upload Data →", use_container_width=True):
            st.info("Select '📂 Upload Your Data' from the sidebar!")

    st.markdown("---")

    # About the model
    st.markdown("### 🔬 About the Model")
    col3, col4, col5, col6 = st.columns(4)
    col3.metric("Algorithm", "XGBoost")
    col4.metric("Accuracy", "46%")
    col5.metric("F1 Score", "0.54")
    col6.metric("Crop Classes", "7")

    st.info("""
    **Why 46% accuracy?**  
    The model genuinely predicts all 7 crop classes including rare intercrops 
    like Cassava & Common Bean. A model that only predicted Maize would score 
    44% accuracy while failing 6 classes entirely. The weighted F1 of 0.54 
    is the more meaningful metric here.
    """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:gray; font-size:12px;">
    🛰️ Kilimo-Space | Crop Mapping — Western Kenya |
    Powered by Sentinel-2 Satellite Imagery & XGBoost
    </div>
    """, unsafe_allow_html=True)

# ================================================
# CROP TYPE MAPPING — SAMPLE DEMO
# ================================================
elif page == "🌍 Crop Type Mapping":
    st.title("🌍 Crop Type Mapping — Sample Demo")
    st.markdown("""
    These are **real test fields** from the dataset that the model 
    has never seen during training. Select a field to see the prediction 
    and full confidence breakdown.
    """)
    st.markdown("---")

    try:
        sample_data = pd.read_csv("sample_data.csv")
        true_labels  = sample_data["true_label"].tolist()
        X_sample     = sample_data.drop(columns=["true_label"])

        # Scale and predict
        X_sample_scaled = scaler.transform(X_sample)
        preds       = model.predict(X_sample_scaled)
        probas      = model.predict_proba(X_sample_scaled)
        pred_labels = le.inverse_transform(preds)

        # Field selector
        field_num = st.selectbox(
            "Select a field to inspect:",
            options=range(1, len(true_labels) + 1),
            format_func=lambda x: f"Field {x}"
        )

        idx        = field_num - 1
        true_crop  = true_labels[idx]
        pred_crop  = pred_labels[idx]
        correct    = true_crop == pred_crop

        # Result card
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🌿 True Crop", true_crop)
        with col2:
            st.metric("🤖 Predicted Crop", pred_crop)
        with col3:
            if correct:
                st.success("✅ Correct Prediction!")
            else:
                st.error("❌ Incorrect Prediction")

        # Confidence bar chart
        st.markdown("#### Prediction Confidence Per Crop Class")
        proba_row  = probas[idx]
        crop_names = le.classes_

        fig, ax = plt.subplots(figsize=(8, 4))
        bar_colors = [CROP_COLORS.get(c, "#95a5a6") for c in crop_names]
        bars = ax.barh(crop_names, proba_row * 100,
                       color=bar_colors, edgecolor="black", linewidth=0.5)
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=10)
        ax.set_xlabel("Confidence (%)")
        ax.set_title(f"Field {field_num} — Prediction Confidence",
                     fontweight="bold", fontsize=13)
        ax.set_xlim(0, 110)
        ax.axvline(x=50, color="black", linestyle="--",
                   alpha=0.4, label="50% threshold")
        ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

        st.caption(
            "⚠️ Confidence below 50% means the model is uncertain — "
            "field verification is recommended before making agronomic decisions."
        )

        # Summary table of all fields
        st.markdown("---")
        st.markdown("#### Summary — All Sample Fields")
        summary = pd.DataFrame({
            "Field #":        range(1, len(true_labels) + 1),
            "True Crop":      true_labels,
            "Predicted Crop": pred_labels,
            "Confidence (%)": (probas.max(axis=1) * 100).round(1),
            "Correct?":       ["✅" if t == p else "❌"
                               for t, p in zip(true_labels, pred_labels)]
        })
        st.dataframe(summary, use_container_width=True)

    except FileNotFoundError:
        st.warning("⚠️ sample_data.csv not found. Make sure it is in your repo.")

# ================================================
# UPLOAD YOUR DATA
# ================================================
elif page == "📂 Upload Your Data":
    st.title("📂 Upload Your Data for Predictions")
    st.markdown("""
    Upload a CSV file containing Sentinel-2 band values for your fields.  
    The CSV must have **169 columns** matching the exact feature names 
    the model was trained on.
    """)

    # Download template
    feature_df   = pd.DataFrame(columns=feature_cols)
    csv_template = feature_df.to_csv(index=False)
    st.download_button(
        label="⬇️ Download CSV Template (empty with correct column names)",
        data=csv_template,
        file_name="kilimo_space_template.csv",
        mime="text/csv"
    )

    st.markdown("---")
    uploaded_file = st.file_uploader("Upload your CSV file here", type=["csv"])

    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded! {len(user_df)} fields detected.")

            # Validate columns
            missing_cols = [c for c in feature_cols if c not in user_df.columns]
            extra_cols   = [c for c in user_df.columns if c not in feature_cols]

            if missing_cols:
                st.error(
                    f"❌ Your CSV is missing {len(missing_cols)} required columns. "
                    f"First few missing: {missing_cols[:5]}"
                )
                st.stop()

            if extra_cols:
                st.warning(
                    f"⚠️ {len(extra_cols)} extra columns found and will be ignored."
                )

            # Reorder, scale, predict
            X_user        = user_df[feature_cols]
            X_user_scaled = scaler.transform(X_user)
            predictions   = model.predict(X_user_scaled)
            probabilities = model.predict_proba(X_user_scaled)
            pred_labels   = le.inverse_transform(predictions)
            confidence    = probabilities.max(axis=1) * 100

            # Results table
            results_df = pd.DataFrame({
                "Field #":        range(1, len(pred_labels) + 1),
                "Predicted Crop": pred_labels,
                "Confidence (%)": confidence.round(1),
            })

            def color_confidence(val):
                if val >= 70:
                    return "background-color:#d5f5e3"
                elif val >= 50:
                    return "background-color:#fef9e7"
                else:
                    return "background-color:#fdecea"

            st.markdown("### 🌾 Prediction Results")
            st.dataframe(
                results_df.style.applymap(
                    color_confidence, subset=["Confidence (%)"]
                ),
                use_container_width=True
            )

            # Crop distribution chart
            st.markdown("### 📊 Crop Distribution in Your Uploaded Data")
            crop_counts  = pd.Series(pred_labels).value_counts()
            colors_chart = [CROP_COLORS.get(c, "#95a5a6")
                            for c in crop_counts.index]

            fig2, ax2 = plt.subplots(figsize=(8, 4))
            bars2 = ax2.barh(crop_counts.index, crop_counts.values,
                             color=colors_chart, edgecolor="black", linewidth=0.5)
            ax2.bar_label(bars2, fmt="%d fields", padding=3, fontsize=10)
            ax2.set_xlabel("Number of Fields")
            ax2.set_title("Predicted Crop Types in Uploaded Data",
                          fontweight="bold", fontsize=13)
            ax2.set_xlim(0, crop_counts.max() * 1.2)
            plt.tight_layout()
            st.pyplot(fig2)

            # Download results
            result_csv = results_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Predictions as CSV",
                data=result_csv,
                file_name="kilimo_space_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

# ================================================
# FOOTER
# ================================================
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:gray; font-size:12px;">
🛰️ Kilimo-Space | Crop Mapping — Western Kenya |
Powered by Sentinel-2 Satellite Imagery & XGBoost
</div>
""", unsafe_allow_html=True)

