# ================================================
# KILIMO-SPACE: Crop Type Prediction App
# Sentinel-2 Satellite Imagery — Western Kenya
# ================================================

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ------------------------------------------------
# Page Configuration
# ------------------------------------------------
st.set_page_config(
    page_title="Kilimo-Space",
    page_icon="🌾",
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
# HEADER
# ================================================
st.markdown("""
    <h1 style='text-align:center; color:#2c7a3a;'>🛰️ Kilimo-Space</h1>
    <h4 style='text-align:center; color:#555;'>Satellite Crop Mapping — Western Kenya</h4>
    <hr>
""", unsafe_allow_html=True)

st.markdown("""
This app uses a trained **XGBoost model** on **Sentinel-2 satellite imagery** 
to predict the crop type for agricultural fields in Western Kenya.  
It works with **169 spectral features** — 12 bands across 13 dates plus NDVI.
""")

# ================================================
# SIDEBAR — App Info
# ================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Sentinel-2.jpg/320px-Sentinel-2.jpg",
             caption="Sentinel-2 Satellite", use_container_width=True)
    st.markdown("### 📊 Model Info")
    st.info("""
    **Algorithm:** XGBoost  
    **Imbalance:** Class Weights  
    **Accuracy:** 46%  
    **F1 Score (weighted):** 0.54  
    **Features:** 169  
    **Crop Classes:** 7  
    **Training Fields:** ~2,629  
    """)

    st.markdown("### 🌾 Crop Classes")
    for crop, color in CROP_COLORS.items():
        st.markdown(
            f"<span style='background:{color};padding:2px 8px;"
            f"border-radius:4px;color:white;font-size:13px'>{crop}</span>",
            unsafe_allow_html=True
        )
        st.write("")

# ================================================
# TABS
# ================================================
tab1, tab2, tab3 = st.tabs([
    "📂 Upload Your Data",
    "🔍 Sample Demo",
    "ℹ️ How To Use"
])

# ================================================
# TAB 1 — CSV UPLOAD
# ================================================
with tab1:
    st.subheader("📂 Upload a CSV for Real Predictions")
    st.markdown("""
    Upload a CSV file containing Sentinel-2 band values for your fields.  
    The CSV must have **169 columns** matching the exact feature names the model was trained on.
    """)

    # Download feature columns as reference
    feature_df = pd.DataFrame(columns=feature_cols)
    csv_template = feature_df.to_csv(index=False)
    st.download_button(
        label="⬇️ Download CSV Template (empty with correct column names)",
        data=csv_template,
        file_name="kilimo_space_template.csv",
        mime="text/csv"
    )

    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Upload your CSV file here",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded! {len(user_df)} fields detected.")

            # ---- Validate columns ----
            missing_cols = [c for c in feature_cols if c not in user_df.columns]
            extra_cols   = [c for c in user_df.columns if c not in feature_cols]

            if missing_cols:
                st.error(f"❌ Your CSV is missing {len(missing_cols)} required columns. "
                         f"First few missing: {missing_cols[:5]}")
                st.stop()

            if extra_cols:
                st.warning(f"⚠️ {len(extra_cols)} extra columns found and will be ignored: "
                           f"{extra_cols[:5]}")

            # ---- Reorder columns to match model ----
            X_user = user_df[feature_cols]

            # ---- Scale ----
            X_user_scaled = scaler.transform(X_user)

            # ---- Predict ----
            predictions    = model.predict(X_user_scaled)
            probabilities  = model.predict_proba(X_user_scaled)
            pred_labels    = le.inverse_transform(predictions)
            confidence     = probabilities.max(axis=1) * 100

            # ---- Results table ----
            results_df = pd.DataFrame({
                "Field #":         range(1, len(pred_labels) + 1),
                "Predicted Crop":  pred_labels,
                "Confidence (%)":  confidence.round(1),
            })

            # Color code confidence
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

            # ---- Summary chart ----
            st.markdown("### 📊 Crop Distribution in Your Uploaded Data")
            crop_counts = pd.Series(pred_labels).value_counts()
            colors_chart = [CROP_COLORS.get(c, "#95a5a6") for c in crop_counts.index]

            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.barh(crop_counts.index, crop_counts.values,
                           color=colors_chart, edgecolor="black", linewidth=0.5)
            ax.bar_label(bars, fmt="%d fields", padding=3, fontsize=10)
            ax.set_xlabel("Number of Fields")
            ax.set_title("Predicted Crop Types in Uploaded Data",
                         fontweight="bold", fontsize=13)
            ax.set_xlim(0, crop_counts.max() * 1.2)
            plt.tight_layout()
            st.pyplot(fig)

            # ---- Download results ----
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
# TAB 2 — SAMPLE DEMO
# ================================================
with tab2:
    st.subheader("🔍 Live Demo — Predictions on Sample Fields")
    st.markdown("""
    These are **real test fields** from the dataset that the model has never seen during training.  
    Select a field below to see the prediction and confidence breakdown.
    """)

    try:
        sample_df = pd.read_csv("sample_data.csv")
        true_labels = sample_df["true_label"].tolist()

        # Drop true label before prediction
        X_sample = sample_df.drop(columns=["true_label"])

        # Scale
        X_sample_scaled = scaler.transform(X_sample)

        # Predict
        preds = model.predict(X_sample_scaled)
        probas = model.predict_proba(X_sample_scaled)
        pred_labels = le.inverse_transform(preds)

        # Field selector
        field_num = st.selectbox(
            "Select a field to inspect:",
            options=range(1, len(true_labels) + 1),
            format_func=lambda x: f"Field {x}"
        )

        idx = field_num - 1
        true_crop = true_labels[idx]
        pred_crop = pred_labels[idx]
        correct    = true_crop == pred_crop

        # ---- Result card ----
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

        # ---- Probability bar chart ----
        st.markdown("#### Prediction Confidence Per Crop Class")
        proba_row = probas[idx]
        crop_names = le.classes_

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        bar_colors = [
            CROP_COLORS.get(c, "#95a5a6") for c in crop_names
        ]
        bars2 = ax2.barh(
            crop_names, proba_row * 100,
            color=bar_colors, edgecolor="black", linewidth=0.5
        )
        ax2.bar_label(bars2, fmt="%.1f%%", padding=3, fontsize=10)
        ax2.set_xlabel("Confidence (%)")
        ax2.set_title(f"Field {field_num} — Prediction Confidence",
                      fontweight="bold", fontsize=13)
        ax2.set_xlim(0, 110)
        ax2.axvline(x=50, color="black", linestyle="--",
                    alpha=0.4, label="50% threshold")
        ax2.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig2)

        st.caption(
            "⚠️ Confidence below 50% means the model is uncertain — "
            "field verification is recommended before making agronomic decisions."
        )

        # ---- Summary table of all 10 fields ----
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
        st.warning("⚠️ sample_data.csv not found. Make sure it was saved from your notebook.")

# ================================================
# TAB 3 — HOW TO USE
# ================================================
with tab3:
    st.subheader("ℹ️ How To Use This App")

    st.markdown("""
    ### Option 1 — Upload Your Own Data (CSV)
    1. Click **Download CSV Template** in the Upload tab to get the correct column names
    2. Fill in your Sentinel-2 band values — one row per field
    3. Upload the filled CSV
    4. The app will predict the crop type and confidence for each field
    5. Download the predictions as CSV

    ---

    ### Option 2 — Explore the Sample Demo
    1. Go to the **Sample Demo** tab
    2. Select any of the 10 pre-loaded test fields
    3. See the true crop vs predicted crop and full confidence breakdown

    ---

    ### Understanding the Results

    | Confidence Level | Meaning |
    |---|---|
    | 🟢 70%+ | High confidence — reliable prediction |
    | 🟡 50–70% | Moderate — consider field verification |
    | 🔴 Below 50% | Low — model is uncertain, verify in field |

    ---

    ### About the Model
    - **Algorithm:** XGBoost with Class Weights
    - **Satellite:** Sentinel-2, Western Kenya, 2019
    - **Features:** 169 (12 spectral bands × 13 dates + NDVI)
    - **Classes:** 7 crop types
    - **Accuracy:** 46% | **F1 Score:** 0.54
    - **Note:** Lower accuracy reflects the model genuinely predicting all 7 classes
      including minority intercrops, not just defaulting to Maize.

    ---

    ### Data Requirements for CSV Upload
    Your CSV must contain **169 columns** with Sentinel-2 band values.  
    Column naming format: `BANDNAME_DATE` (e.g. `B08_20190606`, `NDVI_20191004`)  
    Each row represents one agricultural field.
    """)

    st.info("""
    **For researchers and agronomists:** Band values should be mean reflectance 
    values per field extracted from Sentinel-2 Level-2A imagery 
    (atmospherically corrected surface reflectance).
    """)
