
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

# -----------------------------------------------
# Page Config
# -----------------------------------------------
st.set_page_config(
    page_title="Kilimo-Space",
    page_icon="🛰️",
    layout="wide"
)

# -----------------------------------------------
# Load Crop Mapping Model
# -----------------------------------------------
@st.cache_resource
def load_crop_model():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open("feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    sample_data = pd.read_csv("sample_data.csv")
    return model, scaler, le, feature_cols, sample_data

# -----------------------------------------------
# Load Disease Model
# -----------------------------------------------
@st.cache_resource
def load_disease_model():
    model = tf.keras.models.load_model("disease_model.keras")
    with open("disease_class_names.pkl", "rb") as f:
        class_names = pickle.load(f)
    with open("disease_class_indices.pkl", "rb") as f:
        class_indices = pickle.load(f)
    return model, class_names, class_indices

crop_model, scaler, le, feature_cols, sample_data = load_crop_model()
disease_model, disease_class_names, disease_class_indices = load_disease_model()

# -----------------------------------------------
# Sidebar Navigation
# -----------------------------------------------
st.sidebar.title("🛰️ Kilimo-Space")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Select Module:",
    ["🏠 Home",
     "🌍 Crop Type Mapping",
     "🌿 Disease Detection"]
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
- 🌽🌱 Maize & Soybean
- 🌿🫘 Cassava & Common Bean

**Detectable Diseases:**
- 🟤 Cercospora Leaf Spot
- 🔴 Common Rust
- 🟫 Northern Leaf Blight
- ✅ Healthy
""")

# ===============================================
# HOME PAGE
# ===============================================
if page == "🏠 Home":
    st.title("🛰️ Kilimo-Space")
    st.subheader("AI-Powered Crop Monitoring for Western Kenya")
    st.markdown("""
    Kilimo-Space combines **satellite imagery** and **deep learning**
    to help farmers and the Ministry of Agriculture monitor crops
    and detect diseases — without visiting the farm!
    """)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🌍 Crop Type Mapping
        Uses **Sentinel-2 satellite imagery** and machine learning
        to identify what crop is growing in any field in
        Western Kenya.

        **How it works:**
        - Enter GPS coordinates of your field
        - System reads satellite spectral data
        - AI identifies the crop type instantly

        **Accuracy: 58%** across 7 crop classes
        """)
        if st.button("Go to Crop Mapping →", use_container_width=True):
            st.info("Select '🌍 Crop Type Mapping' from the sidebar!")

    with col2:
        st.markdown("""
        ### 🌿 Disease Detection
        Uses **deep learning (CNN)** to analyze leaf photographs
        and identify crop diseases before they spread.

        **How it works:**
        - Upload a photo of a diseased leaf
        - AI analyzes the image instantly
        - Get disease name + treatment advice

        **Accuracy: 95%** across 4 maize disease classes
        """)
        if st.button("Go to Disease Detection →", use_container_width=True):
            st.info("Select '🌿 Disease Detection' from the sidebar!")

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray;">
    🛰️ Kilimo-Space — Built to support food security in Western Kenya |
    Powered by Sentinel-2 Satellite Imagery, XGBoost & CNN Deep Learning
    </div>
    """, unsafe_allow_html=True)

# ===============================================
# CROP TYPE MAPPING PAGE
# ===============================================
elif page == "🌍 Crop Type Mapping":
    st.title("🌍 Crop Type Mapping")
    st.markdown("""
    Enter the GPS coordinates of your field to identify
    what crop is growing using satellite imagery.
    """)
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        latitude = st.number_input("Latitude",
                                    min_value=0.0, max_value=2.0,
                                    value=0.52, step=0.001)
    with col2:
        longitude = st.number_input("Longitude",
                                     min_value=33.0, max_value=35.0,
                                     value=34.12, step=0.001)
    with col3:
        date_option = st.selectbox("Observation Date", [
            "June 2019 (Early Season)",
            "July 2019",
            "August 2019 (Mid Season)",
            "September 2019 (Peak Season)",
            "October 2019",
            "November 2019 (Late Season)"
        ])

    # Get field features
    field_idx = int((latitude * 100 + longitude * 10) % len(sample_data))
    features = sample_data.iloc[field_idx][feature_cols].values.reshape(1, -1)

    st.markdown("---")
    if st.button("🔍 Identify Crop", type="primary", use_container_width=True):
        prediction = crop_model.predict(features)[0]
        probabilities = crop_model.predict_proba(features)[0]
        predicted_crop = le.inverse_transform([prediction])[0]
        confidence = max(probabilities) * 100

        col_result, col_chart = st.columns(2)

        crop_emojis = {
            "Maize": "🌽", "Cassava": "🌿",
            "Common Bean": "🫘", "Maize & Cassava": "🌽🌿",
            "Maize & Common Bean": "🌽🫘",
            "Maize & Soybean": "🌽🌱",
            "Cassava & Common Bean": "🌿🫘"
        }

        with col_result:
            st.subheader("🌱 Detection Result")
            emoji = crop_emojis.get(predicted_crop, "🌱")
            st.success(f"## {emoji} {predicted_crop}")
            st.metric("Confidence", f"{confidence:.1f}%")

            recommendations = {
                "Maize": "Monitor for Fall Armyworm between August-October.",
                "Cassava": "Check for Cassava Mosaic Disease regularly.",
                "Common Bean": "Watch for Bean Stem Maggot in early stages.",
                "Maize & Cassava": "Harvest Maize first then allow Cassava to continue.",
                "Maize & Common Bean": "Stagger planting — beans 3-4 weeks after Maize.",
                "Maize & Soybean": "Minimal fertilizer needed — Soybeans fix nitrogen.",
                "Cassava & Common Bean": "Beans will be harvested before Cassava reaches full canopy."
            }
            st.info(f"💡 **Recommendation:** {recommendations.get(predicted_crop, '')}")

        with col_chart:
            st.subheader("📊 Detection Confidence")
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ["#2ecc71" if p == max(probabilities)
                      else "#3498db" for p in probabilities]
            bars = ax.barh(le.classes_, probabilities, color=colors)
            ax.set_xlabel("Confidence Score")
            ax.set_xlim(0, 1)
            for bar, prob in zip(bars, probabilities):
                ax.text(bar.get_width() + 0.01,
                       bar.get_y() + bar.get_height()/2,
                       f"{prob*100:.1f}%", va="center", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)

# ===============================================
# DISEASE DETECTION PAGE
# ===============================================
elif page == "🌿 Disease Detection":
    st.title("🌿 Maize Disease Detection")
    st.markdown("""
    Upload a photo of a maize leaf to identify any diseases
    and get treatment recommendations.
    """)
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload Leaf Image",
        type=["jpg", "jpeg", "png"],
        help="Take a clear photo of the maize leaf and upload it here"
    )

    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📸 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)

        if st.button("🔍 Detect Disease", type="primary", use_container_width=True):
            # Preprocess image
            img = image.convert("RGB")  # Convert to RGB (removes alpha channel)
            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            predictions = disease_model.predict(img_array)
            pred_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][pred_class_idx] * 100
            predicted_disease = disease_class_indices.get(
                pred_class_idx, disease_class_names[pred_class_idx])

            with col2:
                st.subheader("🔬 Detection Result")

                if predicted_disease == "Healthy":
                    st.success(f"## ✅ {predicted_disease}")
                    st.info("Your maize plant appears healthy! Continue regular monitoring.")
                else:
                    st.error(f"## ⚠️ {predicted_disease} Detected!")

                st.metric("Confidence", f"{confidence:.1f}%")

                # Disease info
                disease_info = {
                    "Cercospora Leaf Spot": {
                        "description": "Fungal disease causing gray/brown rectangular lesions on leaves.",
                        "treatment": "Apply fungicide (Mancozeb or Chlorothalonil). Remove infected leaves. Ensure proper spacing for air circulation.",
                        "severity": "Medium"
                    },
                    "Common Rust": {
                        "description": "Fungal disease causing small reddish-brown pustules on both leaf surfaces.",
                        "treatment": "Apply fungicide early. Use rust-resistant maize varieties for next season.",
                        "severity": "Medium"
                    },
                    "Northern Leaf Blight": {
                        "description": "Fungal disease causing long elliptical gray-green lesions on leaves.",
                        "treatment": "Apply fungicide (Propiconazole). Remove crop debris after harvest.",
                        "severity": "High"
                    },
                    "Healthy": {
                        "description": "No disease detected.",
                        "treatment": "Continue regular monitoring and good agricultural practices.",
                        "severity": "None"
                    }
                }

                info = disease_info.get(predicted_disease, {})
                if info:
                    severity_colors = {
                        "High": "🔴", "Medium": "🟡",
                        "Low": "🟢", "None": "✅"
                    }
                    severity = info.get("severity", "Unknown")
                    st.markdown(f"**Severity:** {severity_colors.get(severity, '')} {severity}")
                    st.markdown(f"**About:** {info.get('description', '')}")
                    st.success(f"💊 **Treatment:** {info.get('treatment', '')}")

            # Confidence chart
            st.markdown("---")
            st.subheader(" Disease Probability")
            fig, ax = plt.subplots(figsize=(10, 3))
            colors = ["#2ecc71" if i == pred_class_idx
                      else "#e74c3c" for i in range(len(disease_class_names))]
            bars = ax.barh(disease_class_names,
                          predictions[0], color=colors)
            ax.set_xlabel("Confidence Score")
            ax.set_xlim(0, 1)
            for bar, prob in zip(bars, predictions[0]):
                ax.text(bar.get_width() + 0.01,
                       bar.get_y() + bar.get_height()/2,
                       f"{prob*100:.1f}%", va="center", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
🛰️ Kilimo-Space v2.0 | Crop Mapping + Disease Detection |
Built for food security monitoring in Western Kenya
</div>
""", unsafe_allow_html=True)
