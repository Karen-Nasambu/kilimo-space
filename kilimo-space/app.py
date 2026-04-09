import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ee
import json
from geopy.geocoders import Nominatim

# -----------------------------------------------
# Load Models
# -----------------------------------------------
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
    sample_data = pd.read_csv("sample_data.csv")
    return model, scaler, le, feature_cols, sample_data

@st.cache_resource
def init_earth_engine():
    try:
        key_dict = json.loads(st.secrets["EARTHENGINE_TOKEN"])
        credentials = ee.ServiceAccountCredentials(
            key_dict['client_email'],
            key_data=st.secrets["EARTHENGINE_TOKEN"]
        )
        ee.Initialize(credentials)
        return True
    except Exception as e:
        st.error(f"Earth Engine initialization failed: {e}")
        return False

model, scaler, le, feature_cols, sample_data = load_models()
ee_ready = init_earth_engine()

# -----------------------------------------------
# Page Config
# -----------------------------------------------
st.set_page_config(
    page_title="KilimoSpace",
    page_icon="🛰️",
    layout="wide"
)

# -----------------------------------------------
# Sidebar
# -----------------------------------------------
st.sidebar.title("🛰️ KilimoSpace")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate:",
    ["🏠 Home", "🌍 Live Prediction", "📊 Sample Demo", "ℹ️ Model Info"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Coverage:** Western Kenya  
**Crops Detected:** 7 classes  
**Model:** XGBoost + Class Weights  
**Accuracy:** 46% | F1: 0.54  
""")

# -----------------------------------------------
# Home Page
# -----------------------------------------------
if page == "🏠 Home":
    st.title("🛰️ KilimoSpace")
    st.subheader("Satellite-Based Crop Type Mapping for Western Kenya")
    
    st.markdown("""
    KilimoSpace uses **Sentinel-2 satellite imagery** and machine learning 
    to identify crop types in Western Kenya — without visiting the farm.
    """)
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Accuracy", "46%")
    col2.metric("Weighted F1 Score", "0.54")
    col3.metric("vs Random Baseline", "14%")
    
    st.info("""
    **Why 46% is meaningful:**  
    A random classifier across 7 crop classes achieves only 14%.  
    Our model is 3× better than chance.  
    The weighted F1 of 0.54 shows reliable performance on minority classes 
    that accuracy alone would hide.
    """)
    
    st.markdown("---")
    st.subheader("🔄 Prediction Pipeline")
    st.markdown("""
    1. User enters GPS coordinates  
    2. Google Earth Engine fetches real Sentinel-2 spectral bands  
    3. NDVI computed for each observation date  
    4. Values mapped to 169 model features  
    5. StandardScaler normalizes the feature vector  
    6. XGBoost predicts crop type with confidence scores  
    """)

# -----------------------------------------------
# Live Prediction Page
# -----------------------------------------------
elif page == "🌍 Live Prediction":
    st.title("🌍 Live Satellite Prediction")
    st.write("Enter GPS coordinates to fetch real Sentinel-2 data and predict crop type.")
    
    col1, col2 = st.columns(2)
    lat = col1.number_input("Latitude", value=0.515, format="%.5f",
                             min_value=0.0, max_value=2.0)
    lon = col2.number_input("Longitude", value=34.275, format="%.5f",
                             min_value=33.0, max_value=35.0)
    
    if st.button("🔍 Fetch & Predict", type="primary"):
        if not ee_ready:
            st.error("Earth Engine not authenticated.")
        else:
            with st.spinner("Fetching Sentinel-2 data..."):
                try:
                    point = ee.Geometry.Point([lon, lat])
                    
                    collection = (
                        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                        .filterBounds(point)
                        .filterDate('2024-01-01', '2026-04-08')
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                        .select(['B1','B2','B3','B4','B5','B6',
                                 'B7','B8','B8A','B9','B11','B12'])
                        .sort('system:time_start', False)
                        .limit(13)
                    )
                    
                    def extract(image):
                        vals = image.reduceRegion(
                            ee.Reducer.mean(), point, 10)
                        return ee.Feature(None, vals).set(
                            'date', image.date().format('YYYY-MM-dd'))
                    
                    features_ee = collection.map(extract).getInfo()['features']
                    
                    if not features_ee:
                        st.error("No clear imagery found for these coordinates.")
                    else:
                        latest_date = features_ee[0]['properties']['date']
                        st.success(f"Retrieved {len(features_ee)} observations. Latest: {latest_date}")
                        
                        band_map = {
                            'B1':'B01','B2':'B02','B3':'B03','B4':'B04',
                            'B5':'B05','B6':'B06','B7':'B07','B8':'B08',
                            'B8A':'B8A','B9':'B09','B11':'B11','B12':'B12'
                        }
                        
                        feature_lookup = {}
                        for obs in features_ee:
                            props = obs['properties']
                            date_key = props.get('date','')[:10].replace('-','')
                            for gee_band, model_band in band_map.items():
                                val = props.get(gee_band, 0) or 0
                                # Scale from GEE units to reflectance
                                feature_lookup[f"{model_band}_{date_key}"] = val / 10000
                            nir = (props.get('B8', 0) or 0) / 10000
                            red = (props.get('B4', 0) or 0) / 10000
                            ndvi = (nir - red) / (nir + red + 1e-10)
                            feature_lookup[f"NDVI_{date_key}"] = ndvi
                        
                        # Check NDVI
                        ndvi_vals = [v for k,v in feature_lookup.items() if 'NDVI' in k]
                        mean_ndvi = np.mean(ndvi_vals) if ndvi_vals else 0
                        
                        # Check NDWI
                        sample_props = features_ee[0]['properties']
                        green = (sample_props.get('B3', 0) or 0) / 10000
                        nir_s = (sample_props.get('B8', 0) or 0) / 10000
                        ndwi = (green - nir_s) / (green + nir_s + 1e-10)
                        
                        col_r1, col_r2 = st.columns(2)
                        col_r1.metric("Mean NDVI", f"{mean_ndvi:.3f}")
                        col_r2.metric("NDWI", f"{ndwi:.3f}")
                        
                        # Conditions
                        if ndwi > 0.1:
                            st.warning("🌊 Water body detected. Model not applied.")
                        elif mean_ndvi < 0.15:
                            st.warning("🏜️ Low vegetation detected. May be bare soil or urban area.")
                        else:
                            # Build feature vector
                            fetched_vals = [v for v in feature_lookup.values() if v != 0]
                            fallback = float(np.mean(fetched_vals)) if fetched_vals else 0.0
                            
                            row = []
                            matched = 0
                            for col in feature_cols:
                                if col in feature_lookup:
                                    row.append(feature_lookup[col])
                                    matched += 1
                                else:
                                    row.append(fallback)
                            
                            st.write(f"Features matched: {matched} / {len(feature_cols)}")
                            
                            input_df = pd.DataFrame([row], columns=feature_cols)
                            scaled = scaler.transform(input_df)
                            pred = model.predict(scaled)
                            probs = model.predict_proba(scaled)[0]
                            crop = le.inverse_transform(pred)[0]
                            
                            st.success(f"## 🌱 Predicted Crop: {crop}")
                            st.metric("Confidence", f"{max(probs)*100:.1f}%")
                            
                            prob_df = pd.DataFrame({
                                'Crop': le.classes_,
                                'Confidence': probs
                            })
                            st.bar_chart(prob_df.set_index('Crop'))
                
                except Exception as e:
                    st.error(f"Error: {e}")

# -----------------------------------------------
# Sample Demo Page
# -----------------------------------------------
elif page == "📊 Sample Demo":
    st.title("📊 Sample Field Demo")
    st.write("Predictions on real preprocessed fields from the test dataset.")
    
    row_idx = st.selectbox("Select a field:", sample_data.index)
    row = sample_data.iloc[[row_idx]]
    
    X_sample = row[feature_cols].values
    scaled = scaler.transform(X_sample)
    pred = model.predict(scaled)
    probs = model.predict_proba(scaled)[0]
    crop = le.inverse_transform(pred)[0]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Crop", crop)
        if 'true_label' in sample_data.columns:
            true = sample_data.iloc[row_idx]['true_label']
            st.metric("Actual Crop", true)
            result = "✅ Correct!" if crop == true else "❌ Incorrect"
            st.info(result)
    
    with col2:
        st.write("### Confidence per Class")
        prob_df = pd.DataFrame({
            'Crop': le.classes_,
            'Confidence': probs
        })
        st.bar_chart(prob_df.set_index('Crop'))

# -----------------------------------------------
# Model Info Page
# -----------------------------------------------
elif page == "ℹ️ Model Info":
    st.title("ℹ️ Model Information")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Weighted F1 Score", "0.54")
    col2.metric("Overall Accuracy", "46%")
    col3.metric("Random Baseline", "14%")
    
    st.markdown("---")
    st.subheader("Training Summary")
    st.markdown("""
    - **Model:** XGBoost with Class Weights  
    - **Features:** 169 (12 bands × 13 dates + 13 NDVI values)  
    - **Training samples:** 2,629 fields  
    - **Test samples:** 657 fields  
    - **Classes:** 7 crop types including intercrop combinations  
    - **Imbalance handling:** Class Weights (Maize: 1,462 vs rarest: 79 fields)  
    """)
    
    st.subheader("Crop Classes")
    crops = ['Maize', 'Cassava', 'Common Bean',
             'Maize & Cassava', 'Maize & Common Bean',
             'Maize & Soybean', 'Cassava & Common Bean']
    for crop in crops:
        st.write(f"- {crop}")

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:gray; font-size:12px;">
🛰️ KilimoSpace | Sentinel-2 Crop Mapping | Western Kenya
</div>
""", unsafe_allow_html=True)
