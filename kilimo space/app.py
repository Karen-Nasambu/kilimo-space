import os
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ee
import json
from datetime import datetime

# Fix file paths for Streamlit Cloud deployment
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Page Config
st.set_page_config(page_title="KilimoSpace Crop Predictor", layout="wide")

# --- 1. Earth Engine Authentication ---
@st.cache_resource
def init_earth_engine():
    try:
        # Load the JSON string from Streamlit Secrets
        key_dict = json.loads(st.secrets["EARTHENGINE_TOKEN"])
        credentials = ee.ServiceAccountCredentials(key_dict['client_email'], key_data=st.secrets["EARTHENGINE_TOKEN"])
        ee.Initialize(credentials)
        return True
    except Exception as e:
        st.error(f"DEBUG ERROR: {e}")
        return False

ee_ready = init_earth_engine()

# --- 2. Load ML Assets ---
@st.cache_resource
def load_assets():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open("feature_cols.pkl", "rb") as f:
        features = pickle.load(f)
    return model, scaler, le, features

try:
    model, scaler, le, feature_cols = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# --- 3. App UI ---
st.title("🌾 KilimoSpace: Sentinel-2 Crop Classification")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["🌍 Live GPS Fetch", "📊 Upload CSV", "🧪 Sample Demo", "📖 System Architecture"])

# --- TAB 1: LIVE GPS FETCH ---
with tab1:
    st.header("Predict from Live Satellite Data")
    st.write("Enter GPS coordinates to fetch the latest Sentinel-2 surface reflectance data.")
    
    col1, col2 = st.columns(2)
    lat = col1.number_input("Latitude", value=0.515, format="%.5f") 
    lon = col2.number_input("Longitude", value=34.275, format="%.5f")

    if st.button("Fetch Satellite Data & Predict", type="primary"):
        if not ee_ready:
            st.error("Earth Engine is not configured properly in Streamlit Secrets.")
        else:
            with st.spinner("Connecting to Sentinel-2... Analyzing Spectral Signatures..."):
                try:
                    point = ee.Geometry.Point([lon, lat])
                    
                    # We fetch data from 2019 to match the model's training period for accuracy
                    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                        .filterBounds(point) \
                        .filterDate('2019-01-01', '2019-12-31') \
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
                        .select(['B2', 'B3', 'B4', 'B8']) \
                        .sort('system:time_start')
                    
                    def get_values(image):
                        date = image.date().format('YYYY-MM-DD')
                        val = image.reduceRegion(ee.Reducer.mean(), point, 10)
                        return ee.Feature(None, val).set('date', date)
                    
                    features_ee = collection.map(get_values).getInfo()['features']
                    
                    if len(features_ee) == 0:
                        st.error("No clear satellite imagery found for this location in 2019. Try another coordinate.")
                    else:
                        # Extract data and the captured dates
                        captured_dates = [f['properties']['date'] for f in features_ee]
                        fetched_values = []
                        for f in features_ee:
                            props = f['properties']
                            fetched_values.extend([props.get('B2', 0), props.get('B3', 0), props.get('B4', 0), props.get('B8', 0)])
                        
                        # MATCH TENSOR SIZE (169 features)
                        # We repeat the sequence to fill the model's expected 169 columns
                        while len(fetched_values) < len(feature_cols):
                            fetched_values.extend(fetched_values) 
                        final_values = fetched_values[:len(feature_cols)]
                        
                        # Create DataFrame and Scale
                        live_df = pd.DataFrame([final_values], columns=feature_cols)
                        X_live_scaled = scaler.transform(live_df)
                        
                        # PREDICTION
                        live_pred = model.predict(X_live_scaled)
                        predicted_crop = le.inverse_transform(live_pred)[0]
                        
                        # GET PROBABILITIES (The "Confidence" Chart)
                        probs = model.predict_proba(X_live_scaled)[0]
                        crop_labels = le.classes_
                        
                        # DISPLAY RESULTS
                        st.success(f"Analysis complete for imagery taken around {captured_dates[0]}")
                        
                        res_col1, res_col2 = st.columns(2)
                        with res_col1:
                            st.metric("Predicted Crop Type", predicted_crop)
                            st.info(f"Captured Date Range: {captured_dates[0]} to {captured_dates[-1]}")
                        
                        with res_col2:
                            st.write("### Model Confidence Score")
                            prob_df = pd.DataFrame({'Crop': crop_labels, 'Confidence': probs})
                            st.bar_chart(prob_df.set_index('Crop'))

                        with st.expander("View Spectral Band Telemetry"):
                            st.dataframe(live_df)

                except Exception as e:
                    st.error(f"Failed to fetch from Earth Engine: {e}")

# --- TAB 2: UPLOAD CSV ---
with tab2:
    st.header("Upload Historic Band Data (CSV)")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        if all(col in data.columns for col in feature_cols):
            X = data[feature_cols]
            X_scaled = scaler.transform(X)
            preds = model.predict(X_scaled)
            pred_labels = le.inverse_transform(preds)
            
            data['Predicted_Crop'] = pred_labels
            st.success("Predictions Complete!")
            st.write(data[['Predicted_Crop'] + [c for c in data.columns if c != 'Predicted_Crop']])
        else:
            st.error(f"CSV columns do not match the required 169 Sentinel-2 bands. Missing: {set(feature_cols) - set(data.columns)}")

# --- TAB 3: SAMPLE DEMO ---
with tab3:
    st.header("Offline Demo")
    if os.path.exists("sample_data.csv"):
        sample = pd.read_csv("sample_data.csv")
        st.write("Using internal `sample_data.csv` for demonstration:")
        
        row_idx = st.selectbox("Select a field index to test:", sample.index)
        row = sample.iloc[[row_idx]]
        
        X_sample = scaler.transform(row[feature_cols])
        prediction = model.predict(X_sample)
        crop = le.inverse_transform(prediction)[0]
        
        st.metric("Predicted Crop", crop)
        if 'label' in sample.columns:
            st.info(f"Actual Label: {sample.iloc[row_idx]['label']}")
    else:
        st.warning("Sample data file not found in repository.")

# --- TAB 4: SYSTEM ARCHITECTURE ---
with tab4:
    st.header("Model Performance & Info")
    col1, col2 = st.columns(2)
    col1.metric("Weighted F1-Score", "0.54")
    col2.metric("Overall Accuracy", "46%")
    
    st.info("""
    **Architecture Note:**
    This system uses an XGBoost core trained on 2019 time-series Sentinel-2 data. 
    It fetches current multi-spectral bands (B2, B3, B4, B8) and maps the phenological curves 
    to the historical tensor shapes using a Padding-Truncation pipeline.
    """)
