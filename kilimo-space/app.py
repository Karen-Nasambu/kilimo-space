import os
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import ee
import json
import joblib
from geopy.geocoders import Nominatim

# Fix file paths for Streamlit Cloud
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Page Config
st.set_page_config(page_title="KilimoSpace Crop Predictor", layout="wide")

# --- 1. Earth Engine Authentication ---
@st.cache_resource
def init_earth_engine():
    try:
        key_dict = json.loads(st.secrets["EARTHENGINE_TOKEN"])
        credentials = ee.ServiceAccountCredentials(key_dict['client_email'], key_data=st.secrets["EARTHENGINE_TOKEN"])
        ee.Initialize(credentials)
        return True
    except Exception as e:
        st.error(f"Earth Engine Error: {e}")
        return False

ee_ready = init_earth_engine()

# --- 2. Load ML Assets ---
@st.cache_resource
def load_assets():
    # Using joblib or pickle based on your file types
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    le = joblib.load("label_encoder.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    return model, scaler, le, feature_cols

try:
    model, scaler, le, feature_cols = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# --- 3. Helper Functions ---
def get_location_name(lat, lon):
    try:
        geolocator = Nominatim(user_agent="kilimo_space_app")
        location = geolocator.reverse(f"{lat}, {lon}", timeout=10)
        return location.address if location else "Unknown Location"
    except:
        return "Location details unavailable"

# --- 4. App UI ---
st.title("🌾 KilimoSpace: Sentinel-2 Crop Classification")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🌍 Live GPS Fetch", "📊 Upload CSV", "📖 System Info"])

# --- TAB 1: LIVE GPS FETCH ---
with tab1:
    st.header("Predict from Live Satellite Data (2026)")
    
    col1, col2 = st.columns(2)
    lat = col1.number_input("Latitude", value=0.515, format="%.5f") 
    lon = col2.number_input("Longitude", value=34.275, format="%.5f")

    if st.button("Fetch Satellite Data & Predict", type="primary"):
        if not ee_ready:
            st.error("Earth Engine is not initialized.")
        else:
            # Tell the user WHERE they are looking
            location_address = get_location_name(lat, lon)
            st.info(f"📍 **Analyzing Location:** {location_address}")

            with st.spinner("Fetching 2026 Sentinel-2 Imagery..."):
                try:
                    point = ee.Geometry.Point([lon, lat])
                    
                    # Fetching Live 2026 Data
                    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                        .filterBounds(point) \
                        .filterDate('2026-01-01', '2026-04-08') \
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
                        .select(['B2', 'B3', 'B4', 'B8']) \
                        .sort('system:time_start', False)
                    
                    def get_values(image):
                        date = image.date().format('YYYY-MM-DD')
                        val = image.reduceRegion(ee.Reducer.mean(), point, 10)
                        return ee.Feature(None, val).set('date', date)
                    
                    features_ee = collection.map(get_values).getInfo()['features']
                    
                    if len(features_ee) == 0:
                        st.error("No clear 2026 images found. Try different coordinates.")
                    else:
                        # Extract and format for the model
                        captured_dates = [f['properties']['date'] for f in features_ee]
                        fetched_values = []
                        for f in features_ee:
                            props = f['properties']
                            fetched_values.extend([props.get('B2', 0), props.get('B3', 0), props.get('B4', 0), props.get('B8', 0)])
                        
                        # Pad to 169 features
                        while len(fetched_values) < len(feature_cols):
                            fetched_values.extend(fetched_values) 
                        final_values = fetched_values[:len(feature_cols)]
                        
                        # Dataframe -> Scaling -> Prediction
                        input_df = pd.DataFrame([final_values], columns=feature_cols)
                        scaled_data = scaler.transform(input_df)
                        
                        # Prediction + Probabilities
                        prediction = model.predict(scaled_data)
                        crop_name = le.inverse_transform(prediction)[0]
                        probs = model.predict_proba(scaled_data)[0]
                        
                        # --- RESULTS DISPLAY ---
                        st.success(f"Latest data captured on: {captured_dates[0]}")
                        
                        res_col1, res_col2 = st.columns(2)
                        with res_col1:
                            st.metric("Predicted Crop", crop_name)
                            st.write(f"**GPS:** {lat}, {lon}")
                        
                        with res_col2:
                            st.write("### Prediction Confidence")
                            prob_df = pd.DataFrame({'Crop': le.classes_, 'Confidence': probs})
                            st.bar_chart(prob_df.set_index('Crop'))

                        with st.expander("View Spectral Data Details"):
                            st.dataframe(input_df)

                except Exception as e:
                    st.error(f"Pipeline Error: {e}")

# --- TAB 2: CSV UPLOAD ---
with tab2:
    st.header("Batch Prediction via CSV")
    uploaded_file = st.file_uploader("Upload CSV with 169 columns", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        if all(col in data.columns for col in feature_cols):
            scaled_batch = scaler.transform(data[feature_cols])
            batch_preds = model.predict(scaled_batch)
            data['Predicted_Crop'] = le.inverse_transform(batch_preds)
            st.write(data)
        else:
            st.error("Column mismatch in CSV.")

# --- TAB 3: SYSTEM INFO ---
with tab3:
    st.header("Architecture")
    st.write("This model uses **XGBoost** trained on historical Sentinel-2 data.")
    st.write("The Live Fetcher maps **2026 Real-time imagery** to the model's feature space.")
