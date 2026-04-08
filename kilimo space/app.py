import os
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import ee
import json
import joblib
from geopy.geocoders import Nominatim

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
        st.error(f"Earth Engine Initialization Error: {e}")
        return False

ee_ready = init_earth_engine()

# --- 2. Load ML Assets ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("best_model.pkl")
        scaler = joblib.load("scaler.pkl")
        le = joblib.load("label_encoder.pkl")
        feature_cols = joblib.load("feature_cols.pkl")
        return model, scaler, le, feature_cols
    except Exception as e:
        st.error(f"Error loading .pkl files: {e}")
        return None, None, None, None

model, scaler, le, feature_cols = load_assets()

# --- 3. Helper Function for Location ---
def get_location_details(lat, lon):
    try:
        geolocator = Nominatim(user_agent="kilimospace_navigator")
        location = geolocator.reverse(f"{lat}, {lon}", timeout=10)
        return location.address if location else "Coordinates in remote area"
    except:
        return "Location lookup unavailable"

# --- 4. App UI ---
st.title("🌾 KilimoSpace: Sentinel-2 Real-Time Crop Classification")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🌍 Live GPS Fetch", "📊 Batch CSV Upload", "📖 System Architecture"])

# --- TAB 1: LIVE GPS FETCH ---
with tab1:
    st.header("Predict from Live Satellite Data (2026)")
    st.write("Enter coordinates to fetch the latest multispectral data from Sentinel-2.")
    
    col1, col2 = st.columns(2)
    lat = col1.number_input("Latitude", value=0.515, format="%.5f") 
    lon = col2.number_input("Longitude", value=34.275, format="%.5f")

    if st.button("Analyze Current Land Cover", type="primary"):
        if not ee_ready:
            st.error("Earth Engine is not authenticated.")
        elif model is None:
            st.error("Model assets not found.")
        else:
            # 1. Show User where they are looking
            address = get_location_details(lat, lon)
            st.info(f"📍 **Identified Region:** {address}")

            with st.spinner("Accessing Copernicus Sentinel-2 Hub..."):
                try:
                    point = ee.Geometry.Point([lon, lat])
                    
                    # Fetch most recent 2026 images
                    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                        .filterBounds(point) \
                        .filterDate('2026-01-01', '2026-04-08') \
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
                        .select(['B2', 'B3', 'B4', 'B8']) \
                        .sort('system:time_start', False)
                    
                    def extract_props(image):
                        val = image.reduceRegion(ee.Reducer.mean(), point, 10)
                        return ee.Feature(None, val).set('date', image.date().format('YYYY-MM-DD'))
                    
                    features = collection.map(extract_props).getInfo()['features']
                    
                    if not features:
                        st.error("No clear 2026 imagery found for these coordinates. Try a different spot.")
                    else:
                        # Get Data
                        latest_props = features[0]['properties']
                        b2 = latest_props.get('B2', 0)
                        b3 = latest_props.get('B3', 0)
                        b4 = latest_props.get('B4', 0)
                        b8 = latest_props.get('B8', 0)
                        
                        # --- SPECTRAL ANALYSIS (NDVI) ---
                        # NDVI = (NIR - RED) / (NIR + RED)
                        ndvi = (b8 - b4) / (b8 + b4) if (b8 + b4) != 0 else 0
                        
                        # Prepare data for XGBoost (Map 4 bands to 169 features)
                        raw_data = ([b2, b3, b4, b8] * 43)[:len(feature_cols)]
                        input_df = pd.DataFrame([raw_data], columns=feature_cols)
                        
                        # SCALE DATA
                        scaled_input = scaler.transform(input_df)
                        
                        # PREDICTION LOGIC with SAFETY OVERRIDE
                        if ndvi < 0.35:
                            # If it's not green, it's not a crop.
                            prediction_label = "Non-Vegetated (Water / Bare Soil / Building)"
                            probabilities = [0.0] * len(le.classes_)
                        else:
                            # Let the model decide
                            pred_idx = model.predict(scaled_input)
                            prediction_label = le.inverse_transform(pred_idx)[0]
                            probabilities = model.predict_proba(scaled_input)[0]

                        # --- DISPLAY RESULTS ---
                        st.success(f"Satellite Data Captured on: {features[0]['properties']['date']}")
                        
                        res1, res2 = st.columns(2)
                        with res1:
                            st.metric("Detected Classification", prediction_label)
                            st.metric("Vegetation Index (NDVI)", f"{ndvi:.3f}")
                            if ndvi < 0.1:
                                st.warning("Notice: Low spectral reflection confirms no active vegetation found.")
                        
                        with res2:
                            st.write("### Model Confidence Chart")
                            conf_df = pd.DataFrame({'Crop': le.classes_, 'Confidence': probabilities})
                            st.bar_chart(conf_df.set_index('Crop'))

                        with st.expander("Show Detailed Spectral Bands"):
                            st.write(input_df)

                except Exception as e:
                    st.error(f"Live Analysis Failed: {e}")

# --- TAB 2: CSV UPLOAD ---
with tab2:
    st.header("Batch Process CSV")
    file = st.file_uploader("Upload CSV with 169 Sentinel-2 features", type="csv")
    if file:
        df = pd.read_csv(file)
        if all(c in df.columns for c in feature_cols):
            scaled_batch = scaler.transform(df[feature_cols])
            preds = model.predict(scaled_batch)
            df['Predicted_Crop'] = le.inverse_transform(preds)
            st.dataframe(df)
        else:
            st.error("CSV columns mismatch the model's 169 feature requirement.")

# --- TAB 3: SYSTEM INFO ---
with tab3:
    st.header("Technical Pipeline")
    st.info("""
    **Training Basis:** XGBoost Model trained on 2019 Phenological Time-Series data.
    **Real-Time Bridge:** The system fetches current 2026 Band values (B2, B3, B4, B8), 
    calculates the NDVI for immediate land validation, and maps values into the 169-feature tensor.
    """)
