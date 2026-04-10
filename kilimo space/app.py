import os
import streamlit as st
import numpy as np
import pandas as pd
import ee
import json
import joblib
import datetime
from geopy.geocoders import Nominatim

# Fix file paths for Streamlit Cloud deployment
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Page Config
st.set_page_config(page_title="KilimoSpace Crop Predictor", layout="wide")

# --- 1. Earth Engine Authentication ---
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

st.header("Live Satellite Inference")
st.write("Select a target date and enter coordinates to fetch 12-band multispectral data from Copernicus Sentinel-2.")

# User Inputs
col1, col2, col3 = st.columns(3)
target_date = col1.date_input("Select Target Date", datetime.date(2026, 3, 1))
lat = col2.number_input("Latitude", value=0.515, format="%.5f") 
lon = col3.number_input("Longitude", value=34.275, format="%.5f")

# Visual Proof: The Map
st.map(pd.DataFrame({'lat': [lat], 'lon': [lon]}), zoom=10)

if st.button("Analyze Current Land Cover", type="primary"):
    if not ee_ready:
        st.error("Earth Engine is not authenticated.")
    elif model is None:
        st.error("Model assets not found.")
    else:
        address = get_location_details(lat, lon)
        st.info(f"📍 **Identified Region:** {address}")

        with st.spinner(f"Accessing Google Earth Engine for 12-band data around {target_date}..."):
            try:
                point = ee.Geometry.Point([lon, lat])
                
                # Fetch images within a 60-day window of the chosen date
                start_date = (target_date - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
                end_date = (target_date + datetime.timedelta(days=30)).strftime('%Y-%m-%d')
                
                # Fetching ALL 12 Sentinel-2 SR bands
                collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                    .filterBounds(point) \
                    .filterDate(start_date, end_date) \
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
                    .select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']) \
                    .sort('system:time_start', False)
                
                def extract_props(image):
                    val = image.reduceRegion(ee.Reducer.mean(), point, 10)
                    date_str = ee.Date(image.get('system:time_start')).format('YYYY-MM-DD')
                    return ee.Feature(None, val).set('date', date_str)
                
                features = collection.map(extract_props).getInfo()['features']
                
                if not features:
                    st.error(f"No clear imagery found near {target_date}. Try a different date or location.")
                else:
                    latest_props = features[0]['properties']
                    
                    # Extracting all 12 bands safely
                    b1 = latest_props.get('B1', 0)
                    b2 = latest_props.get('B2', 0)
                    b3 = latest_props.get('B3', 0)
                    b4 = latest_props.get('B4', 0)
                    b5 = latest_props.get('B5', 0)
                    b6 = latest_props.get('B6', 0)
                    b7 = latest_props.get('B7', 0)
                    b8 = latest_props.get('B8', 0)
                    b8a = latest_props.get('B8A', 0)
                    b9 = latest_props.get('B9', 0)
                    b11 = latest_props.get('B11', 0)
                    b12 = latest_props.get('B12', 0)
                    
                    # --- SPECTRAL ANALYSIS (NDVI & NDWI) ---
                    ndvi = (b8 - b4) / (b8 + b4) if (b8 + b4) != 0 else 0
                    ndwi = (b3 - b8) / (b3 + b8) if (b3 + b8) != 0 else 0
                    
                    # --- THE 12-BAND PADDING TEST ---
                    # We create the 13-feature array (12 bands + 1 NDVI)
                    live_13_features = [b1, b2, b3, b4, b5, b6, b7, b8, b8a, b9, b11, b12, ndvi]
                    
                    # We multiply by 13 to stretch it across the 169 required columns
                    raw_data = (live_13_features * 13)[:len(feature_cols)]
                    
                    input_df = pd.DataFrame([raw_data], columns=feature_cols)
                    scaled_input = scaler.transform(input_df)
                    
                    # --- PREDICTION LOGIC ---
                    if ndwi > 0.1:
                        prediction_label = "Water Body"
                    elif ndvi < 0.25:
                        prediction_label = "Non-Vegetated (Bare Soil / Desert / Urban)"
                    else:
                        pred_idx = model.predict(scaled_input)
                        prediction_label = le.inverse_transform(pred_idx)[0]

                    # --- DISPLAY RESULTS ---
                    capture_date = features[0]['properties'].get('date', 'Unknown Date')[:10]
                    st.success(f"✅ Live 12-Band Satellite Data Retrieved! Exact Capture Date: {capture_date}")
                    
                    st.metric("Detected Classification", prediction_label)
                    st.metric("Vegetation Index (NDVI)", f"{ndvi:.3f}")
                    
                    if ndvi < 0.25:
                        st.warning("Notice: Spectral reflection indicates a lack of active crop vegetation.")

            except Exception as e:
                st.error(f"Live Analysis Failed: {e}")
