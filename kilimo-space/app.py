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
st.title("🌾 KilimoSpace: Ultimate 13-Month Time-Series Analysis")
st.markdown("---")

st.header("Deep Historical Inference")
st.write("Select an analysis date. The app will scan the 13-month growth phenology of the farm leading up to that date, bypassing the padding hack to feed real historical data to the model.")

# User Inputs: The Dynamic Lookback Window
col1, col2, col3 = st.columns(3)
target_date = col1.date_input("Select Analysis Date", datetime.date.today())
lat = col2.number_input("Latitude", value=0.515, format="%.5f") 
lon = col3.number_input("Longitude", value=34.275, format="%.5f")

# Visual Proof: The Map
st.map(pd.DataFrame({'lat': [lat], 'lon': [lon]}), zoom=10)

if st.button("Run Deep Temporal Analysis", type="primary"):
    if not ee_ready:
        st.error("Earth Engine is not authenticated.")
    elif model is None:
        st.error("Model assets not found.")
    else:
        address = get_location_details(lat, lon)
        st.info(f"📍 **Identified Region:** {address}")

        # --- THE DYNAMIC FIX: Calculate the 13 months leading up to the chosen date ---
        months_to_fetch = []
        for i in range(12, -1, -1):  # Count backward from 12 down to 0
            m = target_date.month - i
            y = target_date.year
            # Math trick to roll back the years if the month goes below 1 (January)
            while m <= 0:
                m += 12
                y -= 1
            # We use the 15th of each month as a safe middle-of-the-month target
            months_to_fetch.append(datetime.date(y, m, 15))
        
        master_169_array = []
        
        st.warning("⏳ Initiating 13-Month Server Query. This will take 1-3 minutes. Please do not refresh...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # 100-meter buffer to analyze the whole farm instead of one pixel
            center_point = ee.Geometry.Point([lon, lat])
            farm_area = center_point.buffer(100) 

            # Loop through all 13 dynamically calculated months
            for i, current_target_date in enumerate(months_to_fetch):
                month_name = current_target_date.strftime('%B %Y')
                status_text.text(f"Fetching cloud-free composite for {month_name} ({i+1}/13)...")
                
                # Create a 30-day window for each month to find a clear image
                start_date = (current_target_date - datetime.timedelta(days=15)).strftime('%Y-%m-%d')
                end_date = (current_target_date + datetime.timedelta(days=15)).strftime('%Y-%m-%d')
                
                collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                    .filterBounds(farm_area) \
                    .filterDate(start_date, end_date) \
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
                    .select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'])
                
                # Get the mathematical median of the month to remove minor clouds
                median_image = collection.median()
                
                # Extract the 12 bands across the farm area
                try:
                    props = median_image.reduceRegion(ee.Reducer.mean(), farm_area, 10).getInfo()
                except Exception as e:
                    props = {} # If the whole month was too cloudy, fallback to empty
                
                # Safely get values (if a month is totally cloudy, it defaults to 0 to prevent crashing)
                b1 = props.get('B1') or 0
                b2 = props.get('B2') or 0
                b3 = props.get('B3') or 0
                b4 = props.get('B4') or 0
                b5 = props.get('B5') or 0
                b6 = props.get('B6') or 0
                b7 = props.get('B7') or 0
                b8 = props.get('B8') or 0
                b8a = props.get('B8A') or 0
                b9 = props.get('B9') or 0
                b11 = props.get('B11') or 0
                b12 = props.get('B12') or 0
                
                # Calculate NDVI for this specific month
                ndvi = (b8 - b4) / (b8 + b4) if (b8 + b4) != 0 else 0
                
                # Append these 13 values to our master array
                master_169_array.extend([b1, b2, b3, b4, b5, b6, b7, b8, b8a, b9, b11, b12, ndvi])
                
                # Update UI Progress Bar
                progress_bar.progress((i + 1) / 13)
            
            # --- FINAL PREDICTION ---
            status_text.text("months retrived")
            
            # Convert the final 169-feature list to a DataFrame and scale it
            input_df = pd.DataFrame([master_169_array], columns=feature_cols)
            scaled_input = scaler.transform(input_df)
            
            # Safety Gate: We check the most recent month's indices (the last 13 items in the array)
            latest_ndvi = master_169_array[-1]
            latest_b3 = master_169_array[-11]
            latest_b8 = master_169_array[-6]
            latest_ndwi = (latest_b3 - latest_b8) / (latest_b3 + latest_b8) if (latest_b3 + latest_b8) != 0 else 0

            if latest_ndwi > 0.1:
                prediction_label = "Water Body"
            elif latest_ndvi < 0.15:
                prediction_label = "Non-Vegetated (Bare Soil / Desert / Urban)"
            else:
                pred_idx = model.predict(scaled_input)
                prediction_label = le.inverse_transform(pred_idx)[0]

            st.metric("Detected Crop Classification (Based on 12-Month History)", prediction_label)
            
            if latest_ndvi < 0.15:
                st.warning("Notice: Recent spectral reflection indicates a lack of active crop vegetation right now.")

        except Exception as e:
            st.error(f"Deep Analysis Failed or Timed Out: {e}")
