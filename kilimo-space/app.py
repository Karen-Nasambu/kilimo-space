import os
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import ee
import json
import joblib
from geopy.geocoders import Nominatim

os.chdir(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="KilimoSpace Crop Predictor", layout="wide")

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
        st.error(f"Earth Engine Error: {e}")
        return False

ee_ready = init_earth_engine()

@st.cache_resource
def load_assets():
    try:
        with open("best_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
        with open("feature_cols.pkl", "rb") as f:
            feature_cols = pickle.load(f)
        return model, scaler, le, feature_cols
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None, None

model, scaler, le, feature_cols = load_assets()

def get_location_name(lat, lon):
    try:
        geolocator = Nominatim(user_agent="kilimospace_app")
        location = geolocator.reverse(f"{lat}, {lon}", timeout=10)
        return location.address if location else "Remote area"
    except:
        return "Location lookup unavailable"

st.title("KilimoSpace: Sentinel-2 Crop Classification")
st.markdown("---")

tab1, tab2, tab3 = st.tabs([
    "Live GPS Fetch",
    "Sample Demo",
    "Model Info"
])

# ================================================
# TAB 1: LIVE GPS FETCH — proper 12-band fetching
# ================================================
with tab1:
    st.header("Live Satellite Prediction")
    st.write("Enter GPS coordinates to fetch real Sentinel-2 data and predict crop type.")

    col1, col2 = st.columns(2)
    lat = col1.number_input("Latitude", value=0.515, format="%.5f")
    lon = col2.number_input("Longitude", value=34.275, format="%.5f")

    if st.button("Fetch and Predict", type="primary"):
        if not ee_ready:
            st.error("Earth Engine not authenticated.")
        elif model is None:
            st.error("Model files not loaded.")
        else:
            location = get_location_name(lat, lon)
            st.info(f"Location: {location}")

            with st.spinner("Fetching Sentinel-2 data from Google Earth Engine..."):
                try:
                    point = ee.Geometry.Point([lon, lat])

                    # Fetch ALL 12 bands across multiple dates
                    # This gives the model proper multi-band
                    # temporal input instead of 4 repeated bands
                    collection = (
                        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                        .filterBounds(point)
                        .filterDate('2024-01-01', '2026-04-08')
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                        .select([
                            'B1', 'B2', 'B3', 'B4',
                            'B5', 'B6', 'B7', 'B8',
                            'B8A', 'B9', 'B11', 'B12'
                        ])
                        .sort('system:time_start', False)
                        .limit(13)  # fetch up to 13 dates matching training
                    )

                    def extract(image):
                        vals = image.reduceRegion(
                            ee.Reducer.mean(), point, 10
                        )
                        return ee.Feature(None, vals).set(
                            'date',
                            image.date().format('YYYY-MM-dd')
                        )

                    features_ee = collection.map(extract).getInfo()['features']

                    if not features_ee:
                        st.error("No clear imagery found. Try different coordinates.")
                    else:
                        st.success(
                            f"Retrieved {len(features_ee)} satellite "
                            f"observations. Latest: "
                            f"{features_ee[0]['properties']['date']}"
                        )

                        # Build feature vector from all 12 bands per date
                        # Band names from GEE map to model's B01-B12 naming
                        band_map = {
                            'B1': 'B01', 'B2': 'B02', 'B3': 'B03',
                            'B4': 'B04', 'B5': 'B05', 'B6': 'B06',
                            'B7': 'B07', 'B8': 'B08', 'B8A': 'B8A',
                            'B9': 'B09', 'B11': 'B11', 'B12': 'B12'
                        }

                        # Build a lookup: feature_col -> value
                        feature_lookup = {}

                        for obs in features_ee:
                            props = obs['properties']
                            date_str = props.get('date', '')[:10]
                            # Convert date format to match training columns
                            # e.g. 2024-03-15 -> 20240315
                            date_key = date_str.replace('-', '')

                            for gee_band, model_band in band_map.items():
                                col_name = f"{model_band}_{date_key}"
                                val = props.get(gee_band, 0) or 0
                                feature_lookup[col_name] = val

                            # Compute NDVI for this date
                            nir = props.get('B8', 0) or 0
                            red = props.get('B4', 0) or 0
                            ndvi = (nir - red) / (nir + red + 1e-10)
                            feature_lookup[f"NDVI_{date_key}"] = ndvi

                        # Map fetched values to model's exact 169 columns
                        # For columns with no matching date, use mean
                        # of fetched values as fallback
                        fetched_vals = [
                            v for v in feature_lookup.values()
                            if v != 0
                        ]
                        fallback = (
                            float(np.mean(fetched_vals))
                            if fetched_vals else 0.0
                        )

                        row = []
                        matched = 0
                        for col in feature_cols:
                            if col in feature_lookup:
                                row.append(feature_lookup[col])
                                matched += 1
                            else:
                                row.append(fallback)

                        st.write(
                            f"Features matched to model columns: "
                            f"{matched} / {len(feature_cols)}"
                        )

                        # Predict
                        input_df = pd.DataFrame(
                            [row], columns=feature_cols
                        )
                        scaled = scaler.transform(input_df)
                        pred = model.predict(scaled)
                        probs = model.predict_proba(scaled)[0]
                        crop = le.inverse_transform(pred)[0]

                        # Display results
                        r1, r2 = st.columns(2)
                        with r1:
                            st.metric("Predicted Crop Type", crop)
                            st.metric(
                                "Confidence",
                                f"{max(probs)*100:.1f}%"
                            )
                            # Show NDVI values fetched
                            ndvi_vals = [
                                v for k, v in feature_lookup.items()
                                if 'NDVI' in k
                            ]
                            if ndvi_vals:
                                st.metric(
                                    "Mean NDVI (fetched dates)",
                                    f"{np.mean(ndvi_vals):.3f}"
                                )

                        with r2:
                            st.write("### Prediction Confidence")
                            prob_df = pd.DataFrame({
                                'Crop': le.classes_,
                                'Confidence': probs
                            })
                            st.bar_chart(prob_df.set_index('Crop'))

                        with st.expander("View Spectral Data"):
                            st.dataframe(input_df)

                except Exception as e:
                    st.error(f"Error: {e}")

# ================================================
# TAB 2: SAMPLE DEMO — reliable predictions
# ================================================
with tab2:
    st.header("Sample Field Demo")
    st.write(
        "This tab uses real preprocessed field data from "
        "the training dataset to demonstrate model predictions."
    )

    if os.path.exists("sample_data.csv"):
        sample = pd.read_csv("sample_data.csv")
        row_idx = st.selectbox(
            "Select a field to predict:", sample.index
        )
        row = sample.iloc[[row_idx]]

        X_sample = scaler.transform(row[feature_cols])
        pred = model.predict(X_sample)
        probs = model.predict_proba(X_sample)[0]
        crop = le.inverse_transform(pred)[0]

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Predicted Crop", crop)
            if 'true_label' in sample.columns:
                true = sample.iloc[row_idx]['true_label']
                st.metric("Actual Crop", true)
                correct = "Correct" if crop == true else "Incorrect"
                st.info(f"Prediction: {correct}")
        with c2:
            st.write("### Confidence per Class")
            prob_df = pd.DataFrame({
                'Crop': le.classes_,
                'Confidence': probs
            })
            st.bar_chart(prob_df.set_index('Crop'))
    else:
        st.warning("sample_data.csv not found in repository.")

# ================================================
# TAB 3: MODEL INFO
# ================================================
with tab3:
    st.header("Model Architecture")
    c1, c2, c3 = st.columns(3)
    c1.metric("Weighted F1 Score", "0.54")
    c2.metric("Overall Accuracy", "46%")
    c3.metric("Random Baseline", "14%")

    st.markdown("---")
    st.subheader("Why 46% is meaningful")
    st.info(
        "The model classifies 7 crop types. A random classifier "
        "achieves only 14%. Our model performs 3x better than chance. "
        "The weighted F1 of 0.54 shows reliable performance even on "
        "minority classes that accuracy alone would hide."
    )

    st.subheader("Prediction Pipeline")
    st.write(
        "1. User enters GPS coordinates\n"
        "2. Google Earth Engine fetches 12 Sentinel-2 bands "
        "across up to 13 clear observation dates\n"
        "3. NDVI computed per date from NIR and Red bands\n"
        "4. Values mapped to model's 169 feature columns\n"
        "5. StandardScaler normalises the feature vector\n"
        "6. XGBoost predicts crop type with probabilities"
    )

    st.subheader("Training Summary")
    st.write(
        "Model: XGBoost with Class Weights\n"
        "Features: 169 (12 bands x 13 dates + 13 NDVI values)\n"
        "Training samples: 2,629 fields\n"
        "Test samples: 657 fields\n"
        "Classes: 7 crop types including intercrop combinations"
    )
