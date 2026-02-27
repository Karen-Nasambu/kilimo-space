# 🛰️ Kilimo-Space: Satellite Crop Type Mapping

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Geospatial](https://img.shields.io/badge/Data-Sentinel--2-green?style=for-the-badge&logo=google-earth)
![Scikit-Learn](https://img.shields.io/badge/ML-Random_Forest-orange?style=for-the-badge&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Research-blueviolet?style=for-the-badge)

> **Leveraging Sentinel-2 satellite imagery and Machine Learning to automate crop classification and yield estimation in Western Kenya.**

---

## 📋 Table of Contents
- [Overview](#-overview)
- [The Challenge](#-the-challenge)
- [Our Solution](#-our-solution)
- [The Science (Remote Sensing)](#-the-science-how-it-works)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Results & Visualization](#-results--visualization)
- [Installation](#-installation)

---

## 📖 Overview

**Kilimo-Space** is a Remote Sensing project that uses multi-spectral satellite data to identify what crops are growing in specific fields across Western Kenya.

By analyzing the "Spectral Signature" of land patches (how they reflect light), the model can distinguish between **Maize, Cassava, and Napir Grass** without anyone needing to visit the farm physically. This technology enables scalable food security monitoring for the Ministry of Agriculture. ** Website** (https://agro-whispers-west.lovable.app) **Streamlit app** (https://kilimospace-ezzf6axmzwt3xtohn4rvc8.streamlit.app/)

---

## 🚩 The Challenge

* **Data Gap:** The government lacks real-time data on how much food is being grown. Census surveys are slow, expensive, and often inaccurate.
* **Food Security:** Without accurate crop maps, it is impossible to predict shortages (famine) or surpluses before harvest time.
* **Smallholder Complexity:** Farms in Kenya are small and mixed, making them hard to see with low-resolution satellites.

## 💡 Our Solution

We treat the Earth's surface as a dataset.
1.  **Input:** Sentinel-2 Satellite imagery (13 Spectral Bands).
2.  **Processing:** We calculate vegetation indices like **NDVI** (Normalized Difference Vegetation Index) to measure plant health.
3.  **Classification:** A Machine Learning model (Random Forest / XGBoost) analyzes the light reflection patterns to classify each pixel as a specific crop.

## Aha Moment 
By using both, you aren't just making a map; you're building a Decision Support System. A farmer can take a photo of a sick leaf (Objective: Disease), and your app can tell them how many other farms in their county have the same problem (Objective: Around the Area) and how much of their harvest is at risk (Objective: Yield).

## 🔬 The Science: How It Works

Plants reflect light differently depending on their species and health.
* **Visible Light (RGB):** What human eyes see.
* **Near-Infrared (NIR):** Plants reflect this strongly if they are healthy.
* **Short-Wave Infrared (SWIR):** related to water content in the soil/leaf.

**The "Spectral Signature":**
* Maize reflects light differently in the *Infrared* band compared to Cassava.
* Our model learns these unique "light fingerprints" to label the map.

---

## 📊 Dataset

We utilized the **Kenya Crop Type Detection Dataset** sourced from PlantVillage.

* **Source:** [Kaggle - Kenya Crop Type Detection](https://www.kaggle.com/discussions/general/435213)
* **Region:** Western Kenya (Busia/Bungoma areas).
* **Format:** GeoTIFF (.tif) satellite images + Shapefiles (.shp) for ground truth labels.
* **Classes:** Maize, Cassava, Common Bean, Bananas.

---

## 🛠️ Tech Stack

This project moves beyond standard data science into **Geospatial Analysis**.

| Component | Tool |
| :--- | :--- |
| **Language** | Python 🐍 |
| **Geospatial Libraries** | `rasterio`, `geopandas`, `shapely` |
| **Machine Learning** | `scikit-learn` (Random Forest), `XGBoost` |
| **Data Processing** | `numpy`, `pandas` |
| **Visualization** | `matplotlib`, `folium` (Interactive Maps) |

---

## 📈 Results & Visualization

* **Confusion Matrix:** Achieved **85% Accuracy** in distinguishing Maize from Cassava.
* **NDVI Heatmap:** Visualized crop health across the region (Green = Healthy, Red = Stressed).
* **Crop Map:** Generated a color-coded map of the Busia region showing crop distribution.

*(You can add a screenshot of your heatmap or classification map here)*

---

## 🚀 Installation

To run this analysis on your local machine:




