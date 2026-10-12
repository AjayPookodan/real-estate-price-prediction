# Real Estate Price Prediction

## Project Description

A machine learning project that predicts real-estate prices using property characteristics, location, and accessibility to public amenities and transportation facilities.

## Team Members

* Ajay Balakrishnan
* Gopika CP
* Devakrishnan

## Objectives

* Analyse property prices and their influencing factors.
* Preprocess and explore real-estate datasets.
* Study the impact of location and nearby amenities on property prices.
* Train machine learning models to predict property prices.
* Develop a user-friendly application for making predictions.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* OSMnx
* GeoPy
* Folium
* Streamlit

## Project Structure

```text
real-estate-price-prediction/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
├── models/
├── app/
├── reports/
├── .gitignore
└── README.md
```

## 1. Data Collection and Understanding

### Dataset Overview

* **Dataset:** Gurgaon Real Estate Listings
* **Number of Records:** 10,704
* **Number of Attributes:** 67
* **Format:** CSV
* **Target Variable:** `PRICE`
* **Tools Used:** Python, Pandas, Geopy, Nominatim, Folium

### Data Source and Selection

The Gurgaon real-estate dataset was selected because it contains diverse property-related information, including listed prices, property types, locality details, area measurements, furnishing, and amenities. These attributes provide a foundation for analysing the factors that influence property prices and developing a machine learning prediction model.

* **Source URL:** To be verified
* **Author/Uploader:** To be verified
* **Licence:** To be verified
* **Download Date:** To be recorded

### Data Dictionary

A data dictionary is maintained in `reports/data_dictionary.md`. It documents all 67 attributes, including their meanings, data types, units, missing-value percentages, and relevant observations. This helps identify useful prediction features and understand the structure of the dataset before preprocessing.

### Geographic Coordinate Collection

The dataset's locality information was used to obtain geographic coordinates where suitable coordinates were unavailable. The Geopy library and Nominatim geocoding service were used to retrieve latitude and longitude for **231 unique localities**.

The results were stored in a cache to reduce repeated requests. An interactive map was generated using Folium, and approximate geographic boundaries were used to flag potentially incorrect coordinates.

**Limitation:** Coordinates represent localities rather than exact property addresses. Consequently, properties within the same locality may share coordinates, making distance calculations to amenities approximate rather than property-specific.

### Initial Data Quality Observations

Initial inspection identified several issues requiring further investigation:

* **Inconsistent price formats:** The `PRICE` column contains values such as `2.63 Cr`, `3.6 Cr`, and comma-separated amounts such as `85,000`. These must be converted into a consistent numerical format after verifying their units and transaction types.
* **Mixed data types:** Some attributes, including `FURNISHING_ATTRIBUTES`, contain mixed data types and require inspection.
* **Missing values:** Missing-value percentages are documented in the data dictionary and will guide preprocessing decisions.
* **Potentially redundant features:** Multiple price, area, identifier, and property-description fields must be examined for overlap and relevance.
* **Geographic accuracy:** Geocoded coordinates must be checked for incorrect locations before calculating distances to nearby amenities.
* **Time coverage:** Listing dates must be inspected to determine the period represented by the dataset.

### Data Storage

The original dataset is preserved in `data/raw/housing_raw.csv`. Geocoded data is stored separately in `data/raw/housing_with_coords.csv`, while the geocoding cache is maintained in `data/raw/geocode_cache.json`. The data dictionary and interactive locality map are stored in the `reports/` directory.

### Outcome

This module establishes the initial dataset and documents its structure, geographic information, and data quality considerations. The findings will guide the next stage: data preprocessing, feature engineering, and preparation of a consistent target variable for machine learning.

