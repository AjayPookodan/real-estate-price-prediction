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


# Real Estate Price Prediction — Data Preprocessing Module

## 1. Introduction

Data preprocessing is one of the most important stages of a machine learning project. Raw datasets often contain duplicate records, missing values, inconsistent text, incorrectly formatted numbers, irrelevant columns, and values that may be unrealistic.

A machine learning model cannot reliably learn from data unless the data is prepared properly.

In this project, we are developing a **Real Estate Price Prediction System** that aims to estimate the sale price of a property using information such as its property type, number of bedrooms, area, age, and location.

The original dataset contains real estate listing information, including property descriptions, transaction details, prices, and other property attributes.

The purpose of this preprocessing module is to transform the raw dataset into a cleaner and more consistent dataset that can be used for exploratory data analysis and machine learning.

### Main objectives

* Inspect the original dataset and understand its structure.
* Standardize column names and text values.
* Remove duplicate records.
* Convert property prices into a consistent numerical format.
* Retain properties listed for sale and exclude rental listings.
* identify invalid or missing target values.
* Examine property-area values and unusually large or small observations.
* Prepare location information for further analysis.
* Preserve useful property attributes.
* Separate the dataset used for analysis from the data prepared for modelling.
* Avoid data leakage by deferring encoding and feature scaling to the machine learning pipeline.

---

## 2. Dataset Information

| Property                   | Description                            |
| -------------------------- | -------------------------------------- |
| Project                    | Real Estate Price Prediction           |
| Dataset type               | Real estate property listings          |
| Original file              | `housing_raw.csv`                      |
| Geocoded file, if used     | `housing_with_coords.csv`              |
| Original number of rows    | 10,704                                 |
| Original number of columns | 67                                     |
| Prediction target          | `price_lakhs`                          |
| Target unit                | Indian rupees, expressed in lakhs      |
| Programming language       | Python                                 |
| Data analysis tools        | Pandas, NumPy, Matplotlib and Seaborn  |
| Main environment           | Jupyter Notebook in Visual Studio Code |

**Note:** The original dimensions describe the raw dataset observed during inspection. The exact input file used by the final preprocessing run must be confirmed from the notebook or preprocessing script.

---

## 3. What Is the Target Variable?

The target variable is the value that the machine learning model attempts to predict.

For this project, the target is:

`price_lakhs`

This represents the property's sale price in lakhs of Indian rupees.

For example:

| Original price | Converted price | Meaning     |
| -------------- | --------------: | ----------- |
| `2.63 Cr`      |             263 | ₹2.63 crore |
| `3.6 Cr`       |             360 | ₹3.6 crore  |
| `85,000`       |            0.85 | ₹85,000     |
| `40,000`       |            0.40 | ₹40,000     |
| `1.95 Cr`      |             195 | ₹1.95 crore |

The conversion is necessary because the original prices may be stored as text and may use different units, such as rupees, lakhs, and crores.

A numerical target allows machine learning algorithms to perform calculations and learn relationships between property features and prices.

**Important assumption:** An unlabelled numeric price such as `85,000` is interpreted as rupees by the price-conversion function used in this project. This assumption must be checked against the dataset's original price conventions. A parser should not automatically assume that every unlabelled value is in rupees if the source sometimes uses other units.

---

## 4. Preprocessing Workflow

The preprocessing process can be divided into the following stages.

1. Load the dataset.
2. Inspect the dataset.
3. Standardize column names.
4. Clean text values.
5. Remove duplicate records.
6. Convert prices into numerical values.
7. Filter the dataset to retain sale listings.
8. Remove records with invalid target values.
9. Convert relevant property attributes to numeric types.
10. Inspect missing values.
11. Examine property-area values and outliers.
12. Prepare and group location categories where appropriate.
13. Separate analysis data from modelling data.
14. Preserve the numerical target and useful input features.
15. Save the processed data.

The exact order and outcome of each operation should match the preprocessing notebook or script used for the final model.

---

## 5. Issues Found and Actions Taken

The following table documents the problems investigated during preprocessing and the corresponding actions.

| Issue                                   | Why it matters                                                                                                                | Action taken or recommended                                                                                                                                             | Reason                                                                                               |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Inconsistent column names               | Different capitalisation, spaces, and punctuation make columns harder to access consistently.                                 | Convert column names to lowercase, remove surrounding spaces, and replace spaces or hyphens with underscores.                                                           | Creates predictable column names for subsequent processing.                                          |
| Unnecessary whitespace in text          | Values such as `"Apartment"` and `" Apartment "` can be treated as different categories.                                      | Strip leading and trailing whitespace from string columns.                                                                                                              | Reduces inconsistent category values.                                                                |
| Duplicate records                       | Duplicate listings can give repeated observations more influence during training and evaluation.                              | Remove exact duplicate rows.                                                                                                                                            | Prevents identical records from being counted multiple times.                                        |
| Prices stored as text                   | Values such as `"2.63 Cr"` cannot be used directly as a numerical prediction target.                                          | Convert prices into a common numerical unit: lakhs of rupees.                                                                                                           | Makes the target suitable for regression models.                                                     |
| Different price units                   | Rupees, lakhs, and crores have different numerical scales.                                                                    | Apply a price parser that accounts for the unit indicated in the original value.                                                                                        | Ensures comparable target values.                                                                    |
| Rental and sale listings mixed together | Rental amounts and property sale prices represent different prediction problems.                                              | Retain records identified as sale listings using the dataset's `preference` field, after verifying its meaning.                                                         | Keeps the model focused on predicting sale prices.                                                   |
| Missing or invalid prices               | A supervised regression model needs a valid target value for each training observation.                                       | Remove records whose converted target is missing, non-numeric, or non-positive.                                                                                         | Prevents invalid target values from being used for training.                                         |
| Property attributes stored as text      | Numerical fields may be stored as strings because of formatting or missing-value markers.                                     | Convert appropriate columns, such as bedroom count, property age, and area, to numeric types.                                                                           | Allows numerical analysis and numerical imputation.                                                  |
| Missing property attributes             | Some listings do not report all their features.                                                                               | Measure missingness and retain potentially useful features when appropriate. Handle missing values in the modelling pipeline.                                           | Avoids discarding large amounts of data unnecessarily.                                               |
| Invalid property-area values            | Zero or negative area values cannot represent a valid positive property area.                                                 | Treat non-positive area values as invalid and replace them with missing values if this rule is applied.                                                                 | Prevents invalid area values from misleading the model.                                              |
| Extreme property-area values            | Extremely large areas may be valid estates or land parcels, but may also be data-entry errors.                                | Inspect extreme values and compare them with property type and source information before deciding whether to retain, correct, or remove them.                           | Avoids deleting legitimate expensive or large properties automatically.                              |
| Extreme property prices                 | High-value properties may be legitimate luxury homes or land listings.                                                        | Investigate unusually high and low prices; remove values only when they are demonstrably invalid or inconsistent with the dataset's conventions.                        | Preserves genuine variation in the property market.                                                  |
| Many distinct locality names            | Rare location categories can produce many encoded columns and may be poorly represented during training.                      | Group sufficiently rare non-missing localities into an `"Other"` category if the rule is used. Represent missing locations separately as `"Unknown"` where appropriate. | Reduces category sparsity while preserving a clear missing-location category.                        |
| Price-derived predictor columns         | Columns calculated from the target price can reveal information that would not be available independently at prediction time. | Exclude price-derived columns, such as price per square foot, from the model's input features.                                                                          | Prevents target leakage and unrealistically optimistic evaluation.                                   |
| Identifiers and administrative metadata | Listing IDs, posting dates, descriptions, and similar fields may not be appropriate for the intended prediction model.        | Exclude irrelevant identifiers and metadata from the model feature set after considering their usefulness.                                                              | Keeps the model focused on relevant property characteristics.                                        |
| Categorical variables                   | Machine learning estimators generally require categorical text to be transformed into a suitable representation.              | Defer categorical encoding to the modelling pipeline.                                                                                                                   | Ensures encoding is learned from training data and applied consistently to validation and test data. |
| Numerical features on different scales  | Features such as area, bedroom count, and coordinates have different numerical ranges.                                        | Defer scaling to the modelling pipeline for models that benefit from it.                                                                                                | Keeps transformations within the training workflow and prevents data leakage.                        |

**Important:** This table documents the preprocessing decisions and checks relevant to the project. Before treating every action as completed, verify that it appears in the final notebook or script. An action listed as recommended or conditional should not be reported as completed unless it was actually performed.

---

## 6. Detailed Explanation of Each Preprocessing Step

### Step 1: Load and inspect the dataset

The raw CSV file is loaded using Pandas.

The initial inspection checks:

* The number of rows and columns.
* Column names.
* Data types.
* Missing values.
* Duplicate records.
* Sample property listings.
* The format of the original price column.

This step establishes the starting point for preprocessing and helps identify potential data-quality problems.

Example:

```python
import pandas as pd

df = pd.read_csv("data/raw/housing_raw.csv")

print("Dataset shape:", df.shape)
print(df.head())
print(df.info())
print(df.isna().sum())
```

Use the actual path of the file being processed. If the notebook uses `housing_with_coords.csv`, change the path accordingly.

### Step 2: Standardize column names

Column names are standardized to lowercase, with spaces and hyphens replaced by underscores.

For example:

| Original column | Standardized column |
| --------------- | ------------------- |
| `PROPERTY_TYPE` | `property_type`     |
| `BEDROOM NUM`   | `bedroom_num`       |
| `TOTAL-FLOOR`   | `total_floor`       |

This makes column names easier to use consistently in Python.

Example:

```python
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
    .str.replace("-", "_", regex=False)
)
```

### Step 3: Clean text values

Leading and trailing spaces are removed from string columns.

For example, `" Apartment "` becomes `"Apartment"`.

This helps prevent duplicate category labels caused only by whitespace.

Example:

```python
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()
```

This operation standardizes whitespace. It does not automatically correct spelling mistakes or make every category semantically identical.

### Step 4: Remove duplicate records

Exact duplicate rows are identified and removed.

Example:

```python
before = len(df)

df = df.drop_duplicates()

after = len(df)

print("Duplicate rows removed:", before - after)
print("Rows remaining:", after)
```

Removing duplicates reduces the chance that identical listings will disproportionately influence model training.

However, two listings describing similar properties are not necessarily duplicates. Exact-row deduplication does not identify every possible duplicate listing.

### Step 5: Convert property prices into lakhs

The original price field contains values represented in different formats, including amounts in crores, lakhs, and rupees.

These values are converted into a common unit.

The target is stored as:

`price_lakhs`

Examples:

* ₹1 crore = 100 lakhs.
* ₹2.63 crore = 263 lakhs.
* ₹85,000 = 0.85 lakhs, assuming the original value represents rupees.

The conversion must preserve the meaning of the original amount. Invalid strings must not be silently converted into plausible-looking prices.

After conversion, inspect the resulting values and compare them with the original price strings.

### Step 6: Retain sale listings

The dataset includes a `preference` field with values such as `S` and `R`.

In the inspected data, `S` represented sale listings and `R` represented rental listings. This interpretation should be verified against the dataset documentation or source conventions.

For a sale-price prediction project, rental listings are excluded.

Example:

```python
print(df["preference"].value_counts(dropna=False))

df = df[df["preference"] == "S"].copy()
```

This ensures the model is trained for a single prediction task: estimating the sale price of a property.

If the source uses a different meaning for these codes, the filtering rule must be adjusted.

### Step 7: Remove invalid target values

The target variable must contain valid positive numerical prices.

Records are removed when the converted price is missing, cannot be interpreted numerically, or is zero or negative.

Example:

```python
df["price_lakhs"] = pd.to_numeric(
    df["price_lakhs"],
    errors="coerce"
)

df = df[
    df["price_lakhs"].notna()
    & (df["price_lakhs"] > 0)
].copy()
```

This is justified because a sale-price regression model needs a meaningful target for each training record.

Missing values in input features are a separate issue and do not automatically require deleting the entire record.

### Step 8: Convert property attributes to numeric types

Relevant numerical attributes may include:

* `bedroom_num`
* `bathroom_num`
* `balcony_num`
* `carpet_sqft`
* `builtup_sqft`
* `superbuiltup_sqft`
* `age`
* `latitude`
* `longitude`

These columns should use numerical data types when they represent numerical measurements.

Example:

```python
numeric_columns = [
    "bedroom_num",
    "bathroom_num",
    "balcony_num",
    "carpet_sqft",
    "builtup_sqft",
    "superbuiltup_sqft",
    "age",
    "latitude",
    "longitude"
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
```

Values that cannot be interpreted numerically become missing values. These should be inspected to ensure that valid information has not been lost through conversion.

### Step 9: Handle invalid area values

Property area is an important feature because it can help explain differences in property prices.

Area values should be examined for:

* Missing values.
* Zero or negative measurements.
* Unusually small values.
* Unusually large values.
* Differences between carpet area, built-up area, and super built-up area.

Non-positive area values are invalid for positive area measurements. If present, they can be changed to missing values.

Example:

```python
area_columns = [
    "carpet_sqft",
    "builtup_sqft",
    "superbuiltup_sqft"
]

for col in area_columns:
    if col in df.columns:
        df.loc[df[col] <= 0, col] = float("nan")
```

The code above changes non-positive values to missing values. It does not remove unusually large positive values.

### Step 10: Investigate outliers

An outlier is an observation that is unusually far from the rest of the data.

In this project, potential outliers may occur in:

* Property prices.
* Carpet area.
* Built-up area.
* Super built-up area.

For example, a property with a very large reported area may be a data-entry error, but it may also be a genuine large estate or land listing.

Therefore, extreme values should not be removed merely because they are large.

The investigation should consider:

1. Whether the value is physically and numerically plausible.
2. Whether the property's type supports the reported measurement.
3. Whether the original listing supports the value.
4. Whether the value could be explained by a unit-conversion error.
5. Whether the value is outside the expected domain of the project.

Only values with a defensible reason for removal should be excluded.

#### Outlier rule justification

| Rule                                        | Justification                                                                                                      |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Remove missing or non-positive sale prices  | These values cannot serve as valid sale-price targets.                                                             |
| Treat non-positive area as missing          | A zero or negative area is not a valid positive area measurement.                                                  |
| Investigate extreme positive prices         | Luxury properties and large land parcels may legitimately have very high prices.                                   |
| Investigate extreme positive areas          | Large properties and land listings may be legitimate, so size alone is not enough to justify deletion.             |
| Apply an IQR rule only if explicitly chosen | The interquartile range rule can flag extreme observations, but flagged observations are not automatically errors. |

**Important:** An IQR-based rule is a statistical screening method, not proof that a value is incorrect. If no IQR rule was applied to remove rows in the final preprocessing code, describe it as an investigation method rather than claiming that outliers were removed by IQR.

### Step 11: Prepare location information

Location can strongly influence property prices.

Depending on availability, location features may include:

* Locality name.
* Latitude.
* Longitude.
* Grouped locality category.

Coordinates can help represent the geographical position of a listing.

If coordinates are obtained by geocoding a locality name, they represent an approximate locality position, not necessarily the exact location of the property. This limitation should be acknowledged in the project report.

Rare locality names may also be grouped into an `"Other"` category to reduce the number of distinct categories.

Missing locality values should be handled separately, for example by assigning `"Unknown"`, rather than automatically treating missing values as rare known localities.

### Step 12: Separate analysis data from modelling data

Exploratory data analysis and machine learning have different requirements.

For analysis, it can be useful to retain descriptive fields and price-related columns so that relationships can be investigated.

For modelling, only features appropriate for predicting the target should be used.

The target variable is:

`price_lakhs`

Possible input features include:

* `property_type`
* `bedroom_num`
* `bathroom_num`
* `balcony_num`
* `carpet_sqft`
* `builtup_sqft`
* `superbuiltup_sqft`
* `age`
* `locality_grouped`
* `latitude`
* `longitude`

The final feature list must include only columns that actually exist in the processed dataset and are appropriate for the prediction task.

### Step 13: Prevent target leakage

Target leakage occurs when a model receives information that would not legitimately be available as an independent input at prediction time.

For example, if the target is the sale price, a feature calculated directly from that price can reveal the answer to the model.

Potential leakage columns include:

* `price`
* `price_per_sqft`
* `price_per_unit_area`
* `price_sqft`
* `min_price`
* `max_price`
* `log_price`
* Other fields directly derived from the target price

These columns should not be used as input features when they reveal the target.

The same principle applies to transformations of the target: a transformed target may be used as the value the model learns to predict, but it must not also be supplied as an input feature.

The original price information may be retained separately for exploratory analysis where appropriate.

---

## 7. Why Encoding and Scaling Are Deferred to the Machine Learning Pipeline

Categorical encoding and feature scaling are handled during model training rather than being fitted on the entire dataset during preprocessing.

This is an important step for preventing data leakage.

### 7.1 Categorical encoding

Categorical columns contain labels rather than directly usable numerical measurements.

Examples include:

* `property_type`
* `locality_grouped`

One-hot encoding can transform a category into numerical indicator columns.

For example:

| Property type     | Apartment | Independent House | Villa |
| ----------------- | --------: | ----------------: | ----: |
| Apartment         |         1 |                 0 |     0 |
| Independent House |         0 |                 1 |     0 |
| Villa             |         0 |                 0 |     1 |

The actual columns depend on the categories present in the training data.

**Why encoding is deferred:**

1. The dataset is divided into training and testing sets.
2. The encoder is fitted using the training data only.
3. The fitted encoder transforms both the training and test data.
4. Previously unseen categories can be handled using an appropriate setting, such as `handle_unknown="ignore"` in OneHotEncoder.

If the encoder is fitted before splitting the data, information about categories in the test set can influence preprocessing. Although this does not directly reveal the target, fitting preprocessing on training data only is the correct evaluation practice.

### 7.2 Feature scaling

Numerical features have different ranges.

For example:

* Bedroom count may range from 1 to 10.
* Area may range from hundreds to thousands of square feet.
* Latitude and longitude have their own numerical ranges.

StandardScaler transforms a numerical feature using its training-set mean and standard deviation.

The transformed value is:

`z = (x - mean) / standard_deviation`

**Why scaling is deferred:**

1. The dataset is split into training and testing sets.
2. The scaler learns its parameters from the training set.
3. The same fitted scaler transforms the training and test features.
4. The test set does not influence the parameters used during training.

Linear Regression often benefits from scaling for numerical stability and consistent feature magnitudes, especially when regularization is used. Ordinary Linear Regression without regularization does not strictly require feature scaling for the same reasons as distance-based algorithms.

Decision Trees, Random Forests, and Gradient Boosting based on decision trees generally do not require standardization because their splits are based on feature thresholds rather than comparable feature magnitudes.

### 7.3 Using a pipeline

A machine learning pipeline combines preprocessing and model training into a single reproducible workflow.

A typical pipeline can include:

* Imputation of missing numerical values.
* Imputation of missing categorical values.
* One-hot encoding of categorical features.
* Scaling of numerical features where appropriate.
* The machine learning estimator.

This approach reduces the chance of applying inconsistent transformations and helps prevent data leakage during model evaluation.

**Conclusion:** Encoding and scaling are not omitted. They are deferred until the modelling stage, where they can be fitted correctly using training data only.

---

## 8. Row-Count Log

A row-count log records how many observations remain after each major preprocessing operation.

This makes the data-cleaning process transparent and helps identify where rows were removed.

The known starting point for the original dataset was 10,704 rows.

An earlier check also found 9,752 sale listings after filtering. The final row count must be regenerated from the exact final notebook or script because subsequent cleaning steps may change it.

| Stage                                 |  Number of rows | Explanation                                         |
| ------------------------------------- | --------------: | --------------------------------------------------- |
| Raw dataset                           |          10,704 | Original dataset size observed during inspection    |
| After duplicate removal               | To be generated | Rows remaining after exact duplicate removal        |
| After price conversion and validation | To be generated | Records with valid positive numerical prices        |
| After filtering sale listings         | To be generated | Records retained for the sale-price prediction task |
| After invalid target removal          | To be generated | Records with valid target values                    |
| After any additional row filters      | To be generated | Depends on the final preprocessing rules            |
| Final processed dataset               | To be generated | Actual output from the final preprocessing run      |

The table should be updated with the actual counts produced by the final preprocessing workflow. Do not use counts from different versions of the notebook as though they came from the same run.

The number of rows removed at each stage should be calculated as:

`Rows removed = Rows before the step - Rows after the step`

The retention percentage is:

`Retention percentage = (Final rows / Raw rows) × 100`

A lower row count is not automatically bad. Removing invalid or irrelevant records may improve data quality. The goal is to remove only observations that have a defensible reason for exclusion.

---

## 9. Before-and-After Boxplots

Boxplots are useful for visually comparing the distribution of numerical values before and after a cleaning rule.

For this project, boxplots can be created for:

* Property sale price.
* Carpet area.
* Built-up area.
* Super built-up area.

A boxplot shows the median, the interquartile range, and observations beyond the whiskers.

Points beyond the whiskers are potential outliers, not automatically invalid records.

Before-and-after plots should use the same units and comparable axis limits wherever possible.

### Interpretation

* A narrower distribution after a cleaning rule may indicate that extreme observations were removed.
* A change in the median may indicate that the removed records affected the centre of the distribution.
* A large change in the number of observations should be explained by the specific rule applied.
* A log-scaled view may help visualize highly skewed prices, but the axis transformation must be clearly labelled.

**Important:** If the preprocessing workflow did not remove price or area outliers, the boxplots should be described as diagnostic visualizations, not evidence that outliers were removed.

---

## 10. Missing-Value Strategy

Not every missing value should result in a row being deleted.

For example, a property listing may have a valid sale price and bedroom count but no reported carpet area.

Removing all rows with any missing feature could discard many potentially useful listings.

The strategy used in modelling is to impute missing feature values within the training pipeline where appropriate.

Typical choices include:

| Feature type         | Possible strategy                                    | Reason                                                       |
| -------------------- | ---------------------------------------------------- | ------------------------------------------------------------ |
| Numerical features   | Median imputation                                    | The median is less affected by extreme values than the mean. |
| Categorical features | Most-frequent imputation                             | Replaces missing categories with a common observed category. |
| Target variable      | Remove records with invalid or missing target values | A supervised training observation needs a valid target.      |
| Location             | Use an `"Unknown"` category where appropriate        | Preserves records whose locality information is missing.     |

The appropriate strategy depends on the final feature list and the missingness patterns in the data.

Imputation parameters should be learned from the training set, not from the full dataset before splitting.

---

## 11. Saved Outputs

The preprocessing workflow may generate the following files, depending on the final script and notebook configuration.

| File                                | Purpose                                                  |
| ----------------------------------- | -------------------------------------------------------- |
| `data/raw/housing_raw.csv`          | Original raw dataset                                     |
| `data/raw/housing_with_coords.csv`  | Dataset with geocoded locality coordinates, if used      |
| `data/processed/clean.csv`          | Processed dataset saved by the preprocessing script      |
| `data/processed/clean_with_pps.csv` | Additional processed dataset, if generated by the script |
| `notebooks/02_preprocessing.ipynb`  | Interactive preprocessing and analysis notebook          |
| `src/preprocess.py`                 | Reusable preprocessing script                            |

The actual output files should be checked before listing them as generated deliverables.

---

## 12. Limitations

The preprocessing process has several limitations.

1. **Source-data accuracy:** Incorrect values in the original listings may remain if they cannot be identified reliably.
2. **Price interpretation:** Price conversion depends on correctly interpreting units and source conventions.
3. **Missing information:** Some listings do not contain all property attributes.
4. **Location precision:** Coordinates obtained from locality names may represent approximate area locations rather than exact property locations.
5. **Outlier ambiguity:** Extremely high prices or large areas may represent legitimate properties, so statistical extremity alone is not sufficient justification for removal.
6. **Market variation:** Property prices can vary by locality, market conditions, amenities, property type, and time.
7. **Temporal limitations:** Historical listing data alone does not guarantee accurate prediction of future market prices. A genuine future-price forecasting task also requires appropriate time-based features, historical observations, and evaluation on later periods.

These limitations should be considered when interpreting the model's predictions.

---

## 13. Conclusion

The preprocessing module converts raw real estate listings into a more consistent dataset for analysis and machine learning.

The main tasks include standardizing column names, cleaning text, removing exact duplicates, converting property prices into lakhs, retaining sale listings, validating the target variable, examining property attributes, preparing location information, and separating suitable model features from price-derived or irrelevant columns.

Categorical encoding, missing-value imputation, and feature scaling are handled within the modelling pipeline so that transformations can be fitted on training data and applied consistently to unseen observations.

The resulting dataset provides the foundation for exploratory data analysis, feature engineering, model training, and evaluation.

The quality of the final model depends not only on the algorithm selected but also on the accuracy of the original data, the validity of the preprocessing rules, and the appropriateness of the evaluation strategy.

