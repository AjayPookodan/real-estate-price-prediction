from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd


def parse_price_to_lakhs(value):
    """
    Convert property prices to lakhs.

    Examples:
        '2.63 Cr' -> 263.0
        '85 L'    -> 85.0
        '40,000'  -> 0.4

    Assumes unlabelled numeric prices are in rupees.
    Verify this assumption against the source dataset.
    """
    if pd.isna(value):
        return np.nan

    text = str(value).strip().replace(",", "").replace("₹", "")

    if not text:
        return np.nan

    match = re.fullmatch(
        r"([+-]?\d+(?:\.\d+)?)\s*(cr|crore|crores|l|lac|lakh|lakhs)?",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return np.nan

    amount = float(match.group(1))
    unit = (match.group(2) or "").lower()

    if unit in {"cr", "crore", "crores"}:
        return amount * 100

    if unit in {"l", "lac", "lakh", "lakhs"}:
        return amount

    # Unlabelled values are assumed to be rupees.
    return amount / 100_000


def preprocess(raw_path, out_path):
    """
    Clean the raw real-estate dataset and save:
      1. Model-ready data
      2. EDA data

    Parameters
    ----------
    raw_path : str or Path
        Path to housing_with_coords.csv or another supported source CSV.

    out_path : str or Path
        Path where clean.csv should be saved.

    Returns
    -------
    pandas.DataFrame
        Cleaned sale listings with a price_lakhs target.
    """
    raw_path = Path(raw_path).resolve()
    out_path = Path(out_path).resolve()

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {raw_path}"
        )

    print(f"Loading dataset: {raw_path}")
    df_raw = pd.read_csv(raw_path, low_memory=False)

    print("Raw shape:", df_raw.shape)

    # --------------------------------------------------
    # Part B: Standardize column names and text
    # --------------------------------------------------
    df = df_raw.copy()

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[\s\-]+", "_", regex=True)
    )

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip()

    # --------------------------------------------------
    # Part C: Remove exact duplicate rows
    # --------------------------------------------------
    duplicates_before = df.duplicated().sum()
    df = df.drop_duplicates().copy()

    print("Exact duplicate rows removed:", duplicates_before)

    # --------------------------------------------------
    # Part D: Convert target price to lakhs
    # --------------------------------------------------
    if "price_lakhs" in df.columns:
        df["price_lakhs"] = pd.to_numeric(
            df["price_lakhs"], errors="coerce"
        )
    elif "price" in df.columns:
        df["price_lakhs"] = df["price"].apply(parse_price_to_lakhs)
    else:
        raise ValueError(
            "No 'price' or 'price_lakhs' column was found."
        )

    # --------------------------------------------------
    # Part E: Keep sale listings only
    # --------------------------------------------------
    if "preference" not in df.columns:
        raise ValueError(
            "The 'preference' column is missing. "
            "Cannot reliably separate sales from rentals."
        )

    df["preference"] = (
        df["preference"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    before_sale_filter = len(df)
    df = df[df["preference"] == "S"].copy()

    print("Rows before sale filter:", before_sale_filter)
    print("Sale listings:", len(df))

    # --------------------------------------------------
    # Part F: Validate target prices
    # --------------------------------------------------
    df = df[
        df["price_lakhs"].notna()
        & np.isfinite(df["price_lakhs"])
        & (df["price_lakhs"] > 0)
    ].copy()

    if df.empty:
        raise ValueError(
            "No valid sale prices remain. "
            "Check price conversion and sale filtering."
        )

    print("Rows after target validation:", len(df))

    # --------------------------------------------------
    # Part G: Convert numeric features
    # --------------------------------------------------
    numeric_candidates = [
        "bedroom_num",
        "bathroom_num",
        "balcony_num",
        "carpet_sqft",
        "builtup_sqft",
        "superbuiltup_sqft",
        "age",
        "latitude",
        "longitude",
    ]

    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Missing areas are retained; the training pipeline
    # will impute them without using the target.
    area_columns = [
        "carpet_sqft",
        "builtup_sqft",
        "superbuiltup_sqft",
    ]

    for col in area_columns:
        if col in df.columns:
            # Non-positive areas are invalid measurements.
            df.loc[df[col] <= 0, col] = np.nan

    # --------------------------------------------------
    # Part H: Group rare localities
    # --------------------------------------------------
    if "locality" in df.columns:
        location_col = "locality"
    elif "location" in df.columns:
        location_col = "location"
    else:
        location_col = None

    if location_col:
        df[location_col] = (
            df[location_col]
            .astype("string")
            .str.strip()
        )

        df.loc[
            df[location_col].isin(["", "<NA>", "nan", "None"]),
            location_col,
        ] = pd.NA

        counts = df[location_col].value_counts(dropna=True)

        common_locations = counts[counts >= 10].index

        df["locality_grouped"] = df[location_col].where(
            df[location_col].isin(common_locations),
            "Other",
        )

        df["locality_grouped"] = (
            df["locality_grouped"]
            .fillna("Unknown")
        )

    elif "locality_grouped" not in df.columns:
        df["locality_grouped"] = "Unknown"

    # --------------------------------------------------
    # Part I: Remove target leakage/helper columns
    # --------------------------------------------------
    df_eda = df.copy()

    leakage_columns = [
        "price",
        "price_per_sqft",
        "price_per_unit_area",
        "price_sqft",
        "min_price",
        "max_price",
        "log_price",
        "log_price_target",
    ]

    df_model = df.drop(
        columns=[
            col for col in leakage_columns
            if col in df.columns
        ]
    ).copy()

    # Listing metadata and transaction identifiers are not
    # intended as model features. They remain in the EDA copy.
    metadata_columns = [
        "preference",
        "transact_type",
        "spid",
        "prop_id",
        "description",
        "features",
        "register_date",
        "posting_date",
        "update_date",
    ]

    df_model = df_model.drop(
        columns=[
            col for col in metadata_columns
            if col in df_model.columns
        ]
    ).copy()

    # --------------------------------------------------
    # Part J: Final checks
    # --------------------------------------------------
    if df_model["price_lakhs"].isna().any():
        raise ValueError("Missing target prices remain.")

    if (df_model["price_lakhs"] <= 0).any():
        raise ValueError("Non-positive target prices remain.")

    print("\nFinal checks")
    print("Model dataset shape:", df_model.shape)
    print("EDA dataset shape:", df_eda.shape)
    print("Missing target values:", df_model["price_lakhs"].isna().sum())
    print("Duplicate model rows:", df_model.duplicated().sum())

    # --------------------------------------------------
    # Part K: Save processed data
    # --------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)

    eda_path = out_path.parent / "clean_with_pps.csv"

    df_model.to_csv(out_path, index=False)
    df_eda.to_csv(eda_path, index=False)

    retention = len(df_model) / len(df_raw) * 100

    print("\nPreprocessing complete!")
    print("Model data saved to:", out_path)
    print("EDA data saved to:", eda_path)
    print(f"Rows retained compared with raw input: {retention:.1f}%")

    return df_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess the real-estate dataset."
    )

    parser.add_argument(
        "--input",
        default="data/raw/housing_with_coords.csv",
        help="Path to the input CSV.",
    )

    parser.add_argument(
        "--output",
        default="data/processed/clean.csv",
        help="Path for the model-ready CSV.",
    )

    args = parser.parse_args()

    preprocess(args.input, args.output)

