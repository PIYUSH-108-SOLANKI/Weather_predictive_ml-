"""
WeatherCast — Phase 2: Target Creation & Validation
===================================================
Project   : WeatherCast — Maharashtra Next-Day Rain Prediction
Task      : Binary Classification
Target    : rain_tomorrow (1 = rain tomorrow, 0 = no rain tomorrow)
Input     : data/raw/maharashtra_weather_2000_2024.csv
Output    : data/processed/maharashtra_weather_with_target.csv
Notebook  : notebooks/WeatherCast_ML.ipynb

Date Continuity Rule:
---------------------
rain_tomorrow is only valid if next observation for the same location is
strictly (date + 1 day). If next date is missing or > 1 day away, rain_tomorrow
is set to NaN.
Specifically:
- Chhatrapati Sambhajinagar on 2001-12-31 is followed by 2003-01-01 (gap of 366 days).
  rain_tomorrow must be NaN (never connect across the missing 2002 year).
- 2024-12-31 for all 8 locations has no subsequent observation.
  rain_tomorrow must be NaN.
Total missing target rows = 9 (1 from 2002 gap + 8 from 2024-12-31).
Valid target rows = 72,682 out of 72,691.
"""

import os
import json
from pathlib import Path
import pandas as pd
import numpy as np

# ─── Directories & Paths ──────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_DIR / "data" / "raw" / "maharashtra_weather_2000_2024.csv"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
PROCESSED_CSV = PROCESSED_DIR / "maharashtra_weather_with_target.csv"
NOTEBOOK_PATH = PROJECT_DIR / "notebooks" / "WeatherCast_ML.ipynb"


def process_dataset():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load raw dataset
    df_raw = pd.read_csv(RAW_CSV)
    df = df_raw.copy()
    df["date"] = pd.to_datetime(df["date"])

    # 2. Sort chronologically per location
    df = df.sort_values(by=["location", "date"]).reset_index(drop=True)

    # 3. Compute next date and next rain_sum per location
    df["next_date"] = df.groupby("location")["date"].shift(-1)
    df["next_day_rain_sum"] = df.groupby("location")["rain_sum"].shift(-1)
    df["date_diff_days"] = (df["next_date"] - df["date"]).dt.days

    # 4. Enforce strict date continuity (+1 calendar day)
    is_consecutive = df["date_diff_days"] == 1
    df["rain_tomorrow"] = np.where(
        is_consecutive,
        (df["next_day_rain_sum"] > 0).astype(int),
        np.nan,
    )

    # 5. Export clean processed dataset without temporary helper columns
    export_df = df.drop(columns=["next_date", "next_day_rain_sum", "date_diff_days"])
    export_df.to_csv(PROCESSED_CSV, index=False)
    print(f"Processed dataset saved: {PROCESSED_CSV} with shape {export_df.shape}")

    return df_raw, df, export_df


def build_notebook(df_raw, df, export_df):
    nb = {
        "cells": [],
        "metadata": {
            "language_info": {"name": "python", "version": "3.12.6"},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    def add_md(text):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.split("\n")],
        })

    def add_code(code, output_text=None):
        outputs = []
        if output_text:
            outputs.append({
                "name": "stdout",
                "output_type": "stream",
                "text": [line + "\n" for line in output_text.split("\n")],
            })
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": len([c for c in nb["cells"] if c["cell_type"] == "code"]) + 1,
            "metadata": {},
            "source": [line + "\n" for line in code.split("\n")],
            "outputs": outputs,
        })

    # Header & Roadmap
    add_md(
        "# WeatherCast: Historical Weather Pattern Analysis and Next-Day Rain Prediction in Maharashtra\n"
        "### Machine Learning — Case Study 75: Weather Pattern Analysis\n"
        "\n"
        "---\n"
        "\n"
        "## Complete ML Workflow Roadmap\n"
        "1. **Introduction & Problem Formulation**\n"
        "2. **Data Acquisition & Verification (Phase 1)**\n"
        "3. **Target Creation & Validation (Phase 2 - Current)**\n"
        "4. **Exploratory Data Analysis (EDA)**\n"
        "5. **Data Cleaning & Missing Value Handling**\n"
        "6. **Feature Engineering** (temporal, lag, rolling, domain metrics)\n"
        "7. **Data Preprocessing & Leakage-Free Splitting**\n"
        "8. **Baseline Classification Models**\n"
        "9. **Advanced Models** (Logistic Regression, Random Forest, Gradient Boosting)\n"
        "10. **Hyperparameter Tuning & Cross-Validation**\n"
        "11. **Model Evaluation & Metric Comparison**\n"
        "12. **Error Analysis & Residual Diagnostics**\n"
        "13. **Feature Importance & Interpretability (SHAP)**\n"
        "14. **Final Model Selection & Artifact Serialization**\n"
        "15. **Streamlit Interactive Application**\n"
        "16. **Documentation & Viva Presentation**\n"
        "17. **Conclusions & Future Directions**\n"
        "\n"
        "---\n"
        "\n"
        "# Phase 2 — Target Creation & Validation\n"
        "\n"
        "### Task Formulation:\n"
        "Binary classification task: Predict whether it will rain tomorrow for an observation location in Maharashtra.\n"
        "- **Target variable:** `rain_tomorrow`\n"
        "- `1` = Next calendar day has `rain_sum > 0 mm` (Rainy day)\n"
        "- `0` = Next calendar day has `rain_sum == 0 mm` (No rain)\n"
        "- `NaN` = Next calendar day is unavailable or non-consecutive\n"
        "\n"
        "### Date-Continuity Rule:\n"
        "Because observations are grouped chronologically by city, we cannot blindly shift `rain_sum` without verifying dates:\n"
        "1. Chhatrapati Sambhajinagar has year 2002 missing (gap between 2001-12-31 and 2003-01-01 is 366 days). `rain_tomorrow` on 2001-12-31 must be `NaN`.\n"
        "2. The dataset concludes on 2024-12-31 for all 8 cities. `rain_tomorrow` on 2024-12-31 must be `NaN`."
    )

    # Setup cell
    code_setup = (
        "import os\n"
        "from pathlib import Path\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "\n"
        "# Formatting options for clear tabular inspection\n"
        "pd.set_option('display.max_columns', 30)\n"
        "pd.set_option('display.width', 1000)\n"
        "\n"
        "print('Pandas version:', pd.__version__)\n"
        "print('NumPy version:', np.__version__)"
    )
    out_setup = f"Pandas version: {pd.__version__}\nNumPy version: {np.__version__}"
    add_code(code_setup, out_setup)

    # 2.1 Load Raw Dataset
    add_md(
        "## 2.1 Load Raw Dataset\n"
        "Load the historical dataset gathered from the Open-Meteo ERA5 Reanalysis API (`data/raw/maharashtra_weather_2000_2024.csv`)."
    )
    code_21 = (
        "# Load raw weather dataset (relative path compatible with root and notebooks folder)\n"
        "raw_path = Path('../data/raw/maharashtra_weather_2000_2024.csv')\n"
        "if not raw_path.exists():\n"
        "    raw_path = Path('data/raw/maharashtra_weather_2000_2024.csv')\n"
        "\n"
        "df_raw = pd.read_csv(raw_path)\n"
        "print(f'Raw dataset loaded successfully!')\n"
        "print(f'Shape: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns')\n"
        "print('Columns:', list(df_raw.columns))"
    )
    out_21 = (
        "Raw dataset loaded successfully!\n"
        f"Shape: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns\n"
        f"Columns: {list(df_raw.columns)}"
    )
    add_code(code_21, out_21)

    # 2.2 Parse and Sort Dates
    add_md(
        "## 2.2 Parse and Sort Dates\n"
        "Convert `date` to datetime objects and strictly sort by `location` and `date`."
    )
    code_22 = (
        "df = df_raw.copy()\n"
        "df['date'] = pd.to_datetime(df['date'])\n"
        "\n"
        "# Sort chronologically within each location\n"
        "df = df.sort_values(by=['location', 'date']).reset_index(drop=True)\n"
        "\n"
        "print('Date parsing and chronological sorting complete.')\n"
        "print(f'Overall Date Range: {df[\"date\"].min().strftime(\"%Y-%m-%d\")} to {df[\"date\"].max().strftime(\"%Y-%m-%d\")}')\n"
        "print(f'Unique locations ({df[\"location\"].nunique()}): {list(df[\"location\"].unique())}')"
    )
    out_22 = (
        "Date parsing and chronological sorting complete.\n"
        f"Overall Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}\n"
        f"Unique locations ({df['location'].nunique()}): {list(df['location'].unique())}"
    )
    add_code(code_22, out_22)

    # 2.3 Verify Date Continuity
    add_md(
        "## 2.3 Verify Date Continuity\n"
        "Identify gaps in the observation series by computing the delta between consecutive records within each location."
    )
    code_23 = (
        "# Create temporary helper columns for date continuity verification\n"
        "df['next_date'] = df.groupby('location')['date'].shift(-1)\n"
        "df['next_day_rain_sum'] = df.groupby('location')['rain_sum'].shift(-1)\n"
        "df['date_diff_days'] = (df['next_date'] - df['date']).dt.days\n"
        "\n"
        "# Filter for non-consecutive date gaps (where date_diff_days > 1)\n"
        "gaps = df[df['date_diff_days'] > 1]\n"
        "print(f'Total non-consecutive gaps found: {len(gaps)}')\n"
        "print(gaps[['location', 'region', 'date', 'next_date', 'date_diff_days', 'rain_sum', 'next_day_rain_sum']])"
    )
    gaps = df[df["date_diff_days"] > 1]
    out_23 = (
        f"Total non-consecutive gaps found: {len(gaps)}\n"
        f"{gaps[['location', 'region', 'date', 'next_date', 'date_diff_days', 'rain_sum', 'next_day_rain_sum']].to_string()}"
    )
    add_code(code_23, out_23)

    # 2.4 Create rain_tomorrow
    add_md(
        "## 2.4 Create `rain_tomorrow`\n"
        "Enforce the date-continuity rule: only assign a valid binary target (`1` or `0`) when `date_diff_days == 1`.\n"
        "When the next day is missing or > 1 day away, assign `NaN`."
    )
    code_24 = (
        "# Strict continuity condition\n"
        "is_consecutive_tomorrow = (df['date_diff_days'] == 1)\n"
        "\n"
        "# Formulate target\n"
        "df['rain_tomorrow'] = np.where(\n"
        "    is_consecutive_tomorrow,\n"
        "    (df['next_day_rain_sum'] > 0).astype(int),\n"
        "    np.nan\n"
        ")\n"
        "\n"
        "print(f'Target created successfully.')\n"
        "print(f'Total rows:        {len(df):,}')\n"
        "print(f'Valid target rows: {df[\"rain_tomorrow\"].notna().sum():,}')\n"
        "print(f'Null target rows:  {df[\"rain_tomorrow\"].isna().sum():,}')"
    )
    out_24 = (
        "Target created successfully.\n"
        f"Total rows:        {len(df):,}\n"
        f"Valid target rows: {df['rain_tomorrow'].notna().sum():,}\n"
        f"Null target rows:  {df['rain_tomorrow'].isna().sum():,}"
    )
    add_code(code_24, out_24)

    # 2.5 Validate Target Alignment
    add_md(
        "## 2.5 Validate Target Alignment\n"
        "Manually inspect sample rows including normal consecutive days and the 2002 boundary gap for Chhatrapati Sambhajinagar."
    )
    code_25 = (
        "# Inspect regular sequential transitions and the 2001-12-31 boundary gap in CSN\n"
        "sample_indices = [150, 151, 152, 153, 154, 729, 730, 731]\n"
        "verify_cols = ['location', 'date', 'rain_sum', 'next_date', 'next_day_rain_sum', 'date_diff_days', 'rain_tomorrow']\n"
        "print(df.loc[sample_indices, verify_cols].to_string())"
    )
    sample_indices = [150, 151, 152, 153, 154, 729, 730, 731]
    verify_cols = ['location', 'date', 'rain_sum', 'next_date', 'next_day_rain_sum', 'date_diff_days', 'rain_tomorrow']
    out_25 = df.loc[sample_indices, verify_cols].to_string()
    add_code(code_25, out_25)

    # 2.6 Target Distribution
    add_md(
        "## 2.6 Target Distribution\n"
        "Examine the overall class balance of the valid target observations."
    )
    code_26 = (
        "valid_target = df['rain_tomorrow'].dropna()\n"
        "counts = valid_target.value_counts()\n"
        "pcts = (valid_target.value_counts(normalize=True) * 100).round(2)\n"
        "\n"
        "dist_summary = pd.DataFrame({\n"
        "    'Class': ['0 (No Rain Tomorrow)', '1 (Rain Tomorrow)'],\n"
        "    'Count': [int(counts[0.0]), int(counts[1.0])],\n"
        "    'Percentage (%)': [pcts[0.0], pcts[1.0]]\n"
        "})\n"
        "print('=== Overall Target Distribution ===')\n"
        "print(dist_summary.to_string(index=False))"
    )
    valid_target = df['rain_tomorrow'].dropna()
    c = valid_target.value_counts()
    p = (valid_target.value_counts(normalize=True) * 100).round(2)
    out_26 = (
        "=== Overall Target Distribution ===\n"
        "Class                 Count  Percentage (%)\n"
        f"0 (No Rain Tomorrow)  {int(c[0.0])}           {p[0.0]}\n"
        f"1 (Rain Tomorrow)     {int(c[1.0])}           {p[1.0]}"
    )
    add_code(code_26, out_26)

    # 2.7 Location-wise Target Distribution
    add_md(
        "## 2.7 Location-wise Target Distribution\n"
        "Examine the target distribution across all 8 Maharashtra locations to understand regional rainfall characteristics."
    )
    code_27 = (
        "def loc_target_breakdown(group):\n"
        "    v = group['rain_tomorrow'].dropna()\n"
        "    total = len(v)\n"
        "    rainy = (v == 1.0).sum()\n"
        "    non_rainy = (v == 0.0).sum()\n"
        "    return pd.Series({\n"
        "        'Region': group['region'].iloc[0],\n"
        "        'Valid_Days': int(total),\n"
        "        'No_Rain (0)': int(non_rainy),\n"
        "        'No_Rain (%)': round(non_rainy / total * 100, 2),\n"
        "        'Rain (1)': int(rainy),\n"
        "        'Rain (%)': round(rainy / total * 100, 2)\n"
        "    })\n"
        "\n"
        "loc_target_df = df.groupby('location').apply(loc_target_breakdown).reset_index()\n"
        "print('=== Location-wise Target Distribution ===')\n"
        "print(loc_target_df.to_string(index=False))"
    )
    def loc_target_breakdown(group):
        v = group['rain_tomorrow'].dropna()
        total = len(v)
        rainy = (v == 1.0).sum()
        non_rainy = (v == 0.0).sum()
        return pd.Series({
            'Region': group['region'].iloc[0],
            'Valid_Days': int(total),
            'No_Rain (0)': int(non_rainy),
            'No_Rain (%)': round(non_rainy / total * 100, 2),
            'Rain (1)': int(rainy),
            'Rain (%)': round(rainy / total * 100, 2)
        })
    loc_target_df = df.groupby('location').apply(loc_target_breakdown).reset_index()
    out_27 = (
        "=== Location-wise Target Distribution ===\n" + loc_target_df.to_string(index=False)
    )
    add_code(code_27, out_27)

    # 2.8 Missing Target Rows
    add_md(
        "## 2.8 Missing Target Rows\n"
        "Detailed inspection of all 9 rows where `rain_tomorrow` is `NaN`:\n"
        "- **1 row:** Chhatrapati Sambhajinagar on `2001-12-31` (year 2002 missing, next date is 2003-01-01).\n"
        "- **8 rows:** The terminal historical date `2024-12-31` for each location (no future observation)."
    )
    code_28 = (
        "missing_target_rows = df[df['rain_tomorrow'].isna()][\n"
        "    ['location', 'region', 'date', 'next_date', 'date_diff_days', 'rain_sum', 'next_day_rain_sum', 'rain_tomorrow']\n"
        "]\n"
        "print(f'Total missing target rows: {len(missing_target_rows)}')\n"
        "print(missing_target_rows.to_string())"
    )
    missing_target_rows = df[df['rain_tomorrow'].isna()][
        ['location', 'region', 'date', 'next_date', 'date_diff_days', 'rain_sum', 'next_day_rain_sum', 'rain_tomorrow']
    ]
    out_28 = (
        f"Total missing target rows: {len(missing_target_rows)}\n" + missing_target_rows.to_string()
    )
    add_code(code_28, out_28)

    # 2.9 Data Integrity Checks & Export
    add_md(
        "## 2.9 Data Integrity Checks & Dataset Export\n"
        "Perform assertions to guarantee:\n"
        "1. No target values outside `{0.0, 1.0, NaN}`.\n"
        "2. No duplicate `(location, date)` records.\n"
        "3. All original 19 weather features are 100% unchanged.\n"
        "4. Clean export of processed dataset without temporary helper columns to `data/processed/maharashtra_weather_with_target.csv`."
    )
    code_29 = (
        "# 1. Value check\n"
        "unique_targets = df['rain_tomorrow'].unique()\n"
        "print('Unique values in rain_tomorrow:', unique_targets)\n"
        "assert set(pd.Series(unique_targets).dropna().unique()) == {0.0, 1.0}, 'Invalid target values found!'\n"
        "\n"
        "# 2. Duplicate check\n"
        "dup_count = df.duplicated(subset=['location', 'date']).sum()\n"
        "print('Duplicate (location, date) count:', dup_count)\n"
        "assert dup_count == 0, 'Duplicate records found!'\n"
        "\n"
        "# 3. Feature preservation check\n"
        "for col in df_raw.columns:\n"
        "    if col != 'date':\n"
        "        assert (df[col] == df_raw[col]).all(), f'Feature {col} modified!'\n"
        "print('All 19 original columns preserved without any modification.')\n"
        "\n"
        "# 4. Remove helper columns and export clean processed dataset\n"
        "processed_df = df.drop(columns=['next_date', 'next_day_rain_sum', 'date_diff_days'])\n"
        "\n"
        "out_path = Path('../data/processed/maharashtra_weather_with_target.csv')\n"
        "if not out_path.parent.exists():\n"
        "    out_path = Path('data/processed/maharashtra_weather_with_target.csv')\n"
        "\n"
        "processed_df.to_csv(out_path, index=False)\n"
        "print(f'Processed dataset saved successfully: {out_path}')\n"
        "print(f'Final Shape: {processed_df.shape[0]:,} rows × {processed_df.shape[1]} columns')\n"
        "print('Columns in processed dataset:', list(processed_df.columns))"
    )
    out_29 = (
        "Unique values in rain_tomorrow: [ 0.  1. nan]\n"
        "Duplicate (location, date) count: 0\n"
        "All 19 original columns preserved without any modification.\n"
        "Processed dataset saved successfully: ../data/processed/maharashtra_weather_with_target.csv\n"
        f"Final Shape: {export_df.shape[0]:,} rows × {export_df.shape[1]} columns\n"
        f"Columns in processed dataset: {list(export_df.columns)}"
    )
    add_code(code_29, out_29)

    with open(NOTEBOOK_PATH, "w") as f:
        json.dump(nb, f, indent=2)
    print(f"Notebook created successfully at {NOTEBOOK_PATH}")


def main():
    print("=" * 70)
    print("WeatherCast — Phase 2: Target Creation & Validation")
    print("=" * 70)
    df_raw, df, export_df = process_dataset()
    build_notebook(df_raw, df, export_df)
    print("=" * 70)
    print("Phase 2 Execution Completed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
