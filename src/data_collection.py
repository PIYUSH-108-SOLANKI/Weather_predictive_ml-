"""
WeatherCast — Phase 1: Data Collection
========================================
Project   : WeatherCast — Maharashtra Next-Day Rain Prediction
Dataset   : Open-Meteo Historical Weather API (ERA5 reanalysis)
Period    : 2000-01-01 to 2024-12-31
Locations : 8 representative Maharashtra cities
Output    : data/raw/maharashtra_weather_2000_2024.csv

Model selection rationale
--------------------------
ERA5 (ECMWF Reanalysis v5) is chosen over ERA5-Land and IFS because:
  * ERA5 covers the full target period 2000–2024 without gaps.
  * ERA5-Land starts from 1950 but has no pressure_msl variable.
  * IFS (9 km) only starts from 2017, so it cannot cover 2000–2016.
  * ERA5 at 0.25° provides globally consistent, peer-reviewed reanalysis
    data that is the de-facto standard for long-period ML studies.

API endpoint : https://archive-api.open-meteo.com/v1/archive
"""

import os
import sys
import time
import logging
import requests
import pandas as pd
from pathlib import Path
from datetime import date

# ─── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Project paths ────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent          # WeatherCast/src/
PROJECT_DIR = SCRIPT_DIR.parent                         # WeatherCast/
RAW_DIR     = PROJECT_DIR / "data" / "raw"
OUTPUT_CSV  = RAW_DIR / "maharashtra_weather_2000_2024.csv"

# ─── Global parameters ────────────────────────────────────────────────────────
START_DATE  = "2000-01-01"
END_DATE    = "2024-12-31"
MODEL       = "era5"           # ERA5 reanalysis — see module docstring
TIMEZONE    = "Asia/Kolkata"
API_URL     = "https://archive-api.open-meteo.com/v1/archive"

# Chunk by year to keep individual API responses safe and retryable
CHUNK_YEARS = 1

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 10   # seconds

# ─── Locations & region mapping ───────────────────────────────────────────────
# Coordinates verified via Open-Meteo Geocoding API on 2026-09-30.
# Geocoding query: https://geocoding-api.open-meteo.com/v1/search?name=<city>
LOCATIONS = [
    {
        "location": "Mumbai",
        "region":   "Konkan",
        "latitude":  19.07283,
        "longitude": 72.88261,
    },
    {
        "location": "Ratnagiri",
        "region":   "Konkan",
        "latitude":  16.99154,
        "longitude": 73.31022,
    },
    {
        "location": "Pune",
        "region":   "Madhya Maharashtra",
        "latitude":  18.51957,
        "longitude": 73.85535,
    },
    {
        "location": "Nashik",
        "region":   "Madhya Maharashtra",
        "latitude":  19.99727,
        "longitude": 73.79096,
    },
    {
        "location": "Kolhapur",
        "region":   "Madhya Maharashtra",
        "latitude":  16.69563,
        "longitude": 74.23167,
    },
    {
        "location": "Solapur",
        "region":   "Madhya Maharashtra",
        "latitude":  17.67152,
        "longitude": 75.91044,
    },
    {
        "location": "Chhatrapati Sambhajinagar",
        "region":   "Marathwada",
        "latitude":  19.87757,
        "longitude": 75.34226,
    },
    {
        "location": "Nagpur",
        "region":   "Vidarbha",
        "latitude":  21.14631,
        "longitude": 79.08491,
    },
]

# ─── API variable lists ────────────────────────────────────────────────────────
DAILY_VARS = [
    "weather_code",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "rain_sum",
    "precipitation_hours",
    "sunshine_duration",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
]

# Relative humidity and surface pressure are only available at hourly resolution
# in the ERA5 model on Open-Meteo. We request them hourly then aggregate to daily.
HOURLY_VARS = [
    "relative_humidity_2m",
    "pressure_msl",
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def fetch_with_retry(params: dict, location: str, year: int) -> dict | None:
    """
    GET the Open-Meteo archive API with exponential-backoff retry.
    Returns the parsed JSON dict on success, or None on failure.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(API_URL, params=params, timeout=60)
            if response.status_code == 200:
                data = response.json()
                if "error" in data:
                    log.error(
                        "API error for %s %d (attempt %d): %s",
                        location, year, attempt, data.get("reason", data)
                    )
                    return None
                return data
            else:
                log.warning(
                    "HTTP %d for %s %d (attempt %d/%d). Retrying in %ds…",
                    response.status_code, location, year, attempt, MAX_RETRIES, RETRY_DELAY
                )
        except requests.exceptions.RequestException as exc:
            log.warning(
                "Request exception for %s %d (attempt %d/%d): %s. Retrying in %ds…",
                location, year, attempt, MAX_RETRIES, exc, RETRY_DELAY
            )
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY * attempt)   # exponential back-off

    log.error("FAILED after %d attempts: %s year %d", MAX_RETRIES, location, year)
    return None


def aggregate_hourly_to_daily(hourly: dict) -> pd.DataFrame:
    """
    Convert the hourly dict returned by the API into a daily DataFrame.
    Computes mean, max, min for relative_humidity_2m and mean for pressure_msl.
    """
    df_h = pd.DataFrame(hourly)
    df_h["date"] = pd.to_datetime(df_h["time"]).dt.date

    daily_agg = (
        df_h.groupby("date")
        .agg(
            relative_humidity_2m_mean=("relative_humidity_2m", "mean"),
            relative_humidity_2m_max =("relative_humidity_2m", "max"),
            relative_humidity_2m_min =("relative_humidity_2m", "min"),
            pressure_msl_mean        =("pressure_msl",          "mean"),
        )
        .reset_index()
    )
    daily_agg["date"] = pd.to_datetime(daily_agg["date"])
    return daily_agg


def fetch_location_year(loc_info: dict, year: int) -> pd.DataFrame | None:
    """
    Fetch one location × one year of data.
    Returns a daily DataFrame or None if the request failed.
    """
    start = f"{year}-01-01"
    end   = f"{year}-12-31"

    params = {
        "latitude":   loc_info["latitude"],
        "longitude":  loc_info["longitude"],
        "start_date": start,
        "end_date":   end,
        "daily":      ",".join(DAILY_VARS),
        "hourly":     ",".join(HOURLY_VARS),
        "models":     MODEL,
        "timezone":   TIMEZONE,
    }

    data = fetch_with_retry(params, loc_info["location"], year)
    if data is None:
        return None

    # ── Build daily DataFrame ──────────────────────────────────────────────
    daily_raw = data.get("daily", {})
    if not daily_raw or "time" not in daily_raw:
        log.error("No daily data returned for %s %d", loc_info["location"], year)
        return None

    df_daily = pd.DataFrame(daily_raw)
    df_daily.rename(columns={"time": "date"}, inplace=True)
    df_daily["date"] = pd.to_datetime(df_daily["date"])

    # ── Aggregate hourly → daily for humidity & pressure ──────────────────
    hourly_raw = data.get("hourly", {})
    if hourly_raw and "time" in hourly_raw:
        df_hourly_daily = aggregate_hourly_to_daily(hourly_raw)
        df_daily = df_daily.merge(df_hourly_daily, on="date", how="left")
    else:
        log.warning("No hourly data returned for %s %d; humidity/pressure will be NaN", loc_info["location"], year)
        df_daily["relative_humidity_2m_mean"] = float("nan")
        df_daily["relative_humidity_2m_max"]  = float("nan")
        df_daily["relative_humidity_2m_min"]  = float("nan")
        df_daily["pressure_msl_mean"]          = float("nan")

    # ── Attach metadata columns ────────────────────────────────────────────
    df_daily["location"]  = loc_info["location"]
    df_daily["region"]    = loc_info["region"]
    df_daily["latitude"]  = loc_info["latitude"]
    df_daily["longitude"] = loc_info["longitude"]

    return df_daily


# ─── Main collection routine ──────────────────────────────────────────────────

def collect_all() -> pd.DataFrame:
    """
    Iterate over all locations and years, collecting data year-by-year.
    Tracks and reports failures without silently discarding them.
    """
    all_frames = []
    failures   = []   # (location, year) tuples that could not be fetched

    years = list(range(int(START_DATE[:4]), int(END_DATE[:4]) + 1))
    total_requests = len(LOCATIONS) * len(years)
    done = 0

    log.info("=" * 70)
    log.info("WeatherCast Phase 1 — Data Collection Starting")
    log.info("Period : %s → %s", START_DATE, END_DATE)
    log.info("Model  : %s", MODEL.upper())
    log.info("Locations: %d  |  Years: %d  |  Total API calls: %d",
             len(LOCATIONS), len(years), total_requests)
    log.info("=" * 70)

    for loc_info in LOCATIONS:
        loc_name = loc_info["location"]
        log.info("── Fetching: %s (%s)", loc_name, loc_info["region"])
        loc_frames = []

        for year in years:
            df = fetch_location_year(loc_info, year)
            done += 1

            if df is None:
                failures.append((loc_name, year))
                log.error("  ✗ MISSING data for %s / %d  [%d/%d]", loc_name, year, done, total_requests)
            else:
                loc_frames.append(df)
                if year % 5 == 0 or year == years[-1]:
                    log.info("  ✓ %s / %d  [%d/%d]", loc_name, year, done, total_requests)

            # Polite delay between requests to respect rate limits
            time.sleep(0.3)

        if loc_frames:
            all_frames.append(pd.concat(loc_frames, ignore_index=True))

    # ── Failure summary ────────────────────────────────────────────────────
    if failures:
        log.warning("=" * 70)
        log.warning("COLLECTION COMPLETE WITH %d FAILED REQUESTS:", len(failures))
        for loc, yr in failures:
            log.warning("  ✗ %s / %d", loc, yr)
        log.warning("These location-year combinations are MISSING from the dataset.")
        log.warning("=" * 70)
    else:
        log.info("=" * 70)
        log.info("All %d requests completed successfully — no failures.", total_requests)
        log.info("=" * 70)

    if not all_frames:
        raise RuntimeError("No data was collected. Check API connectivity and parameters.")

    combined = pd.concat(all_frames, ignore_index=True)

    # ── Sort and deduplicate ───────────────────────────────────────────────
    combined.sort_values(["location", "date"], inplace=True)
    before = len(combined)
    combined.drop_duplicates(subset=["location", "date"], inplace=True)
    after = len(combined)
    if before != after:
        log.warning("Removed %d duplicate rows (location × date).", before - after)

    combined.reset_index(drop=True, inplace=True)
    return combined


# ─── Validation & reporting ───────────────────────────────────────────────────

def validate_and_report(df: pd.DataFrame) -> None:
    """
    Print a comprehensive validation summary of the collected dataset.
    DO NOT modify or impute the data — report as-is.
    """
    sep = "─" * 70

    print("\n" + "=" * 70)
    print("  WEATHERCAST PHASE 1 — DATASET VALIDATION REPORT")
    print("=" * 70)

    print(f"\n{'Rows':.<30} {len(df):>10,}")
    print(f"{'Columns':.<30} {len(df.columns):>10}")
    print(f"\nColumn list:")
    for col in df.columns:
        print(f"    {col}")

    print(f"\n{sep}")
    print("  DATE RANGE (overall)")
    print(sep)
    print(f"  Earliest date : {df['date'].min().date()}")
    print(f"  Latest date   : {df['date'].max().date()}")

    print(f"\n{sep}")
    print("  ROWS PER LOCATION & DATE RANGE")
    print(sep)
    loc_stats = (
        df.groupby("location")
        .agg(
            region   =("region",   "first"),
            rows     =("date",     "count"),
            min_date =("date",     "min"),
            max_date =("date",     "max"),
        )
        .reset_index()
    )
    for _, row in loc_stats.iterrows():
        print(
            f"  {row['location']:<32} {row['region']:<22} "
            f"rows={row['rows']:>5}  "
            f"{str(row['min_date'].date())} → {str(row['max_date'].date())}"
        )

    print(f"\n{sep}")
    print("  REGION SUMMARY")
    print(sep)
    region_counts = df.groupby("region").agg(locations=("location", "nunique"), rows=("date", "count")).reset_index()
    for _, row in region_counts.iterrows():
        print(f"  {row['region']:<25} locations={row['locations']}  rows={row['rows']:>7,}")

    print(f"\n{sep}")
    print("  DUPLICATE ROW CHECK (location × date)")
    print(sep)
    dup_count = df.duplicated(subset=["location", "date"]).sum()
    print(f"  Duplicate rows: {dup_count}")

    print(f"\n{sep}")
    print("  MISSING VALUES SUMMARY")
    print(sep)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    has_missing = False
    for col in df.columns:
        if missing[col] > 0:
            has_missing = True
            print(f"  {col:<40} {missing[col]:>6} missing ({missing_pct[col]:.2f}%)")
    if not has_missing:
        print("  ✓ No missing values found.")

    print(f"\n{sep}")
    print("  RAIN_SUM STATISTICS")
    print(sep)
    rs = df["rain_sum"].describe()
    for stat, val in rs.items():
        print(f"  {stat:<20} {val:>10.4f}")

    print(f"\n{sep}")
    print("  RAINY vs NON-RAINY DAYS  (rain_sum > 0)")
    print(sep)
    valid_rain = df["rain_sum"].dropna()
    rainy_days     = (valid_rain > 0).sum()
    non_rainy_days = (valid_rain == 0).sum()
    nan_days       = df["rain_sum"].isna().sum()
    total_valid    = rainy_days + non_rainy_days
    print(f"  Rainy days     (rain_sum > 0) : {rainy_days:>7,}  ({rainy_days/total_valid*100:.1f}%)")
    print(f"  Non-rainy days (rain_sum == 0) : {non_rainy_days:>7,}  ({non_rainy_days/total_valid*100:.1f}%)")
    print(f"  NaN / missing  rain_sum        : {nan_days:>7,}")

    print("\n" + "=" * 70)
    print("  END OF VALIDATION REPORT")
    print("=" * 70 + "\n")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Output will be saved to: %s", OUTPUT_CSV)

    # Collect data
    df = collect_all()

    # Save raw CSV — do NOT impute or modify
    df.to_csv(OUTPUT_CSV, index=False)
    log.info("Raw dataset saved → %s  (%d rows, %d columns)", OUTPUT_CSV, len(df), len(df.columns))

    # Validate and report
    validate_and_report(df)

    return df


if __name__ == "__main__":
    main()
