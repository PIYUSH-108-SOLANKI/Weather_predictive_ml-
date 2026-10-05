# WeatherCast — Data Source Documentation

**Phase 1 | ML Case Study 75: Weather Pattern Analysis**

---

## Source

| Property | Value |
|---|---|
| **Provider** | [Open-Meteo](https://open-meteo.com) — Free, open-source weather API |
| **API Type** | Historical Weather API (Reanalysis) |
| **Base Endpoint** | `https://archive-api.open-meteo.com/v1/archive` |
| **Geocoding API** | `https://geocoding-api.open-meteo.com/v1/search` |
| **License** | Open data, free for non-commercial use |
| **Data Retrieved On** | 2026-09-30 |

---

## Historical Model Selected: ERA5

**Selected model:** `era5`

### Why ERA5?

| Model | Coverage Start | Resolution | pressure_msl | Suitable for 2000–2024 |
|---|---|---|---|---|
| **ERA5** | 1940 | 0.25° (~28 km) | ✅ Yes | ✅ Full period |
| ERA5-Land | 1950 | 0.1° (~11 km) | ❌ No | ⚠️ Missing pressure |
| IFS (ECMWF) | 2017 | 9 km | ✅ Yes | ❌ Only 2017 onwards |

**ERA5 (ECMWF Reanalysis v5)** was selected because:
1. It covers the full required period (2000–2024) without gaps, starting from 1940.
2. It provides `pressure_msl` (mean sea-level pressure), which ERA5-Land does not.
3. It is the gold-standard reanalysis product used globally for climate and ML studies.
4. It provides gap-free, consistent data across all 25 years — critical for a long-period binary classifier.

---

## Date Range

| Property | Value |
|---|---|
| **Start date** | 2000-01-01 |
| **End date** | 2024-12-31 |
| **Total years** | 25 |
| **Expected rows per location** | ~9,131 (25 years × ~365.24 days/year) |
| **Total expected rows** | ~73,048 (8 locations × ~9,131) |

---

## Locations & Verified Coordinates

Coordinates verified via the Open-Meteo Geocoding API on 2026-09-30.

| Location | Region | Latitude | Longitude |
|---|---|---|---|
| Mumbai | Konkan | 19.07283 | 72.88261 |
| Ratnagiri | Konkan | 16.99154 | 73.31022 |
| Pune | Madhya Maharashtra | 18.51957 | 73.85535 |
| Nashik | Madhya Maharashtra | 19.99727 | 73.79096 |
| Kolhapur | Madhya Maharashtra | 16.69563 | 74.23167 |
| Solapur | Madhya Maharashtra | 17.67152 | 75.91044 |
| Chhatrapati Sambhajinagar | Marathwada | 19.87757 | 75.34226 |
| Nagpur | Vidarbha | 21.14631 | 79.08491 |

> **Note:** Chhatrapati Sambhajinagar was formerly known as Aurangabad. The Open-Meteo Geocoding API returns it under "Aurangabad" — coordinates verified as correct for the same city.

---

## Variables Collected

### Daily Variables (directly from API)

| Variable | Unit | Relevance to Next-Day Rain Prediction |
|---|---|---|
| `weather_code` | WMO code | Encodes current weather conditions (thunderstorm, rain, snow, etc.) |
| `temperature_2m_max` | °C | Daily maximum temperature — warm-moist air drives convective rainfall |
| `temperature_2m_min` | °C | Daily minimum temperature — night-time cooling and dew point |
| `precipitation_sum` | mm | Total precipitation — includes all forms |
| `rain_sum` | mm | Liquid rain specifically — direct precursor signal for next-day rain |
| `precipitation_hours` | hours | Duration of precipitation — indicates persistence of rain systems |
| `sunshine_duration` | seconds | Hours of sunshine — inverse proxy for cloud cover |
| `wind_speed_10m_max` | km/h | Maximum wind — associated with approaching weather systems |
| `wind_gusts_10m_max` | km/h | Maximum wind gusts — front/storm passage indicator |
| `wind_direction_10m_dominant` | ° | Dominant wind direction — south-westerly winds bring monsoon moisture |

### Hourly Variables (aggregated to daily)

Relative humidity and mean sea-level pressure are only available at hourly resolution in ERA5 on Open-Meteo. They are collected at hourly frequency and then aggregated to daily statistics:

| Hourly Variable | Aggregation | Output Column | Unit | Relevance |
|---|---|---|---|---|
| `relative_humidity_2m` | mean | `relative_humidity_2m_mean` | % | High humidity → precondition for rainfall |
| `relative_humidity_2m` | max | `relative_humidity_2m_max` | % | Peak daily humidity |
| `relative_humidity_2m` | min | `relative_humidity_2m_min` | % | Daily humidity range |
| `pressure_msl` | mean | `pressure_msl_mean` | hPa | Falling pressure → approaching low-pressure systems = rain |

---

## Final Dataset

| Property | Value |
|---|---|
| **Output file** | `data/raw/maharashtra_weather_2000_2024.csv` |
| **Rows** | See validation report |
| **Columns** | 18 |
| **Format** | UTF-8 CSV |
| **Date column** | `date` (YYYY-MM-DD) |

---

## API Limitations Discovered

| Limitation | Impact |
|---|---|
| `relative_humidity_2m` and `pressure_msl` are hourly-only in ERA5 | Resolved by requesting hourly + aggregating to daily; adds slight processing overhead |
| `sunshine_duration` is in **seconds** (not hours) | Noted — will convert to hours in Phase 3 preprocessing |
| ERA5 resolution is 0.25° (~28 km grid) | Point data is interpolated from the nearest grid cell — acceptable for city-level analysis |
| Free tier has no authentication but may rate-limit burst requests | Mitigated with 0.3 s inter-request delay and year-by-year chunking |
| No API availability for future forecasts via archive API | By design — all data is historical; no leakage possible |

---

## Data Quality Notes

- The raw CSV is **not imputed or modified** — missing values (if any) are preserved as `NaN`.
- Duplicate rows (same location × date) are removed during collection.
- This raw file is the input to Phase 2 (EDA) and Phase 3 (preprocessing).
- `rain_tomorrow` target variable is **not** created in this phase.
