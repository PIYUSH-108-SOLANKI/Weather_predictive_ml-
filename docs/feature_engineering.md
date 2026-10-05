# WeatherCast — Feature Engineering & Preprocessing Documentation

**Phase 4 | ML Case Study 75: Weather Pattern Analysis**

---

## 1. Overview and Preprocessing Philosophy

Phase 4 transforms the raw meteorological observations into a clean, leakage-free feature matrix tailored for binary next-day rain prediction (`rain_tomorrow`). 

The guiding principle of this phase is **zero data leakage**:
1. No future observation or target precursor is ever visible to input features.
2. All lagged and rolling features are strictly backward-looking.
3. Preprocessing parameters (scalers, encoders) are fitted strictly on the Training partition (2000–2022).
4. Validation (2023) and Test (2024) partitions remain completely unseen during pipeline fitting.

---

## 2. Feature Selection & Pruning Decisions

### A. Redundant Precipitation Variables
- **Finding from EDA:** `precipitation_sum` and `rain_sum` have an exact correlation of $r = 1.00$ because snowfall is non-existent across the selected Maharashtra stations.
- **Decision:** `precipitation_sum` was **dropped**. `rain_sum` was retained as the domain-relevant metric for rain prediction.

### B. Geographic Features: Location vs Region
- Every location has a deterministic 1-to-1 relationship with its meteorological region:
  - Mumbai, Ratnagiri → Konkan
  - Pune, Nashik, Kolhapur, Solapur → Madhya Maharashtra
  - Chhatrapati Sambhajinagar → Marathwada
  - Nagpur → Vidarbha
- **Decision:** Retain `location` (one-hot encoded), `latitude`, and `longitude` in the feature matrix $X$. `region` was excluded from $X$ to eliminate redundant categorical degrees of freedom, but retained in the dataset for grouping and reporting.

### C. Excluded Metadata
- `date`: Raw sequential dates are non-stationary. Instead, cyclical seasonal features were extracted.
- `next_date` and `next_day_rain_sum`: Excluded and purged (these were temporary target-validation helpers from Phase 2).
- `rain_tomorrow`: The prediction target; strictly isolated from $X$.

---

## 3. Engineered Features

### A. Temporal & Cyclical Features
Rainfall follows strong annual and seasonal cycles. Integer month (1–12) or day of year (1–366) creates an artificial numerical jump between December 31 and January 1. To preserve cyclical continuity, we apply sine and cosine transformations:

$$\\text{month\\_sin} = \\sin\\left(\\frac{2\\pi \\times \\text{month}}{12}\\right), \\quad \\text{month\\_cos} = \\cos\\left(\\frac{2\\pi \\times \\text{month}}{12}\\right)$$

$$\\text{day\\_of\\_year\\_sin} = \\sin\\left(\\frac{2\\pi \\times \\text{day\\_of\\_year}}{365.25}\\right), \\quad \\text{day\\_of\\_year\\_cos} = \\cos\\left(\\frac{2\\pi \\times \\text{day\\_of\\_year}}{365.25}\\right)$$

Additionally, categorical `season` was derived based on India Meteorological Department (IMD) guidelines:
- **Winter:** January – February
- **Summer / Pre-Monsoon:** March – May
- **Southwest Monsoon:** June – September
- **Post-Monsoon:** October – December

### B. Calendar-Strict Lag Features
Precipitation events exhibit temporal autocorrelation (weather persistence). We construct 8 lag features per location:
- `rain_sum_lag_1`, `rain_sum_lag_3`, `rain_sum_lag_7`
- `temperature_max_lag_1`, `temperature_min_lag_1`
- `humidity_mean_lag_1`, `pressure_mean_lag_1`, `wind_speed_max_lag_1`

**Strict Calendar Alignment:**
Observations were reindexed against a continuous calendar (`2000-01-01` to `2024-12-31`). For Chhatrapati Sambhajinagar, where the entire year 2002 is absent, the lag on `2003-01-01` is strictly `NaN` rather than mistakenly pulling data from `2001-12-31`.

### C. Strictly Backward-Looking Rolling Features
Trailing atmospheric tendencies over 3-day and 7-day spans:
- `rain_sum_3d_total` (sum over days $D-3$ to $D-1$)
- `rain_sum_7d_total` (sum over days $D-7$ to $D-1$)
- `temperature_max_3d_mean` (mean over days $D-3$ to $D-1$)
- `temperature_min_3d_mean` (mean over days $D-3$ to $D-1$)
- `humidity_3d_mean` (mean over days $D-3$ to $D-1$)
- `pressure_3d_mean` (mean over days $D-3$ to $D-1$)

Current-day observations ($D$) are deliberately excluded from these rolling aggregations to maintain clear semantic separation between current state and antecedent trajectory.

---

## 4. Handling Missing Historical Context

Requiring a 7-day lookback window naturally creates missing values:
- First 7 days of 2000 for each of the 8 locations: $8 \\times 7 = 56$ rows.
- First 7 days of 2003 for Chhatrapati Sambhajinagar following the 2002 gap: 7 rows.
- **Total context-lacking rows:** 63 rows out of 72,682 instances (0.087%).

**Action Taken:** These 63 rows were dropped from the supervised modeling dataset. Dropping <0.09% of records is superior to synthetic backfilling or forward-imputing, which would introduce fabricated historical dynamics.

---

## 5. Feature Inventory

| Group | Features | Count | Type | Encoding / Scaling |
|---|---|---|---|---|
| **Current Meteorological** | `temperature_2m_max`, `temperature_2m_min`, `rain_sum`, `precipitation_hours`, `sunshine_duration`, `wind_speed_10m_max`, `wind_gusts_10m_max`, `wind_direction_10m_dominant`, `relative_humidity_2m_mean`, `relative_humidity_2m_max`, `relative_humidity_2m_min`, `pressure_msl_mean` | 12 | Continuous | `StandardScaler` |
| **Geographic Coordinates** | `latitude`, `longitude` | 2 | Continuous | `StandardScaler` |
| **Temporal Cyclical** | `month_sin`, `month_cos`, `day_of_year_sin`, `day_of_year_cos` | 4 | Continuous | Already bounded $[-1, 1]$ / `StandardScaler` |
| **Historical Lags** | `rain_sum_lag_1`, `rain_sum_lag_3`, `rain_sum_lag_7`, `temperature_max_lag_1`, `temperature_min_lag_1`, `humidity_mean_lag_1`, `pressure_mean_lag_1`, `wind_speed_max_lag_1` | 8 | Continuous | `StandardScaler` |
| **Rolling History** | `rain_sum_3d_total`, `rain_sum_7d_total`, `temperature_max_3d_mean`, `temperature_min_3d_mean`, `humidity_3d_mean`, `pressure_3d_mean` | 6 | Continuous | `StandardScaler` |
| **Categorical Predictors** | `location` (8 levels), `weather_code` (10 levels), `season` (4 levels) | 3 | Categorical | `OneHotEncoder(drop='first')` |
| **Total Base Features** | — | **35** | — | — |
| **Total Transformed Features** | — | **54** | — | After one-hot dummy expansion |

---

## 6. Chronological Train / Validation / Test Split

A time-series prediction task must never use randomized cross-validation, which causes temporal data leakage. We establish a forward-looking chronological split:

| Partition | Time Span | Total Years | Observations | Percentage | Purpose |
|---|---|---|---|---|---|
| **Training** | 2000-01-08 to 2022-12-31 | 23 years | **66,779** | **91.96%** | Model fitting, parameter estimation |
| **Validation** | 2023-01-01 to 2023-12-31 | 1 year | **2,920** | **4.02%** | Model comparison, probability threshold calibration |
| **Final Test** | 2024-01-01 to 2024-12-30 | 1 year | **2,920** | **4.02%** | Unseen out-of-time generalization evaluation |
| **Total** | 2000–2024 | 25 years | **72,619** | **100.00%** | Full analytical dataset |

*(Note: 2024 had 366 days, but December 31, 2024 has no known next-day target, leaving exactly 365 test days × 8 locations = 2,920 test observations).*

---

## 7. Scaling Strategy

- **Linear Models (Logistic Regression):** Highly sensitive to magnitude differences between sunshine duration (~40,000 s) and rainfall (~5 mm). Numerical features are standardized ($\mu=0, \sigma=1$) using `StandardScaler`.
- **Tree-Based Ensembles (Random Forest, Gradient Boosting):** Invariant to monotonic scaling. The preprocessing pipeline supports both unscaled and scaled pipelines via modular `ColumnTransformer`.
- **Leakage Prevention:** Scalers and encoders are fitted **strictly on the Training partition** ($X_{\\text{train}}$) and applied to Validation and Test via `transform()`.

---

## 8. Data-Leakage Audit Results

| # | Audit Criterion | Verification Method | Status |
|---|---|---|---|
| 1 | **No target-derived features in $X$** | Verified `rain_tomorrow` and `next_day_rain_sum` absent from $X$ | **PASS ✅** |
| 2 | **No future observations in $X$** | Verified no future shifted columns exist | **PASS ✅** |
| 3 | **Strict backward lag shifting** | Verified all lag indices shift $\ge 1$ | **PASS ✅** |
| 4 | **Strict trailing rolling window** | Verified rolling windows use indices $[t-7, t-1]$ | **PASS ✅** |
| 5 | **2002 CSN gap isolation** | Verified no bridging across the missing year 2002 | **PASS ✅** |
| 6 | **Training-only preprocessor fitting** | Scalers and encoders fitted exclusively on 2000–2022 | **PASS ✅** |
| 7 | **Strict chronological split** | Train $\le$ 2022 < Val == 2023 < Test == 2024 | **PASS ✅** |

---

## 9. Output Dataset

- **Path:** [`data/processed/weather_ml_features.csv`](../data/processed/weather_ml_features.csv)
- **Dimensions:** 72,619 rows × 41 columns (including metadata and target)
