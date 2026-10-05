# WeatherCast 🌦️

> **ML Case Study 75 — Weather Pattern Analysis**  
> Binary Classification | Maharashtra Rain Prediction

---

## Assignment / Problem Statement

An environmental dataset contains historical weather observations, and the organization wants to identify meaningful patterns or estimate a selected future measure.

**Our selected problem:** Predict whether it will rain tomorrow for a selected location/region in Maharashtra, India.

---

## ML Task

| Property | Details |
|---|---|
| **Task Type** | Binary Classification |
| **Target Variable** | `rain_tomorrow` |
| **Target Encoding** | `1` = Rain tomorrow &nbsp;·&nbsp; `0` = No rain tomorrow |
| **Region** | Maharashtra, India |

---

## Project Status

**Current Phase: Phase 8 — Final Documentation & Viva Submission** ✅

---

## Phased Development Approach

| Phase | Title | Status |
|---|---|---|
| **Phase 0** | Project Initialization & Environment Setup | ✅ Complete |
| **Phase 1** | Data Collection & Raw Dataset Acquisition | ✅ Complete |
| **Phase 2** | Target Creation & Strict Date-Continuity Validation | ✅ Complete |
| **Phase 3** | Exploratory Data Analysis (EDA) | ✅ Complete |
| **Phase 4** | Data Preprocessing & Feature Engineering | ✅ Complete |
| **Phase 5** | Model Building & Training | ✅ Complete |
| **Phase 6** | Model Evaluation & Hyperparameter Tuning | ✅ Complete |
| **Phase 7** | Streamlit Application Development | ✅ Complete |
| **Phase 8** | Final Documentation & Viva Submission | ✅ Complete |

---

## Project Structure

```
WeatherCast/
├── data/
│   ├── raw/            # Raw downloaded datasets (not tracked by Git)
│   └── processed/      # Cleaned & feature-engineered data (not tracked by Git)
├── notebooks/          # Jupyter notebooks for EDA and experiments
├── src/                # Reusable Python source code / modules
├── models/             # Trained model artifacts (not tracked by Git)
├── app/                # Streamlit application (app.py)
├── docs/               # Documentation, reports, assignment writeup
├── requirements.txt    # Python dependencies
├── .gitignore          # Git exclusions
└── README.md           # This file
```

---

## Setup & Running Instructions

```bash
# 1. Clone the repository
git clone <repo-url>
cd WeatherCast

# 2. Create and activate the virtual environment
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Streamlit Web Application (Phase 7)
streamlit run app/app.py
```

---

## Datasets
- **Raw Dataset (2000–2024):** `data/raw/maharashtra_weather_2000_2024.csv` (72,691 rows × 19 columns)
- **Processed Dataset with Target:** `data/processed/maharashtra_weather_with_target.csv` (72,691 rows × 20 columns)
- **Master Jupyter Notebook:** [`notebooks/WeatherCast_ML.ipynb`](notebooks/WeatherCast_ML.ipynb)

| Property | Details |
|---|---|
| **Source** | [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) |
| **Model** | ERA5 (ECMWF Reanalysis v5) |
| **Period** | 2000-01-01 → 2024-12-31 (25 years) |
| **Locations** | 8 Maharashtra cities across 4 regions |
| **Raw file** | `data/raw/maharashtra_weather_2000_2024.csv` |
| **Documentation** | [`docs/data_source.md`](docs/data_source.md) |

---

## Target Definition & Date-Continuity Handling

- **Target Variable:** `rain_tomorrow`
  - `1` = Measurable liquid precipitation on next calendar day (`rain_sum > 0 mm`) — **30,687 observations (42.22%)**
  - `0` = No measurable precipitation on next calendar day (`rain_sum == 0 mm`) — **41,995 observations (57.78%)**
  - `NaN` = Next calendar day missing or discontinuous — **9 observations**
- **Strict Date-Continuity Rule:**
  - A valid target is formed **only** when `next_date == current_date + 1 calendar day`.
  - **Chhatrapati Sambhajinagar (2002 Gap):** On `2001-12-31`, the subsequent recorded observation is `2003-01-01` (366 days later). The target for `2001-12-31` is explicitly set to `NaN` to avoid predicting across a 1-year gap.
  - **Dataset End (`2024-12-31`):** For all 8 locations, `2024-12-31` has no subsequent observation, so its target is explicitly set to `NaN`.
- **Target Documentation:** [`docs/target_definition.md`](docs/target_definition.md)

---

## Phase 3: Exploratory Data Analysis (EDA) Highlights

Comprehensive exploratory data analysis was conducted across all 72,691 observations (2000–2024) in [`notebooks/WeatherCast_ML.ipynb`](notebooks/WeatherCast_ML.ipynb), generating 12 analytical visualizations stored in `reports/figures/`:

1. **Target Balance:** The target is moderately balanced (42.22% rainy vs 57.78% non-rainy), establishing a well-conditioned binary classification task without requiring synthetic resampling.
2. **Extreme Skewness:** Daily rainfall (`rain_sum`) is heavily zero-inflated (57.78% dry days) with an extreme right tail reaching 341 mm/day.
3. **Coastal & Orographic Gradient:** Konkan coastal locations (Ratnagiri: 8.10 mm/day, Mumbai: 5.56 mm/day) receive over 2.4x the daily rainfall volume of rain-shadow inland plateau regions (Solapur: 2.39 mm/day).
4. **Monsoon Dominance:** Rainfall is heavily concentrated in the Southwest Monsoon (June–September), peaking in July (~96.8% rainy days) and August (~96.3% rainy days), contrasted with dry winters (<5% rainy days).
5. **Physical Precursors of Rain:**
   - **Relative Humidity:** Averages 80.21% on days preceding rain versus 56.28% on dry days (+23.93% delta).
   - **Barometric Pressure:** Decreases by ~4.3 hPa on days prior to rain events.
   - **Wind Dynamics:** Maximum wind gusts elevate by ~7.2 km/h prior to rainfall.
6. **Weather Code Persistence:** WMO codes reflecting active drizzle/rain (51–65) lead to >75–98% probability of rain the next day, whereas clear sky codes (0, 1) result in <7%.
7. **Feature Redundancy Identified:** `precipitation_sum` and `rain_sum` have identical distributions and collinearity ($r = 1.00$) in Maharashtra; one will be pruned during feature engineering.

---

## Phase 5: Model Training & Baseline Evaluation

Three binary-classification pipelines were trained on the chronological 2000–2022 partition and evaluated against Validation (2023) and Final Test (2024) in [`notebooks/WeatherCast_ML.ipynb`](notebooks/WeatherCast_ML.ipynb):

| Model | Val Accuracy | Val F1 | Val ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.8740 | 0.8508 | 0.9448 |
| Random Forest | 0.8705 | 0.8462 | 0.9474 |
| Gradient Boosting | 0.8729 | 0.8486 | 0.9483 |

Trained pipelines (preprocessor + classifier) saved in `models/`. No model is declared best at this stage — full evaluation pending Phase 6/7.

---

## Phase 6: Model Evaluation, Hyperparameter Tuning & Final Selection

Phase 6 performed exhaustive evaluation, time-aware tuning, error analysis, and final model selection in [`notebooks/WeatherCast_ML.ipynb`](notebooks/WeatherCast_ML.ipynb):

### 1. Baseline vs Tuned Performance

| Model | Stage | Val (2023) Accuracy | Val (2023) F1 | Val (2023) ROC-AUC | Test (2024) Accuracy | Test (2024) F1 | Test (2024) ROC-AUC |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | Baseline | 0.8740 | 0.8508 | 0.9448 | 0.9038 | 0.9000 | 0.9681 |
| **Logistic Regression** | Tuned | 0.8736 | 0.8504 | 0.9448 | 0.9045 | 0.9006 | 0.9680 |
| **Random Forest** | Baseline | 0.8750 | 0.8519 | 0.9477 | 0.9065 | 0.9011 | 0.9718 |
| **Random Forest** | Tuned | 0.8702 | 0.8452 | 0.9468 | 0.9055 | 0.9003 | 0.9722 |
| **Gradient Boosting** | Baseline | **0.8740** | **0.8499** | **0.9499** | **0.9062** | **0.9014** | **0.9714** |
| **Gradient Boosting** | Tuned (Adopted) | **0.8740** | **0.8499** | **0.9499** | **0.9062** | **0.9014** | **0.9714** |

### 2. Hyperparameter Tuning Protocol
- **Method:** `GridSearchCV` with `TimeSeriesSplit(n_splits=3)` strictly executed on 2000–2022 training rows.
- **Leakage Safeguard:** 2023 validation and 2024 test data were strictly excluded from tuning.
- **Selected Hyperparameters:**
  - **Logistic Regression:** `{'clf__C': 1.0, 'clf__solver': 'liblinear', 'clf__max_iter': 1000}` (CV AUC = 0.9625)
  - **Random Forest:** `{'clf__n_estimators': 100, 'clf__max_depth': 20, 'clf__min_samples_leaf': 8}` (CV AUC = 0.9637)
  - **Gradient Boosting:** Retained Phase 5 baseline configuration (`n_estimators=100, max_depth=4, learning_rate=0.1, subsample=0.8`), which achieved the top baseline validation ROC-AUC (0.9499).

### 3. Error Analysis Highlights (Val 2023, Gradient Boosting)
- **False Alarms (FP):** 100 observations (3.4% of validation data). Distributed across cities (highest in Ratnagiri at 17, Pune at 16, and Nagpur at 14) during borderline humidity conditions.
- **Missed Rain (FN):** 268 observations (9.2% of validation data). Heavily concentrated in the **Summer / Pre-Monsoon** transition period (165 FNs, 22.4% FN rate) when sporadic pre-monsoon convective showers occur without sustained synoptic signatures.

### 4. Probability Calibration & Feature Importance
- **Calibration (Brier Score):** Gradient Boosting achieved the top probability quality (Val Brier = **0.0897**, Test Brier = **0.0665**).
- **Dominant Features:** `rain_sum_lag_1`, `rain_sum_3d_total`, `weather_code`, `relative_humidity_2m_mean`, and `humidity_mean_lag_1`.

### 5. Final Model Selected for Phase 7 Deployment
- **Selected Model:** **Gradient Boosting** (`models/best_model_phase6.joblib`)
- **Justification:** Top validation ROC-AUC (0.9499), superior probability calibration (Brier 0.0897), highest test F1 (0.9014), and lowest validation-to-test performance degradation ($\Delta \text{AUC} = 0.0215$).

---

## Phase 4: Feature Engineering & Preprocessing Highlights

The validated dataset was transformed into an ML-ready feature matrix in [`notebooks/WeatherCast_ML.ipynb`](notebooks/WeatherCast_ML.ipynb) and saved to [`data/processed/weather_ml_features.csv`](data/processed/weather_ml_features.csv):

1. **Anti-Leakage Architecture:** Strict zero-leakage pipeline design. All lag and rolling features rely purely on historical calendar dates ($t-1$ to $t-7$).
2. **Redundant Feature Pruning:** `precipitation_sum` was dropped due to $r = 1.00$ collinearity with `rain_sum`. `region` was excluded from model inputs $X$ as it is redundant with `location`.
3. **Temporal Cyclical Encoding:** Replaced raw dates with continuous sine and cosine projections (`month_sin`, `month_cos`, `day_of_year_sin`, `day_of_year_cos`) to eliminate the artificial December–January numerical boundary.
4. **Calendar-Strict Lags & Rolling Features:** 8 lag features and 6 rolling window aggregations computed with complete calendar awareness. The 2002 Chhatrapati Sambhajinagar gap was strictly preserved without bridging.
5. **Context Gap Handling:** Dropped the first 7 days lacking prior historical context (63 rows, 0.087% of data), leaving **72,619 clean supervised instances**.
6. **Chronological Splitting:**
   - **Train (2000–2022):** 66,779 observations (91.96%)
   - **Validation (2023):** 2,920 observations (4.02%)
   - **Test (2024):** 2,920 observations (4.02%)
7. **Feature Inventory:** 35 base input features (32 numerical, 3 categorical) expanding to 54 transformed features after one-hot encoding.
8. **Leakage Audit:** Successfully passed an explicit 7-point anti-leakage audit.
9. **Documentation:** Detailed engineering writeup available in [`docs/feature_engineering.md`](docs/feature_engineering.md).

---

## Technologies Used

- **Python 3.12**
- pandas · numpy · matplotlib · seaborn
- scikit-learn
- streamlit
- requests

---

*Assignment submitted for: Machine Learning (Sem 5)*
