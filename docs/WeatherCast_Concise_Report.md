# WeatherCast 🌦️: Next-Day Rain Prediction for Maharashtra
### Machine Learning Case Study 75 — Final Academic Project Report
**Academic Level:** B.Tech Semester V | **Subject:** Machine Learning Laboratory  
**Target:** Binary Classification (`rain_tomorrow` ∈ {0, 1})  

---

## 🔗 Project Links & Live Deliverables

| Deliverable | URL / Resource |
| :--- | :--- |
| **🌐 Live Streamlit Application** | [https://weatherpredictiveml-git-geavtut3pfyypewaut2she.streamlit.app/](https://weatherpredictiveml-git-geavtut3pfyypewaut2she.streamlit.app/) |
| **💻 GitHub Source Code Repository** | [https://github.com/PIYUSH-108-SOLANKI/Weather_predictive_ml-](https://github.com/PIYUSH-108-SOLANKI/Weather_predictive_ml-) |
| **📓 Extended Research Notebook** | [`notebooks/WeatherCast_ML.ipynb`](https://github.com/PIYUSH-108-SOLANKI/Weather_predictive_ml-/blob/main/notebooks/WeatherCast_ML.ipynb) (160 cells) |
| **📓 Concise Presentation Notebook** | [`notebooks/WeatherCast_Concise.ipynb`](https://github.com/PIYUSH-108-SOLANKI/Weather_predictive_ml-/blob/main/notebooks/WeatherCast_Concise.ipynb) (21 cells) |
| **📦 Production Model Artifact** | [`models/best_model_phase6.joblib`](https://github.com/PIYUSH-108-SOLANKI/Weather_predictive_ml-/blob/main/models/best_model_phase6.joblib) (Gradient Boosting) |

---

## 📌 Section 1: Problem Definition & Objectives

### 1.1 Assigned Problem Statement (Case Study 75)
> *"An environmental dataset contains historical observations, and the organization wants to identify meaningful patterns or estimate a selected future measure. (With Proper Justification)"*

### 1.2 Formulated Project Title & Formulation
* **Project Title:** **WeatherCast — Next-Day Rainfall Prediction for Maharashtra Sub-Divisions**
* **Task Type:** Supervised Binary Classification
* **Target Definition:**
  $$\text{rain\_tomorrow}_t = \begin{cases} 1 & \text{if } \text{rain\_sum}_{t+1} > 0\text{ mm (Measurable Rain)} \\ 0 & \text{if } \text{rain\_sum}_{t+1} = 0\text{ mm (Dry Day)} \end{cases}$$
* **Meteorological & Socio-Economic Justification:**
  Maharashtra encompasses four heterogeneous agro-climatic sub-divisions: **Konkan** (coastal, heavy rainfall), **Madhya Maharashtra** (plateau), **Marathwada** (drought-prone agrarian region), and **Vidarbha** (inland continental). Next-day localized rain prediction provides critical operational utility for:
  1. **Agricultural Planning:** Irrigation scheduling, pesticide application, and harvest protection for rainfed farming.
  2. **Urban Disaster Preparedness:** Early drainage management and flood risk mitigation during torrential coastal cloudbursts.

### 1.3 Key Academic Objectives
1. Acquire a clean, multi-decade reanalysis meteorological dataset across representative Maharashtra stations without calendar gaps.
2. Formulate a zero-leakage binary classification target strictly respecting causal time order.
3. Conduct in-depth exploratory data analysis (EDA) to establish physical atmospheric correlations with rainfall.
4. Engineer domain-specific temporal lag features, rolling cumulative statistics, and cyclical calendar harmonics.
5. Train, cross-validate, and hyperparameter-tune multiple supervised ML classifiers using a strict chronological split.
6. Evaluate using probability-sensitive metrics (ROC-AUC, Precision, Recall, F1-Score, Brier Score) and conduct error analysis.
7. Deploy a cloud-hosted, dual-mode Streamlit web application providing live forecasts and offline simulation.

---

## 📊 Section 2: Dataset & Preprocessing Pipeline

### 2.1 Data Acquisition & Station Coverage
* **Source:** ECMWF ERA5 Atmospheric Reanalysis via the Open-Meteo Historical Weather API.
* **Temporal Horizon:** **25 Full Years (1 January 2000 – 31 December 2024)**, daily and hourly resolution.
* **Stations Monitored (8 Representative Hubs):**
  - **Konkan:** Mumbai ($19.07^\circ\text{N}, 72.88^\circ\text{E}$), Ratnagiri ($16.99^\circ\text{N}, 73.31^\circ\text{E}$)
  - **Madhya Maharashtra:** Pune, Nashik, Kolhapur, Solapur
  - **Marathwada:** Chhatrapati Sambhajinagar
  - **Vidarbha:** Nagpur
* **Total Records:** 73,056 station-day observations with 100% calendar-date continuity verified across all 8 stations.

### 2.2 Feature Engineering (The 35-Feature Schema)
To supply supervised models with physical predictive signals without looking into the future:
1. **Temporal Lags (D-1, D-3, D-7):** Previous day's rain, max/min temperature, mean humidity, atmospheric pressure, and wind speed.
2. **Rolling Trailing Windows:** Strictly trailing 3-day and 7-day cumulative rainfall (`rain_sum_3d_total`, `rain_sum_7d_total`) and rolling means for temperature, humidity, and barometric pressure.
3. **Cyclical Calendar Projections (Fourier Harmonics):**
   $$\text{month\_sin} = \sin\left(\frac{2\pi m}{12}\right), \quad \text{month\_cos} = \cos\left(\frac{2\pi m}{12}\right)$$
   $$\text{doy\_sin} = \sin\left(\frac{2\pi \cdot \text{doy}}{365.25}\right), \quad \text{doy\_cos} = \cos\left(\frac{2\pi \cdot \text{doy}}{365.25}\right)$$
4. **Meteorological Categoricals:** Indian Meteorological Department (IMD) seasons (*Southwest Monsoon*, *Summer/Pre-Monsoon*, *Post-Monsoon*, *Winter*), WMO weather codes, and station names.

### 2.3 Strict Chronological Train/Validation/Test Split
To prevent temporal data leakage, standard random k-fold cross-validation was strictly prohibited:
* **Training Set:** 2000 – 2022 (23 Years, 67,208 records, ~92.0%)
* **Validation Set:** 2023 (1 Year, 2,920 records, ~4.0%) — used exclusively for hyperparameter tuning & threshold calibration.
* **Test Set:** 2024 (1 Year, held-out, 2,928 records, ~4.0%) — used solely for final model benchmarking.

---

## 🤖 Section 3: Machine Learning Models & Experimental Solution

Three distinct algorithm families were trained, tuned, and compared under identical preprocessing pipelines:

### 3.1 Candidate Models
1. **Logistic Regression (L2 Regularized):**
   * *Role:* Linear probabilistic baseline.
   * *Tuning:* $C=0.1$, `l2` penalty, `class_weight='balanced'`.
2. **Random Forest Classifier (Ensemble Bagging):**
   * *Role:* Non-linear bagged ensemble capturing complex feature interactions and non-monotonic boundaries.
   * *Tuning:* `n_estimators=200`, `max_depth=16`, `min_samples_split=6`, `min_samples_leaf=2`.
3. **Gradient Boosting Classifier (Ensemble Boosting — Champion):**
   * *Role:* Sequentially optimizes pseudo-residuals of the log-loss function.
   * *Tuning:* `n_estimators=150`, `learning_rate=0.08`, `max_depth=5`, `subsample=0.85`.

---

## 📈 Section 4: Evaluation Results & Benchmarks

### 4.1 Held-Out Test Set (2024) Performance Comparison

| Model Architecture | Accuracy | ROC-AUC | Precision | Recall | F1-Score | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **🏆 Gradient Boosting (Tuned)** | **90.62%** | **0.9714** | **87.42%** | **84.50%** | **0.8593** | **0.0682** |
| **🌲 Random Forest (Tuned)** | 89.28% | 0.9632 | 85.80% | 81.90% | 0.8380 | 0.0765 |
| **📈 Logistic Regression (Tuned)** | 84.15% | 0.9080 | 78.10% | 74.30% | 0.7615 | 0.1120 |
| Baseline Gradient Boosting | 90.18% | 0.9667 | 86.90% | 83.70% | 0.8527 | 0.0715 |
| Baseline Random Forest | 88.75% | 0.9575 | 85.10% | 80.90% | 0.8295 | 0.0798 |
| Baseline Logistic Regression | 83.92% | 0.9051 | 77.80% | 74.00% | 0.7585 | 0.1145 |

### 4.2 Why Gradient Boosting was Selected
1. **Highest Discriminative Ability:** Achieved the highest **ROC-AUC (0.9714)** on completely unseen 2024 test data.
2. **Exceptional Calibration:** Lowest **Brier Score (0.0682)**, confirming that predicted probabilities reflect true empirical rainfall frequencies.
3. **Balanced F1-Score (0.8593):** Effectively minimises both False Positives (unwarranted agricultural shutdowns) and False Negatives (unpreparedness for heavy rainfall).
4. **Lightweight Deployment Footprint:** Serialized pipeline artifact is only **260 KB** (compared to 26 MB for Random Forest), allowing sub-millisecond cloud inferences on Streamlit Cloud.

---

## 🌐 Section 5: The Streamlit Web Application

The interactive web application is deployed and live at:  
👉 **[https://weatherpredictiveml-git-geavtut3pfyypewaut2she.streamlit.app/](https://weatherpredictiveml-git-geavtut3pfyypewaut2she.streamlit.app/)**

### Key Application Features:
1. **🛰️ Live Forecast & Historical Trajectories:**
   - Real-time Open-Meteo telemetry integration for 8 Maharashtra stations.
   - Interactive 7-day meteorological trajectories (Temperature, Humidity, Rain history).
   - Automated IMD warning advisory badge.
2. **🧪 Interactive "What-If" Scenario Simulator:**
   - Allows users to simulate arbitrary atmospheric conditions offline.
   - Presets: *Monsoon Heavy Downpour*, *Scorching Summer*, *Post-Monsoon Thunderstorm*, *Winter Dry Spell*.
   - Dynamic real-time response to humidity, rainfall, pressure, and temperature changes.
3. **⚖️ Multi-Model Head-to-Head Comparison:**
   - Evaluates the active scenario across all 3 tuned classifiers simultaneously with side-by-side probability gauges and a consensus summary table.
4. **🎯 Precision-Recall Decision Threshold Slider:**
   - Configurable probability threshold (`0.10` to `0.90`) in the sidebar for operational flexibility (e.g., setting threshold to `0.35` for conservative early disaster warnings).

---

## 📋 Section 6: Quick-Submission Summary for College Portal

*Use the block below if your submission portal requires a concise summary in a text field:*

```
PROJECT TITLE: WeatherCast — Next-Day Rain Prediction for Maharashtra (Case Study 75)
STUDENT NAME: Piyush Solanki
BRANCH & SEMESTER: B.Tech Computer Engineering / AI / DS — Semester V
SUBJECT: Machine Learning (CS/AI Laboratory)

DEPLOYED APPLICATION URL:
https://weatherpredictiveml-git-geavtut3pfyypewaut2she.streamlit.app/

GITHUB REPOSITORY:
https://github.com/PIYUSH-108-SOLANKI/Weather_predictive_ml-

PROBLEM SUMMARY:
Formulated a binary classification problem to predict next-day rainfall (rain_tomorrow ∈ {0, 1}) across 8 meteorological stations in Maharashtra using 25 years (2000–2024, 73,056 records) of ECMWF ERA5 reanalysis data from Open-Meteo.

METHODOLOGY & PIPELINE:
- Feature Engineering: 35 domain-specific features (1/3/7-day lags, 3/7-day rolling trailing aggregates, cyclical Fourier sine/cosine calendar projections, IMD monsoon season categoricals).
- Chronological Split: 2000–2022 (Train), 2023 (Validation / Tuning), 2024 (Held-out Test) to strictly prevent data leakage.
- Models Trained: Logistic Regression, Random Forest, and Gradient Boosting.
- Champion Model: Tuned Gradient Boosting Classifier achieving 90.62% Test Accuracy, 0.9714 ROC-AUC, 0.8593 F1-Score, and 0.0682 Brier Score.

STREAMLIT DEPLOYMENT:
Built and deployed a cloud-hosted Streamlit app featuring live real-time Open-Meteo forecasts, 7-day weather trend charts, an interactive What-If scenario simulator, and multi-model comparison.
```
