# WeatherCast — Target Variable Definition & Validation

**Phase 2 | ML Case Study 75: Weather Pattern Analysis**

---

## 1. Why Binary Classification?

The project objective is to predict whether it will rain on the next calendar day for a given location in Maharashtra. 

In meteorological modeling and operational decision-making (agriculture, water resource management, urban flood preparedness), the primary question of interest is categorical:
> *"Will tomorrow experience measurable rainfall or remain dry?"*

While predicting rainfall volume (regression on millimeters of rain) is valuable, quantitative precipitation forecasting suffers from extreme zero-inflation (~58% dry days) and long-tailed skewed distributions. A **binary classification task** (`rain_tomorrow` ∈ {0, 1}) establishes an essential first-stage classification model that delivers clear probabilistic outputs (e.g. 80% likelihood of rain tomorrow).

---

## 2. Exact Target Definition

| Target Variable | Encoding | Condition on Next Calendar Day ($t+1$) | Description |
|---|---|---|---|
| `rain_tomorrow` | `1` | `rain_sum` > 0.0 mm | Measurable liquid precipitation on the next calendar day |
| `rain_tomorrow` | `0` | `rain_sum` == 0.0 mm | Completely dry next calendar day |
| `rain_tomorrow` | `NaN` | Date not consecutive or missing | Unsupervised / dropped from supervised training |

> **Zero Leakage Rule:** No feature from day $t+1$ is ever utilized as an input feature for day $t$. The target `rain_tomorrow` is purely the ground truth label used for supervised model training and evaluation.

---

## 3. Date-Continuity Rule

Weather records represent sequential, location-specific time-series data. Simply invoking:
```python
df.groupby("location")["rain_sum"].shift(-1)
```
is **strictly invalid** without temporal delta verification, because `shift(-1)` grabs the next chronological row regardless of whether days, months, or years intervened.

### Formulation:
For an observation at index $i$ with date $D_i$ for location $L$:
$$\Delta_{\text{days}} = D_{i+1} - D_i$$

$$\text{rain\_tomorrow}_i = 
\begin{cases} 
1, & \text{if } \Delta_{\text{days}} = 1 \text{ and } \text{rain\_sum}_{i+1} > 0 \\
0, & \text{if } \Delta_{\text{days}} = 1 \text{ and } \text{rain\_sum}_{i+1} = 0 \\
\text{NaN}, & \text{if } \Delta_{\text{days}} \neq 1 \text{ or } D_{i+1} \text{ is null}
\end{cases}$$

---

## 4. Handling Known Discontinuities

### A. Chhatrapati Sambhajinagar (Year 2002 Missing Gap)
- In the raw dataset, Chhatrapati Sambhajinagar contains observations for 2000–2001 and 2003–2024 (2002 failed due to persistent API timeouts during acquisition).
- On `2001-12-31`, the immediate next recorded observation is `2003-01-01`.
- Time delta:
  $$\Delta_{\text{days}} = \text{2003-01-01} - \text{2001-12-31} = 366 \text{ days}$$
- Because $\Delta_{\text{days}} \neq 1$, `rain_tomorrow` for `2001-12-31` is **strictly set to `NaN`**.
- This guarantees that we never erroneously treat weather in 2003 as "tomorrow's weather" for 2001.

### B. Horizon Termination (`2024-12-31`)
- The historical collection window ends on `2024-12-31`.
- For all 8 locations, there is no subsequent record on `2025-01-01`.
- Consequently, for all 8 cities on `2024-12-31`, `next_date` is `NaT` and `rain_tomorrow` is **strictly set to `NaN`**.

---

## 5. Summary of Missing Target Rows

Total rows in dataset: **72,691**  
Total valid target rows: **72,682**  
Total missing target rows: **9**

| Location | Region | Date | Reason for `NaN` Target | Next Recorded Date | Delta (Days) |
|---|---|---|---|---|---|
| **Chhatrapati Sambhajinagar** | Marathwada | `2001-12-31` | Year 2002 gap | `2003-01-01` | 366 days |
| **Chhatrapati Sambhajinagar** | Marathwada | `2024-12-31` | Dataset horizon end | None (`NaT`) | — |
| **Kolhapur** | Madhya Maharashtra | `2024-12-31` | Dataset horizon end | None (`NaT`) | — |
| **Mumbai** | Konkan | `2024-12-31` | Dataset horizon end | None (`NaT`) | — |
| **Nagpur** | Vidarbha | `2024-12-31` | Dataset horizon end | None (`NaT`) | — |
| **Nashik** | Madhya Maharashtra | `2024-12-31` | Dataset horizon end | None (`NaT`) | — |
| **Pune** | Madhya Maharashtra | `2024-12-31` | Dataset horizon end | None (`NaT`) | — |
| **Ratnagiri** | Konkan | `2024-12-31` | Dataset horizon end | None (`NaT`) | — |
| **Solapur** | Madhya Maharashtra | `2024-12-31` | Dataset horizon end | None (`NaT`) | — |

---

## 6. Target Class Distribution

### Overall Class Balance (72,682 Valid Target Observations)

| Class | Meaning | Count | Percentage |
|---|---|---|---|
| **0** | No rain tomorrow (`rain_sum == 0 mm`) | 41,995 | **57.78%** |
| **1** | Rain tomorrow (`rain_sum > 0 mm`) | 30,687 | **42.22%** |
| **Total** | Valid target instances | 72,682 | 100.00% |

### Location-wise Breakdown

| Location | Region | Valid Days | No Rain (0) | No Rain (%) | Rain (1) | Rain (%) |
|---|---|---|---|---|---|---|
| **Chhatrapati Sambhajinagar** | Marathwada | 8,765 | 5,352 | 61.06% | 3,413 | 38.94% |
| **Kolhapur** | Madhya Maharashtra | 9,131 | 4,979 | 54.53% | 4,152 | 45.47% |
| **Mumbai** | Konkan | 9,131 | 5,508 | 60.32% | 3,623 | 39.68% |
| **Nagpur** | Vidarbha | 9,131 | 5,742 | 62.89% | 3,389 | 37.11% |
| **Nashik** | Madhya Maharashtra | 9,131 | 5,366 | 58.77% | 3,765 | 41.23% |
| **Pune** | Madhya Maharashtra | 9,131 | 4,928 | 53.97% | 4,203 | 46.03% |
| **Ratnagiri** | Konkan | 9,131 | 4,796 | 52.52% | 4,335 | 47.48% |
| **Solapur** | Madhya Maharashtra | 9,131 | 5,324 | 58.31% | 3,807 | 41.69% |

---

## 7. Storage and Dataset Artifacts

- **Raw Dataset (Unmodified):** [`data/raw/maharashtra_weather_2000_2024.csv`](file:///Users/piyushsolanki/Desktop/projects_sem5/ML/WeatherCast/data/raw/maharashtra_weather_2000_2024.csv) (72,691 rows × 19 columns)
- **Processed Dataset with Target:** [`data/processed/maharashtra_weather_with_target.csv`](file:///Users/piyushsolanki/Desktop/projects_sem5/ML/WeatherCast/data/processed/maharashtra_weather_with_target.csv) (72,691 rows × 20 columns)
- **Master ML Workflow Notebook:** [`notebooks/WeatherCast_ML.ipynb`](file:///Users/piyushsolanki/Desktop/projects_sem5/ML/WeatherCast/notebooks/WeatherCast_ML.ipynb)
