import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
from pathlib import Path
from datetime import datetime

# ─── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WeatherCast — Maharashtra Rain Prediction",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Constants & Configuration ────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_FILES = {
    "Gradient Boosting (Tuned - Production)": "best_model_phase6.joblib",
    "Random Forest (Tuned)": "random_forest_tuned.joblib",
    "Logistic Regression (Tuned)": "logistic_regression_tuned.joblib",
}

LOCATIONS = {
    "Mumbai": {"latitude": 19.07283, "longitude": 72.88261, "region": "Konkan"},
    "Ratnagiri": {"latitude": 16.99154, "longitude": 73.31022, "region": "Konkan"},
    "Pune": {"latitude": 18.51957, "longitude": 73.85535, "region": "Madhya Maharashtra"},
    "Nashik": {"latitude": 19.99727, "longitude": 73.79096, "region": "Madhya Maharashtra"},
    "Kolhapur": {"latitude": 16.69563, "longitude": 74.23167, "region": "Madhya Maharashtra"},
    "Solapur": {"latitude": 17.67152, "longitude": 75.91044, "region": "Madhya Maharashtra"},
    "Chhatrapati Sambhajinagar": {"latitude": 19.87757, "longitude": 75.34226, "region": "Marathwada"},
    "Nagpur": {"latitude": 21.14631, "longitude": 79.08491, "region": "Vidarbha"},
}

NUM_COLS = [
    'temperature_2m_max', 'temperature_2m_min', 'rain_sum', 'precipitation_hours',
    'sunshine_duration', 'wind_speed_10m_max', 'wind_gusts_10m_max', 'wind_direction_10m_dominant',
    'relative_humidity_2m_mean', 'relative_humidity_2m_max', 'relative_humidity_2m_min', 'pressure_msl_mean',
    'latitude', 'longitude', 'month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos',
    'rain_sum_lag_1', 'rain_sum_lag_3', 'rain_sum_lag_7',
    'temperature_max_lag_1', 'temperature_min_lag_1',
    'humidity_mean_lag_1', 'pressure_mean_lag_1', 'wind_speed_max_lag_1',
    'rain_sum_3d_total', 'rain_sum_7d_total',
    'temperature_max_3d_mean', 'temperature_min_3d_mean',
    'humidity_3d_mean', 'pressure_3d_mean'
]
CAT_COLS = ['location', 'weather_code', 'season']
FEATURE_COLS = NUM_COLS + CAT_COLS

WMO_CODE_MAP = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail"
}

# ─── Model Loading ────────────────────────────────────────────────────────────
@st.cache_resource
def load_all_models():
    """Loads all tuned models into memory cache."""
    loaded = {}
    for label, fname in MODEL_FILES.items():
        path = MODELS_DIR / fname
        if path.exists():
            loaded[label] = joblib.load(path)
    return loaded

# ─── Open-Meteo API Fetcher ───────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_weather_data(lat: float, lon: float):
    """Fetches 8 past days + 1 forecast day from Open-Meteo API."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,rain_sum,precipitation_hours,sunshine_duration,wind_speed_10m_max,wind_gusts_10m_max,wind_direction_10m_dominant",
        "hourly": "relative_humidity_2m,pressure_msl",
        "past_days": 8,
        "forecast_days": 1,
        "timezone": "Asia/Kolkata",
    }
    response = requests.get(url, params=params, timeout=12)
    response.raise_for_status()
    return response.json()

# ─── Feature Engineering from Live API ────────────────────────────────────────
def compute_features(api_data: dict, city_name: str, lat: float, lon: float):
    daily_raw = api_data.get("daily", {})
    if not daily_raw or "time" not in daily_raw:
        raise ValueError("Open-Meteo API response missing daily data.")

    df_daily = pd.DataFrame(daily_raw).rename(columns={"time": "date"})
    df_daily["date"] = pd.to_datetime(df_daily["date"])

    hourly_raw = api_data.get("hourly", {})
    if not hourly_raw or "time" not in hourly_raw:
        raise ValueError("Open-Meteo API response missing hourly data.")

    df_hourly = pd.DataFrame(hourly_raw)
    df_hourly["date"] = pd.to_datetime(df_hourly["time"]).dt.date
    daily_agg = (
        df_hourly.groupby("date")
        .agg(
            relative_humidity_2m_mean=("relative_humidity_2m", "mean"),
            relative_humidity_2m_max=("relative_humidity_2m", "max"),
            relative_humidity_2m_min=("relative_humidity_2m", "min"),
            pressure_msl_mean=("pressure_msl", "mean"),
        )
        .reset_index()
    )
    daily_agg["date"] = pd.to_datetime(daily_agg["date"])

    df_merged = df_daily.merge(daily_agg, on="date", how="left").sort_values("date").reset_index(drop=True)
    df_merged["location"] = city_name
    df_merged["latitude"] = lat
    df_merged["longitude"] = lon

    # Lags
    df_merged["rain_sum_lag_1"] = df_merged["rain_sum"].shift(1)
    df_merged["rain_sum_lag_3"] = df_merged["rain_sum"].shift(3)
    df_merged["rain_sum_lag_7"] = df_merged["rain_sum"].shift(7)

    df_merged["temperature_max_lag_1"] = df_merged["temperature_2m_max"].shift(1)
    df_merged["temperature_min_lag_1"] = df_merged["temperature_2m_min"].shift(1)

    df_merged["humidity_mean_lag_1"] = df_merged["relative_humidity_2m_mean"].shift(1)
    df_merged["pressure_mean_lag_1"] = df_merged["pressure_msl_mean"].shift(1)
    df_merged["wind_speed_max_lag_1"] = df_merged["wind_speed_10m_max"].shift(1)

    # Rolling Windows
    df_merged["rain_sum_3d_total"] = df_merged["rain_sum"].shift(1).rolling(3, min_periods=3).sum()
    df_merged["rain_sum_7d_total"] = df_merged["rain_sum"].shift(1).rolling(7, min_periods=7).sum()

    df_merged["temperature_max_3d_mean"] = df_merged["temperature_2m_max"].shift(1).rolling(3, min_periods=3).mean()
    df_merged["temperature_min_3d_mean"] = df_merged["temperature_2m_min"].shift(1).rolling(3, min_periods=3).mean()

    df_merged["humidity_3d_mean"] = df_merged["relative_humidity_2m_mean"].shift(1).rolling(3, min_periods=3).mean()
    df_merged["pressure_3d_mean"] = df_merged["pressure_msl_mean"].shift(1).rolling(3, min_periods=3).mean()

    # Target day row (Today = index -1)
    today_row = df_merged.iloc[[-1]].copy()
    m = today_row["date"].dt.month.values[0]
    doy = today_row["date"].dt.dayofyear.values[0]

    today_row["month_sin"] = np.sin(2 * np.pi * m / 12.0)
    today_row["month_cos"] = np.cos(2 * np.pi * m / 12.0)
    today_row["day_of_year_sin"] = np.sin(2 * np.pi * doy / 365.25)
    today_row["day_of_year_cos"] = np.cos(2 * np.pi * doy / 365.25)

    def get_imd_season(month: int) -> str:
        if month in [1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Summer_PreMonsoon"
        elif month in [6, 7, 8, 9]:
            return "Southwest_Monsoon"
        else:
            return "Post_Monsoon"

    today_row["season"] = get_imd_season(m)
    today_row["weather_code"] = int(today_row["weather_code"].values[0])

    context_values = {
        "date": today_row["date"].dt.strftime("%d %B %Y").values[0],
        "weather_code": today_row["weather_code"].values[0],
        "weather_desc": WMO_CODE_MAP.get(today_row["weather_code"].values[0], "Unknown Weather"),
        "temp_max": today_row["temperature_2m_max"].values[0],
        "temp_min": today_row["temperature_2m_min"].values[0],
        "humidity_mean": today_row["relative_humidity_2m_mean"].values[0],
        "pressure_mean": today_row["pressure_msl_mean"].values[0],
        "rain_today": today_row["rain_sum"].values[0],
        "rain_lag1": today_row["rain_sum_lag_1"].values[0],
        "rain_3d_sum": today_row["rain_sum_3d_total"].values[0],
        "season": today_row["season"].values[0],
    }

    X_input = today_row[FEATURE_COLS]
    return X_input, context_values, df_merged

# ─── Sidebar Controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/cloud-lighting--v1.png", width=64)
    st.title("WeatherCast 🌦️")
    st.caption("**Case Study 75** · Maharashtra Rain Classification")
    st.markdown("---")

    st.subheader("⚙️ Model Configuration")
    selected_model_name = st.selectbox(
        "Active Classifier:",
        options=list(MODEL_FILES.keys()) + ["⚖️ Compare All Tuned Models"],
        index=0,
        help="Select which hyperparameter-tuned pipeline to use for inference, or compare all 3 side-by-side."
    )

    decision_threshold = st.slider(
        "🎯 Rain Decision Threshold:",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Custom probability cut-off. Default: 0.50. Lower values (e.g. 0.35) yield higher recall / early warning alerts."
    )

    if decision_threshold < 0.45:
        st.info("⚠️ **Conservative / High Recall Mode**: Model will alert for rain even at lower probability scores.")
    elif decision_threshold > 0.55:
        st.info("🔒 **Strict / High Precision Mode**: Model only flags rain when confidence is exceptionally high.")

    st.markdown("---")
    st.subheader("📍 Station Coverage")
    st.markdown("8 Representative Stations across Maharashtra:")
    for city, info in LOCATIONS.items():
        st.markdown(f"- **{city}** ({info['region']})")

    st.markdown("---")
    st.caption("ECMWF Open-Meteo · Scikit-Learn · B.Tech ML Sem V")

# ─── App Header ───────────────────────────────────────────────────────────────
st.title("WeatherCast: Maharashtra Rain Predictor 🌦️")
st.markdown(
    "Predicting **next-day rainfall (`rain_tomorrow` ∈ {0, 1})** across meteorological sub-divisions of Maharashtra "
    "using 25 years of reanalysis data and machine learning pipelines."
)

models = load_all_models()
if not models:
    st.error("No model artifacts found in `models/`. Please check project files.")
    st.stop()

# ─── Navigation Tabs ──────────────────────────────────────────────────────────
tab_live, tab_sim, tab_compare, tab_bench = st.tabs([
    "🛰️ Live Real-Time Forecast",
    "🧪 What-If Scenario Simulator",
    "⚖️ Multi-Model Comparison",
    "📊 Benchmarks & Architecture"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: LIVE FORECAST & TRENDS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_live:
    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        live_city = st.selectbox(
            "📍 Select Maharashtra City / Station:",
            options=list(LOCATIONS.keys()),
            index=2, # Pune
            key="live_city_select"
        )
    with col_btn:
        st.write("")
        st.write("")
        fetch_btn = st.button("🔄 Fetch & Predict", type="primary", use_container_width=True)

    city_info = LOCATIONS[live_city]
    st.caption(f"**Station:** {live_city} | **Region:** {city_info['region']} | **Coordinates:** {city_info['latitude']}°N, {city_info['longitude']}°E | **Source:** Open-Meteo Reanalysis API")

    # Run prediction automatically or on button click
    with st.spinner(f"Fetching atmospheric telemetry for {live_city}..."):
        try:
            raw_data = fetch_weather_data(city_info["latitude"], city_info["longitude"])
            X_live, ctx, df_trend = compute_features(raw_data, live_city, city_info["latitude"], city_info["longitude"])
            
            # Predict with champion model
            champion_pipe = models.get("Gradient Boosting (Tuned - Production)", list(models.values())[0])
            prob_rain = float(champion_pipe.predict_proba(X_live)[0, 1])
            pred_rain = 1 if prob_rain >= decision_threshold else 0

            st.divider()

            # Prediction Banner
            res_col1, res_col2, res_col3 = st.columns([2, 1.5, 1.5])
            with res_col1:
                if pred_rain == 1:
                    st.error("### 🌧️ RAIN TOMORROW: YES")
                    st.markdown(f"Rain is predicted tomorrow (**threshold: {decision_threshold:.2f}**).")
                else:
                    st.success("### ☀️ RAIN TOMORROW: NO")
                    st.markdown(f"No significant rain expected tomorrow (**threshold: {decision_threshold:.2f}**).")

            with res_col2:
                st.metric(
                    label="Calibrated Rain Probability",
                    value=f"{prob_rain * 100:.1f}%",
                    delta=f"{'+' if prob_rain >= decision_threshold else '-'}{abs(prob_rain - decision_threshold)*100:.1f}% vs threshold"
                )

            with res_col3:
                # IMD Advisory badge
                if prob_rain >= 0.70:
                    st.warning("⚠️ **IMD Advisory**: High likelihood of widespread precipitation. Plan farm/travel activities accordingly.")
                elif prob_rain >= 0.40:
                    st.info("⛅ **IMD Advisory**: Moderate chance of isolated or patchy showers.")
                else:
                    st.info("☀️ **IMD Advisory**: Favorable dry conditions anticipated.")

            # Confidence Progress Bar
            st.progress(prob_rain, text=f"Model Confidence: {prob_rain * 100:.1f}% probability of measurable rain (rain_sum > 0 mm)")

            # Atmospheric Context Cards
            st.markdown(f"#### 📊 Telemetry Context for {live_city} ({ctx['date']})")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Max Temp", f"{ctx['temp_max']:.1f} °C")
            m2.metric("Min Temp", f"{ctx['temp_min']:.1f} °C")
            m3.metric("Mean Humidity", f"{ctx['humidity_mean']:.1f} %")
            m4.metric("Mean Pressure", f"{ctx['pressure_mean']:.1f} hPa")

            m5, m6, m7, m8 = st.columns(4)
            m5.metric("Today's Rain", f"{ctx['rain_today']:.1f} mm")
            m6.metric("Yesterday's Rain", f"{ctx['rain_lag1']:.1f} mm" if pd.notna(ctx['rain_lag1']) else "0.0 mm")
            m7.metric("Trailing 3-Day Rain", f"{ctx['rain_3d_sum']:.1f} mm" if pd.notna(ctx['rain_3d_sum']) else "0.0 mm")
            m8.metric("Sky Condition", f"{ctx['weather_desc']}")

            # 7-Day Atmospheric Trend Charts
            st.markdown("---")
            st.markdown(f"#### 📈 7-Day Meteorological Trajectory ({live_city})")
            st.caption("Historical build-up over the trailing 7 days leading to today.")

            chart_df = df_trend[["date", "temperature_2m_max", "temperature_2m_min", "relative_humidity_2m_mean", "rain_sum"]].dropna().copy()
            chart_df["date_str"] = chart_df["date"].dt.strftime("%d %b")

            c_tab1, c_tab2, c_tab3 = st.tabs(["🌡️ Temperature Range", "💧 Atmospheric Moisture (Humidity)", "🌧️ Rainfall History"])
            with c_tab1:
                st.line_chart(chart_df.set_index("date_str")[["temperature_2m_max", "temperature_2m_min"]])
            with c_tab2:
                st.line_chart(chart_df.set_index("date_str")["relative_humidity_2m_mean"])
            with c_tab3:
                st.bar_chart(chart_df.set_index("date_str")["rain_sum"])

        except Exception as e:
            st.error(f"⚠️ Live data retrieval failed: {e}. You can still use the What-If Simulator tab.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: WHAT-IF SCENARIO SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════
with tab_sim:
    st.markdown("### 🧪 What-If Weather Simulator (Interactive Stress Testing)")
    st.markdown("Adjust atmospheric indicators using the controls below to observe how the trained pipeline responds in real-time.")

    # Preset scenarios
    preset = st.selectbox(
        "⚡ Quick Scenarios / Presets:",
        ["Custom Inputs", "Monsoon Heavy Downpour", "Scorching Dry Summer", "Post-Monsoon Thunderstorm", "Winter Clear Dry"]
    )

    # Defaults based on preset
    def_temp_max, def_temp_min = 32.0, 23.0
    def_hum, def_pressure = 75.0, 1008.0
    def_rain, def_rain_3d = 12.0, 35.0
    def_season = "Southwest_Monsoon"
    def_wcode = 61 # Slight rain

    if preset == "Monsoon Heavy Downpour":
        def_temp_max, def_temp_min = 28.0, 22.0
        def_hum, def_pressure = 92.0, 1002.0
        def_rain, def_rain_3d = 45.0, 110.0
        def_season = "Southwest_Monsoon"
        def_wcode = 65
    elif preset == "Scorching Dry Summer":
        def_temp_max, def_temp_min = 41.0, 28.0
        def_hum, def_pressure = 25.0, 1006.0
        def_rain, def_rain_3d = 0.0, 0.0
        def_season = "Summer_PreMonsoon"
        def_wcode = 0
    elif preset == "Post-Monsoon Thunderstorm":
        def_temp_max, def_temp_min = 33.0, 24.0
        def_hum, def_pressure = 80.0, 1007.0
        def_rain, def_rain_3d = 18.0, 22.0
        def_season = "Post_Monsoon"
        def_wcode = 95
    elif preset == "Winter Clear Dry":
        def_temp_max, def_temp_min = 29.0, 14.0
        def_hum, def_pressure = 40.0, 1016.0
        def_rain, def_rain_3d = 0.0, 0.0
        def_season = "Winter"
        def_wcode = 0

    st.markdown("---")
    sim_col1, sim_col2, sim_col3 = st.columns(3)

    with sim_col1:
        sim_city = st.selectbox("Simulated City:", options=list(LOCATIONS.keys()), index=2)
        sim_season = st.selectbox(
            "IMD Meteorological Season:",
            ["Southwest_Monsoon", "Summer_PreMonsoon", "Post_Monsoon", "Winter"],
            index=["Southwest_Monsoon", "Summer_PreMonsoon", "Post_Monsoon", "Winter"].index(def_season)
        )
        sim_wcode = st.selectbox(
            "Current Sky Condition (WMO):",
            options=list(WMO_CODE_MAP.keys()),
            format_func=lambda x: f"{x}: {WMO_CODE_MAP[x]}",
            index=list(WMO_CODE_MAP.keys()).index(def_wcode) if def_wcode in WMO_CODE_MAP else 0
        )

    with sim_col2:
        sim_temp_max = st.slider("Max Temperature (°C):", 15.0, 48.0, float(def_temp_max), 0.5)
        sim_temp_min = st.slider("Min Temperature (°C):", 8.0, 35.0, float(def_temp_min), 0.5)
        sim_humidity = st.slider("Mean Relative Humidity (%):", 10.0, 100.0, float(def_hum), 1.0)
        sim_pressure = st.slider("Mean Sea-Level Pressure (hPa):", 990.0, 1030.0, float(def_pressure), 0.5)

    with sim_col3:
        sim_rain_today = st.slider("Today's Rainfall (mm):", 0.0, 200.0, float(def_rain), 0.5)
        sim_rain_lag1 = st.slider("Yesterday's Rain (mm):", 0.0, 150.0, float(def_rain * 0.8), 0.5)
        sim_rain_3d = st.slider("Trailing 3-Day Total Rain (mm):", 0.0, 350.0, float(def_rain_3d), 1.0)
        sim_wind_max = st.slider("Max Wind Speed (km/h):", 0.0, 80.0, 18.0, 1.0)

    # Build feature row for simulator
    sim_meta = LOCATIONS[sim_city]
    now_date = datetime.now()
    m_val = {"Southwest_Monsoon": 7, "Summer_PreMonsoon": 4, "Post_Monsoon": 10, "Winter": 1}[sim_season]
    doy_val = m_val * 30

    sim_row = {
        'temperature_2m_max': sim_temp_max,
        'temperature_2m_min': sim_temp_min,
        'rain_sum': sim_rain_today,
        'precipitation_hours': 6.0 if sim_rain_today > 0 else 0.0,
        'sunshine_duration': 14000.0 if sim_rain_today == 0 else 4000.0,
        'wind_speed_10m_max': sim_wind_max,
        'wind_gusts_10m_max': sim_wind_max * 1.35,
        'wind_direction_10m_dominant': 240.0 if sim_season == "Southwest_Monsoon" else 70.0,
        'relative_humidity_2m_mean': sim_humidity,
        'relative_humidity_2m_max': min(100.0, sim_humidity + 10.0),
        'relative_humidity_2m_min': max(10.0, sim_humidity - 15.0),
        'pressure_msl_mean': sim_pressure,
        'latitude': sim_meta["latitude"],
        'longitude': sim_meta["longitude"],
        'month_sin': np.sin(2 * np.pi * m_val / 12.0),
        'month_cos': np.cos(2 * np.pi * m_val / 12.0),
        'day_of_year_sin': np.sin(2 * np.pi * doy_val / 365.25),
        'day_of_year_cos': np.cos(2 * np.pi * doy_val / 365.25),
        'rain_sum_lag_1': sim_rain_lag1,
        'rain_sum_lag_3': sim_rain_today * 0.5,
        'rain_sum_lag_7': sim_rain_today * 0.3,
        'temperature_max_lag_1': sim_temp_max,
        'temperature_min_lag_1': sim_temp_min,
        'humidity_mean_lag_1': sim_humidity,
        'pressure_mean_lag_1': sim_pressure,
        'wind_speed_max_lag_1': sim_wind_max,
        'rain_sum_3d_total': sim_rain_3d,
        'rain_sum_7d_total': sim_rain_3d * 1.8,
        'temperature_max_3d_mean': sim_temp_max,
        'temperature_min_3d_mean': sim_temp_min,
        'humidity_3d_mean': sim_humidity,
        'pressure_3d_mean': sim_pressure,
        'location': sim_city,
        'weather_code': int(sim_wcode),
        'season': sim_season,
    }

    X_sim = pd.DataFrame([sim_row])[FEATURE_COLS]

    st.markdown("---")
    st.subheader("🔮 Simulated Prediction Results")

    sim_res_col1, sim_res_col2 = st.columns([1, 1])

    # Gradient Boosting prediction
    gb_pipe = models["Gradient Boosting (Tuned - Production)"]
    sim_prob = float(gb_pipe.predict_proba(X_sim)[0, 1])
    sim_pred = 1 if sim_prob >= decision_threshold else 0

    with sim_res_col1:
        if sim_pred == 1:
            st.error(f"### 🌧️ SIMULATED RESULT: RAIN (Probability: {sim_prob*100:.1f}%)")
        else:
            st.success(f"### ☀️ SIMULATED RESULT: NO RAIN (Probability: {sim_prob*100:.1f}%)")
        st.progress(sim_prob, text=f"Simulated Rain Probability: {sim_prob*100:.1f}% (Threshold: {decision_threshold:.2f})")

    with sim_res_col2:
        st.markdown("**Key Meteorological Drivers Influencing Result:**")
        if sim_humidity > 80:
            st.markdown("✅ **High Humidity (>80%)**: Strongly primes the atmosphere for rainfall.")
        if sim_season == "Southwest_Monsoon":
            st.markdown("✅ **Southwest Monsoon Season**: Highest historical rain prior in Maharashtra.")
        if sim_rain_today > 5.0 or sim_rain_3d > 20.0:
            st.markdown("✅ **Persistent Precipitation**: Consecutive rainy days heavily correlate with next-day rain.")
        if sim_pressure < 1005:
            st.markdown("✅ **Low Pressure Trough (<1005 hPa)**: Atmospheric depression encourages cloud formation.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: MULTI-MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("### ⚖️ Multi-Model Head-to-Head Comparison")
    st.markdown(
        "Evaluate the exact same input features simultaneously across all 3 hyperparameter-tuned classifiers "
        "trained during Phase 6."
    )

    # Use X_live if available, else X_sim
    eval_df = X_live if 'X_live' in locals() else X_sim
    eval_city = live_city if 'live_city' in locals() else sim_city

    st.caption(f"Evaluating telemetry for: **{eval_city}**")

    m_cols = st.columns(3)
    results_summary = []

    for i, (m_label, pipe) in enumerate(models.items()):
        prob = float(pipe.predict_proba(eval_df)[0, 1])
        cls_pred = 1 if prob >= decision_threshold else 0
        results_summary.append({"Model": m_label, "Rain Probability": f"{prob*100:.1f}%", "Prediction": "RAIN" if cls_pred == 1 else "NO RAIN"})

        with m_cols[i]:
            st.markdown(f"#### {m_label}")
            if cls_pred == 1:
                st.error("🌧️ **RAIN TOMORROW**")
            else:
                st.success("☀️ **NO RAIN**")
            st.metric("Rain Probability", f"{prob*100:.1f}%")
            st.progress(prob)

    st.markdown("---")
    st.markdown("#### 📋 Consensus Summary Table")
    st.table(pd.DataFrame(results_summary))

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: BENCHMARKS & ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_bench:
    st.markdown("### 📊 Test Set Benchmark Performance (Phase 6 Final)")
    st.markdown("Evaluated strictly on held-out **Test Year (2024)** across all 8 stations without temporal data leakage:")

    benchmarks_data = {
        "Model": [
            "🏆 Gradient Boosting (Tuned - Final)",
            "🌲 Random Forest (Tuned)",
            "📈 Logistic Regression (Tuned)",
            "Baseline Gradient Boosting",
            "Baseline Random Forest",
            "Baseline Logistic Regression"
        ],
        "Accuracy": ["90.62%", "89.28%", "84.15%", "90.18%", "88.75%", "83.92%"],
        "ROC-AUC": ["0.9714", "0.9632", "0.9080", "0.9667", "0.9575", "0.9051"],
        "Precision": ["87.42%", "85.80%", "78.10%", "86.90%", "85.10%", "77.80%"],
        "Recall": ["84.50%", "81.90%", "74.30%", "83.70%", "80.90%", "74.00%"],
        "F1-Score": ["0.8593", "0.8380", "0.7615", "0.8527", "0.8295", "0.7585"]
    }
    st.dataframe(pd.DataFrame(benchmarks_data), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🏗️ Feature Engineering Architecture (35 Schema)")
    arch1, arch2 = st.columns(2)
    with arch1:
        st.markdown("""
        **1. Temporal Lag Features (1, 3, 7 Days):**
        - `rain_sum_lag_1`, `rain_sum_lag_3`, `rain_sum_lag_7`
        - `temperature_max_lag_1`, `temperature_min_lag_1`
        - `humidity_mean_lag_1`, `pressure_mean_lag_1`, `wind_speed_max_lag_1`
        
        **2. Rolling Trailing Aggregations:**
        - `rain_sum_3d_total`, `rain_sum_7d_total` (Cumulative rainfall)
        - `temperature_max_3d_mean`, `temperature_min_3d_mean`
        - `humidity_3d_mean`, `pressure_3d_mean`
        """)

    with arch2:
        st.markdown("""
        **3. Cyclical Calendar Encodings (Fourier Harmonic):**
        - `month_sin`, `month_cos` (Captures annual cycle smoothly)
        - `day_of_year_sin`, `day_of_year_cos`
        
        **4. Meteorological Domain Categoricals:**
        - `location` (OneHotEncoded across 8 stations)
        - `season` (IMD monsoon sub-divisions)
        - `weather_code` (WMO meteorological sky condition)
        """)

st.divider()
st.caption("WeatherCast ML Case Study 75 · Academic Year 2024-2026 · Powered by Streamlit & Scikit-Learn")
