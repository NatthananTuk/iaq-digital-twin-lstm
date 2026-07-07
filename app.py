"""
Minimal Streamlit dashboard for the PM2.5 LSTM.

It loads the trained model + scaler, shows recent history, and forecasts the
next 6 hours from the most recent 24 hours of data.

Run: streamlit run app.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from tensorflow.keras.models import load_model

# Resolve file paths relative to THIS script, so `streamlit run` works no
# matter which directory it is launched from.
BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR / "data" / "pm25_raw.csv"
MODEL_PATH = BASE_DIR / "lstm_pm25_model.keras"
SCALER_PATH = BASE_DIR / "scaler.pkl"
STATION_TZ = "Asia/Bangkok"              # show times in the station's local zone
WINDOW = 24                              # hours of history the model needs
HORIZON = 6                              # hours it forecasts


# @st.cache_resource keeps the model/scaler in memory across reruns (fast).
@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH)       # the trained LSTM
    scaler = joblib.load(SCALER_PATH)    # same MinMax scaler used in training
    return model, scaler


# @st.cache_data caches the loaded dataframe (it only changes when the file does).
@st.cache_data
def load_data():
    df = pd.read_csv(RAW_PATH, parse_dates=["datetime"])
    df = df.sort_values("datetime").drop_duplicates("datetime")
    df["local"] = df["datetime"].dt.tz_convert(STATION_TZ)   # UTC -> local time
    return df


def forecast_next_hours(model, scaler, recent_values):
    """Feed the last 24 hours into the model and return the next 6 hours (ug/m3)."""
    scaled = scaler.transform(recent_values.reshape(-1, 1))  # to [0,1]
    x = scaled.reshape(1, WINDOW, 1)                          # (1, 24, 1) for Keras
    pred_scaled = model.predict(x, verbose=0)                # (1, 6) scaled output
    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1))  # back to ug/m3
    return pred.ravel()                                      # 1-D array of 6 values


# ---- Page ----------------------------------------------------------------
st.set_page_config(page_title="IAQ PM2.5 Forecast", page_icon="🌫️")
st.title("🌫️ PM2.5 6-Hour Forecast")
st.caption("LSTM digital-twin demo · Pluakdaeng station, Thailand (OpenAQ)")

model, scaler = load_artifacts()         # load once (cached)
df = load_data()                         # load data (cached)

# Take the most recent 24 readings as the model input window.
recent = df.tail(WINDOW)
recent_values = recent["pm25"].to_numpy()

# Run the forecast for the next 6 hours.
forecast = forecast_next_hours(model, scaler, recent_values)

# Build the 6 future timestamps (1..6 hours after the last reading).
last_time = recent["local"].iloc[-1]
future_times = [last_time + pd.Timedelta(hours=h) for h in range(1, HORIZON + 1)]

# --- Show the latest reading and the next-hour forecast as headline numbers.
col1, col2 = st.columns(2)
col1.metric("Latest reading", f"{recent_values[-1]:.1f} µg/m³")
col2.metric("Forecast +1h", f"{forecast[0]:.1f} µg/m³",
            delta=f"{forecast[0] - recent_values[-1]:+.1f}")

# --- Chart: recent history followed by the 6-hour forecast on one timeline.
st.subheader("Recent history + forecast")
history = pd.DataFrame({"PM2.5 (history)": recent["pm25"].to_numpy()},
                       index=recent["local"])
future = pd.DataFrame({"PM2.5 (forecast)": forecast}, index=future_times)
# Concat so both lines share the x-axis; missing cells are left as NaN.
chart_df = pd.concat([history, future])
st.line_chart(chart_df)

# --- Table: the exact forecast values, hour by hour.
st.subheader("Next 6 hours")
st.dataframe(
    pd.DataFrame({
        "Time (local)": [t.strftime("%Y-%m-%d %H:%M") for t in future_times],
        "PM2.5 (µg/m³)": [round(v, 1) for v in forecast],
    }),
    hide_index=True,
)
