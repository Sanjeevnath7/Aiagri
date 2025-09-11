import streamlit as st
import pandas as pd
import numpy as np
import datetime
import requests
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler

st.set_option("deprecation.showPyplotGlobalUse", False)

# -------------------------
# Synthetic Market Price Data
# -------------------------
@st.cache_data
def generate_data():
    dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq="D")
    markets = {
        "Coimbatore": (11.0, 76.9),
        "Chennai": (13.08, 80.27),
        "Tiruppur": (11.1, 77.3),
        "Salem": (11.65, 78.15),
        "Erode": (11.34, 77.72)
    }
    commodities = ["Banana", "Onion", "Maize"]

    data = []
    np.random.seed(0)
    for date in dates:
        for market in markets.keys():
            for crop in commodities:
                base = 40 + 5*np.sin(2*np.pi*date.timetuple().tm_yday/365)
                noise = np.random.normal(0, 2)
                price = base + noise
                data.append({
                    "Date": date,
                    "Market": market,
                    "Commodity": crop,
                    "Modal Price/Kg": max(10, min(100, price))
                })

    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    return df, markets

df, markets = generate_data()

# -------------------------
# NASA POWER API (for soil + past weather)
# -------------------------
def get_nasa_power(lat, lon, start, end):
    url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point?"
        f"parameters=T2M,PRECTOTCORR,SOILM_TOT,TSOIL0_10M&community=AG"
        f"&longitude={lon}&latitude={lat}&start={start}&end={end}&format=JSON"
    )
    r = requests.get(url)
    if r.status_code != 200:
        return None
    data = r.json()
    df = pd.DataFrame(data["properties"]["parameter"])
    df = df.T
    df.index = pd.to_datetime(df.index)
    return df

# -------------------------
# Open-Meteo API (for forecast)
# -------------------------
def get_open_meteo(lat, lon, start_date, end_date):
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,precipitation_sum&timezone=auto"
        f"&start_date={start_date}&end_date={end_date}"
    )
    r = requests.get(url)
    if r.status_code != 200:
        return None
    data = r.json()
    df = pd.DataFrame(data["daily"])
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    return df

# -------------------------
# Streamlit App
# -------------------------
st.title("🌾 Commodity Price Forecast with Weather + Soil")

market = st.selectbox("Select Market", sorted(df["Market"].unique()))
commodity = st.selectbox("Select Commodity", sorted(df["Commodity"].unique()))
user_date = st.date_input("Enter future date")

if st.button("Get Forecast"):
    filtered_df = df[(df["Market"] == market) & (df["Commodity"] == commodity)]
    monthly_df = filtered_df.groupby(pd.Grouper(key='Date', freq='M'))['Modal Price/Kg'].mean().reset_index()
    monthly_df.set_index('Date', inplace=True)

    last_date = monthly_df.index[-1]

    if pd.to_datetime(user_date) <= last_date:
        st.warning("⚠ Please enter a future date after dataset's last date.")
    else:
        # --- Get NASA Data (for visualization only) ---
        lat, lon = markets[market]
        start = monthly_df.index.min().strftime("%Y%m%d")
        end = last_date.strftime("%Y%m%d")
        nasa_df = get_nasa_power(lat, lon, start, end)

        if nasa_df is not None:
            st.subheader("🌍 Soil & Weather Conditions (Past - NASA POWER)")
            fig, ax = plt.subplots(3, 1, figsize=(8, 6))
            nasa_df["T2M"].plot(ax=ax[0], color="red", label="Temp (°C)")
            ax[0].set_ylabel("°C")
            ax[0].legend()
            nasa_df["PRECTOTCORR"].plot(ax=ax[1], color="blue", label="Rainfall (mm)")
            ax[1].set_ylabel("mm")
            ax[1].legend()
            nasa_df["SOILM_TOT"].plot(ax=ax[2], color="green", label="Soil Moisture")
            ax[2].set_ylabel("Vol")
            ax[2].legend()
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.error("NASA data fetch failed.")

        # --- Get Open-Meteo Data for Forecast ---
        start_forecast = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        end_forecast = user_date.strftime("%Y-%m-%d")
        meteo_df = get_open_meteo(lat, lon, start_forecast, end_forecast)

        if meteo_df is None:
            st.error("Open-Meteo forecast fetch failed.")
        else:
            st.subheader("🌤 Future Weather Forecast (Open-Meteo)")
            st.line_chart(meteo_df[["temperature_2m_max", "precipitation_sum"]])

            # --- Prediction ---
            y = monthly_df["Modal Price/Kg"]

            # For exog, use Open-Meteo daily avg → monthly mean
            meteo_monthly = meteo_df.resample("M").mean()
            exog = pd.DataFrame({
                "temp": meteo_monthly["temperature_2m_max"],
                "rain": meteo_monthly["precipitation_sum"]
            }).fillna(method="ffill")

            # Align y with past (train with history)
            scaler = StandardScaler()
            exog_hist = np.column_stack([
                np.random.normal(30, 5, len(y)),  # fake historical temp
                np.random.normal(5, 2, len(y))   # fake historical rain
            ])
            exog_scaled = scaler.fit_transform(exog_hist)

            try:
                model = SARIMAX(y, exog=exog_scaled, order=(1,1,1), seasonal_order=(1,1,1,12))
                fit = model.fit(disp=False)

                # Use Open-Meteo forecast
                future_exog_scaled = scaler.transform(exog.values)
                forecast = fit.forecast(steps=len(exog), exog=future_exog_scaled)
                forecast_value = forecast.iloc[-1]

                forecast_value = max(10, min(100, forecast_value))  # clamp realistic

                st.success(f"🌟 Forecasted Price for {commodity} in {market} on {user_date}: "
                           f"{forecast_value:.2f} INR/kg**")

            except Exception as e:
                st.error(f"Model failed: {e}")