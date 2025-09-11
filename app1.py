import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX
import time


# ---------------------------------------------------
# Synthetic Market Price Data
# ---------------------------------------------------
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
                    "Modal Price/Kg": price
                })

    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    return df, markets

df, markets = generate_data()

# ---------------------------------------------------
# NASA POWER (for Visualization)
# ---------------------------------------------------
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

# ---------------------------------------------------
# Open-Meteo (for Prediction)
# ---------------------------------------------------
def get_open_meteo_forecast(lat, lon, days_ahead=30):
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&daily=temperature_2m_max,precipitation_sum&forecast_days={days_ahead}&timezone=auto"
    )
    r = requests.get(url)
    if r.status_code != 200:
        return None
    data = r.json()
    df = pd.DataFrame({
        "Date": data["daily"]["time"],
        "Temp_Max": data["daily"]["temperature_2m_max"],
        "Rain": data["daily"]["precipitation_sum"]
    })
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    return df

# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------
st.title("🌾 Crop Price Prediction with Weather & Soil Data")

market = st.selectbox("📍 Select Market", sorted(df["Market"].unique()))
commodity = st.selectbox("🥬 Select Commodity", sorted(df["Commodity"].unique()))
future_date = st.date_input("📅 Enter Future Date")

if st.button("🚀 Get Forecast"):
    with st.spinner("Fetching data & training model..."):
        time.sleep(1.5)

        # Filter price data
        filtered_df = df[(df["Market"] == market) & (df["Commodity"] == commodity)]
        monthly_df = filtered_df.groupby(pd.Grouper(key='Date', freq='M'))['Modal Price/Kg'].mean().reset_index()
        monthly_df.set_index('Date', inplace=True)
        last_date = monthly_df.index[-1]

        # Fetch NASA historical data (for visualization only)
        lat, lon = markets[market]
        start = monthly_df.index.min().strftime("%Y%m%d")
        end = monthly_df.index.max().strftime("%Y%m%d")
        nasa_df = get_nasa_power(lat, lon, start, end)

        if nasa_df is not None:
            nasa_monthly = nasa_df.resample("M").mean()
            fig = px.line(nasa_monthly, x=nasa_monthly.index, y=["T2M","PRECTOTCORR","SOILM_TOT","TSOIL0_10M"],
                          title="🌍 NASA POWER: Soil & Weather Trends", markers=True)
            st.plotly_chart(fig, use_container_width=True)

        # Ensure future date valid
        if pd.to_datetime(future_date) <= last_date:
            st.warning("⚠ Please enter a future date after dataset's last date.")
        else:
            months_ahead = (pd.to_datetime(future_date).year - last_date.year) * 12 + \
                           (pd.to_datetime(future_date).month - last_date.month)

            # Get Open-Meteo forecast
            openmeteo_df = get_open_meteo_forecast(lat, lon, days_ahead=months_ahead*30)
            if openmeteo_df is None:
                st.error("❌ Failed to fetch Open-Meteo forecast")
            else:
                st.write("📊 Open-Meteo Forecast (sample)", openmeteo_df.head())

                # Aggregate to monthly
                openmeteo_monthly = openmeteo_df.resample("M").mean()

                # Train SARIMAX
                y = monthly_df["Modal Price/Kg"]
                exog = openmeteo_monthly.iloc[:len(y)].reindex(y.index).fillna(method="ffill")

                try:
                    model = SARIMAX(y, exog=exog, order=(1,1,1), seasonal_order=(1,1,1,12))
                    fit = model.fit(disp=False)

                    # Future exog
                    future_exog = openmeteo_monthly.tail(months_ahead)
                    forecast = fit.forecast(steps=months_ahead, exog=future_exog)
                    forecast_value = forecast.iloc[-1]

                    # 🎬 Animated chart
                    forecast_df = pd.DataFrame({"Date": forecast.index, "Forecast Price": forecast.values})
                    fig2 = px.line(forecast_df, x="Date", y="Forecast Price", title="📈 Forecast Animation", markers=True)
                    fig2.update_traces(line=dict(dash="dot"))
                    st.plotly_chart(fig2, use_container_width=True)

                    st.success(f"🌟 Forecasted Price for {commodity} in {market} on {future_date}: "
                               f"{forecast_value:.2f} INR/kg**")

                except Exception as e:
                    st.error(f"Model failed: {e}")