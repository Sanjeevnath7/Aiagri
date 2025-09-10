import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ---------------------------
# Generate Synthetic Data
# ---------------------------
@st.cache_data
def generate_data():
    dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq="D")
    markets = ["Coimbatore", "chennai", "tiruppur", "salem", "erode"]
    commodities = ["Banana", "Onion", "Maize"]

    data = []
    np.random.seed(0)

    for date in dates:
        for market in markets:
            for crop in commodities:
                price = np.random.randint(30, 50)  # modal price
                data.append({
                    "Date": date,
                    "Market": market,
                    "Commodity": crop,
                    "Modal Price/Kg": price
                })

    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = generate_data()

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🌾 Commodity Price Forecast (SARIMAX)")

# User inputs
market = st.selectbox("Select Market", sorted(df["Market"].unique()))
commodity = st.selectbox("Select Commodity", sorted(df["Commodity"].unique()))
future_months = st.number_input("Months ahead to forecast", min_value=1, max_value=24, value=6)

# Filter dataset
filtered_df = df[(df["Market"] == market) & (df["Commodity"] == commodity)]
monthly_df = filtered_df.groupby(pd.Grouper(key='Date', freq='M'))['Modal Price/Kg'].mean().reset_index()
monthly_df.set_index('Date', inplace=True)

# SARIMAX Model
try:
    sarima_model = SARIMAX(monthly_df['Modal Price/Kg'], order=(1,1,1), seasonal_order=(1,1,1,12))
    sarima_fit = sarima_model.fit(disp=False)

    forecast = sarima_fit.forecast(steps=future_months)
    future_dates = pd.date_range(start=monthly_df.index[-1] + pd.DateOffset(months=1),
                                 periods=future_months, freq='M')

    st.subheader(f"📢 Forecasted Prices for {commodity} in {market}")
    forecast_df = pd.DataFrame({"Month": future_dates.strftime("%Y-%m"), "Forecasted Price": forecast.round(2)})
    st.write(forecast_df)

except Exception as e:
    st.error(f"Model failed: {e}")