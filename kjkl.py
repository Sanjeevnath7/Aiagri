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
user_date = st.date_input("Enter the future date (YYYY-MM-DD)")

# Filter dataset
filtered_df = df[(df["Market"] == market) & (df["Commodity"] == commodity)]
monthly_df = filtered_df.groupby(pd.Grouper(key='Date', freq='M'))['Modal Price/Kg'].mean().reset_index()
monthly_df.set_index('Date', inplace=True)

# Forecast if user enters a future date
if st.button("Get Forecast"):
    last_date = monthly_df.index[-1]

    if pd.to_datetime(user_date) <= last_date:
        st.warning("⚠ Please enter a date after the dataset's last date.")
    else:
        # Calculate months ahead
        months_ahead = (pd.to_datetime(user_date).year - last_date.year) * 12 + (pd.to_datetime(user_date).month - last_date.month)

        try:
            # SARIMAX Model
            sarima_model = SARIMAX(monthly_df['Modal Price/Kg'], order=(1,1,1), seasonal_order=(1,1,1,12))
            sarima_fit = sarima_model.fit(disp=False)

            forecast = sarima_fit.forecast(steps=months_ahead)
            forecast_value = forecast.iloc[-1]

            st.success(f"📢 Forecasted Price for {commodity} in {market} on {user_date}: *{forecast_value:.2f} INR/kg*")

        except Exception as e:
            st.error(f"Model failed: {e}")