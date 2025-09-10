import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.title("Commodity Price Dashboard & Forecast")

# --- Use DataFrame from model.ipynb ---
# If you have saved your DataFrame as a CSV in model.ipynb, load it here:
# Example: df = pd.read_csv("data.csv", parse_dates=["Date"])

# For demonstration, here's how you would load it:
# df = pd.read_csv("your_dataframe.csv", parse_dates=["Date"])

# If you want to share the DataFrame directly between notebook and app,
# save it in model.ipynb:
# df.to_csv("data.csv", index=False)

# Then in app.py:
df = pd.read_csv("data.csv", parse_dates=["Date"])

# Sidebar for commodity selection
commodity = st.sidebar.selectbox("Select Commodity", df["Commodity"].unique())
filtered_df = df[df["Commodity"] == commodity]

st.subheader(f"Daily Prices for {commodity}")
st.dataframe(filtered_df)

# Line chart of daily prices
st.line_chart(filtered_df.set_index("Date")[["Price/Kg"]])

# Monthly SARIMA Forecast
st.subheader(f"SARIMA Monthly Forecast for {commodity}")

# Prepare monthly average price data
monthly_df = filtered_df.groupby(pd.Grouper(key='Date', freq='M'))['Price/Kg'].mean().reset_index()
monthly_df.set_index('Date', inplace=True)

if len(monthly_df) > 1:
    sarima_model = SARIMAX(monthly_df['Price/Kg'], order=(1,1,1), seasonal_order=(1,1,1,12))
    sarima_fit = sarima_model.fit(disp=False)
    forecast_steps = 12
    sarima_forecast = sarima_fit.forecast(steps=forecast_steps)
    last_month = monthly_df.index[-1]
    future_months = pd.date_range(start=last_month + pd.DateOffset(months=1), periods=forecast_steps, freq='M')

    # Plot actual and forecasted prices
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monthly_df.index, monthly_df['Price/Kg'], label="Actual Monthly Avg Price")
    ax.plot(future_months, sarima_forecast, label="SARIMA Forecast", linestyle='--', color='purple')
    ax.set_title(f"SARIMA Monthly Forecast Prices for {commodity}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Price/Kg")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.write("**Forecast for next 12 months:**")
    forecast_table = pd.DataFrame({
        "Month": [d.strftime('%Y-%m') for d in future_months],
        "Forecasted Price/Kg": sarima_forecast.round(2)
    })
    st.dataframe(forecast_table)
else:
    st.warning("Not enough monthly data for forecasting.")