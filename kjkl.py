import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ---------------------------
# Load / Generate Synthetic Data
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
                price = {
                    "Min Price/Kg": np.random.randint(15, 30),
                    "Modal Price/Kg": np.random.randint(30, 50),
                    "Max Price/Kg": np.random.randint(50, 100)
                }
                data.append({
                    "Date": date,
                    "Market": market,
                    "Commodity": crop,
                    **price
                })

    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = generate_data()

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🌾 Agri Commodity Price Dashboard")

# Sidebar filters
market = st.sidebar.selectbox("Select Market", sorted(df["Market"].unique()))
commodity = st.sidebar.selectbox("Select Commodity", sorted(df["Commodity"].unique()))

filtered_df = df[(df["Market"] == market) & (df["Commodity"] == commodity)]

st.subheader(f"📊 Price Trend: {commodity} in {market}")
st.line_chart(filtered_df.set_index("Date")["Modal Price/Kg"])

# ---------------------------
# ML Model: Linear Regression
# ---------------------------
X = pd.get_dummies(df[['Market', 'Commodity']])
y = df['Modal Price/Kg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

st.subheader("🤖 Machine Learning Model")
st.write(f"Linear Regression MAE: *{mae:.2f} INR*")

# ---------------------------
# Forecast with ARIMA
# ---------------------------
st.subheader("📈 ARIMA Forecast (Next 12 Months)")
monthly_df = df.groupby(pd.Grouper(key='Date', freq='M'))['Modal Price/Kg'].mean().reset_index()
monthly_df.set_index('Date', inplace=True)

try:
    arima_model = ARIMA(monthly_df['Modal Price/Kg'], order=(1,1,1))
    arima_fit = arima_model.fit()
    forecast_steps = 12
    arima_forecast = arima_fit.forecast(steps=forecast_steps)
    future_months = pd.date_range(start=monthly_df.index[-1] + pd.DateOffset(months=1),
                                  periods=forecast_steps, freq='M')

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(monthly_df.index, monthly_df['Modal Price/Kg'], label="Actual Price")
    ax.plot(future_months, arima_forecast, '--', label="Forecast", color="red")
    ax.set_title("ARIMA Forecast")
    ax.legend()
    st.pyplot(fig)
except Exception as e:
    st.warning(f"ARIMA failed: {e}")