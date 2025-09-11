# agri_connect_openmeteo.py
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import requests
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ----------------------------
# Dummy users
# ----------------------------
if 'users' not in st.session_state:
    st.session_state.users = {"farmer": "123"}  # default

if 'posts' not in st.session_state:
    st.session_state.posts = []

if 'marketplace' not in st.session_state:
    st.session_state.marketplace = []

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = None

# ----------------------------
# Sidebar Login
# ----------------------------
st.sidebar.title("🌱 Farmer Login")

if st.session_state.logged_in:
    st.sidebar.success(f"Welcome, {st.session_state.logged_in}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = None
else:
    action = st.sidebar.radio("Select Action:", ["Login", "Register"])
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if action == "Register":
        if st.sidebar.button("Sign Up"):
            if username in st.session_state.users:
                st.sidebar.warning("User already exists!")
            else:
                st.session_state.users[username] = password
                st.sidebar.success("User registered successfully!")

    if action == "Login":
        if st.sidebar.button("Login"):
            if username in st.session_state.users and st.session_state.users[username] == password:
                st.session_state.logged_in = username
                st.sidebar.success(f"Logged in as {username}")
            else:
                st.sidebar.error("Invalid credentials")

# ----------------------------
# Generate synthetic price data
# ----------------------------
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
                price = np.random.randint(30, 50)
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

# ----------------------------
# Open-Meteo Forecast
# ----------------------------
def get_future_weather(lat, lon, days_ahead=30):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=auto&forecast_days={min(days_ahead,16)}"
    r = requests.get(url)
    data = r.json()
    df = pd.DataFrame({
        "Date": pd.to_datetime(data["daily"]["time"]),
        "T2M": (np.array(data["daily"]["temperature_2m_max"]) + np.array(data["daily"]["temperature_2m_min"])) / 2,
        "PRECTOTCORR": data["daily"]["precipitation_sum"]
    })
    df.set_index("Date", inplace=True)

    # Estimate soil from weather
    df["SOILM_TOT"] = df["PRECTOTCORR"].rolling(3, min_periods=1).mean() / 100
    df["TSOIL0_10M"] = df["T2M"] - 2

    return df

# ----------------------------
# Main App
# ----------------------------
if st.session_state.logged_in:
    st.title(f"🌾 AgriConnect with Open-Meteo - Welcome, {st.session_state.logged_in}")

    menu = st.radio("Choose Feature:", ["Social Feed", "Marketplace", "Price Prediction"])

    # --- Social Feed ---
    if menu == "Social Feed":
        st.subheader("📢 Farmer Social Feed")
        post_text = st.text_area("Write your post")
        if st.button("Post"):
            st.session_state.posts.append({
                "user": st.session_state.logged_in,
                "content": post_text,
                "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            })
            st.success("Post added!")
        for post in reversed(st.session_state.posts):
            st.info(f"🧑 {post['user']} ({post['date']}): {post['content']}")

    # --- Marketplace ---
    elif menu == "Marketplace":
        st.subheader("🌽 Marketplace")
        option = st.radio("Choose:", ["List Commodity", "View Listings"])
        if option == "List Commodity":
            commodity = st.selectbox("Commodity", ["Banana", "Onion", "Maize"])
            qty = st.number_input("Quantity (Kg)", min_value=1)
            price = st.number_input("Price (INR/Kg)", min_value=1)
            if st.button("Add Listing"):
                st.session_state.marketplace.append({
                    "user": st.session_state.logged_in,
                    "commodity": commodity,
                    "qty": qty,
                    "price": price
                })
                st.success("Listing added!")
        else:
            if len(st.session_state.marketplace) == 0:
                st.info("No listings yet.")
            else:
                st.dataframe(pd.DataFrame(st.session_state.marketplace))

    # --- Price Prediction ---
    elif menu == "Price Prediction":
        st.subheader("📈 Commodity Price Forecast with Weather + Soil")

        market = st.selectbox("Select Market", sorted(df["Market"].unique()))
        commodity = st.selectbox("Select Commodity", sorted(df["Commodity"].unique()))
        user_date = st.date_input("Enter future date")

        if st.button("Get Forecast"):
            filtered_df = df[(df["Market"] == market) & (df["Commodity"] == commodity)]
            monthly_df = filtered_df.groupby(pd.Grouper(key='Date', freq='M'))['Modal Price/Kg'].mean().reset_index()
            monthly_df.set_index('Date', inplace=True)

            last_date = monthly_df.index[-1]

            if pd.to_datetime(user_date) <= last_date:
                st.warning("⚠ Please enter a future date beyond dataset range.")
            else:
                months_ahead = (pd.to_datetime(user_date).year - last_date.year) * 12 + \
                               (pd.to_datetime(user_date).month - last_date.month)

                # Historical price series
                y = monthly_df["Modal Price/Kg"]

                # Dummy exogenous vars for training (trend-based)
                exog = pd.DataFrame({
                    "T2M": np.linspace(25, 30, len(y)),
                    "PRECTOTCORR": np.linspace(5, 20, len(y)),
                    "SOILM_TOT": np.linspace(0.1, 0.3, len(y)),
                    "TSOIL0_10M": np.linspace(23, 28, len(y))
                }, index=y.index)

                try:
                    model = SARIMAX(y, exog=exog, order=(1,1,1), seasonal_order=(1,1,1,12))
                    fit = model.fit(disp=False)

                    # Get Open-Meteo forecast
                    lat, lon = markets[market]
                    future_weather = get_future_weather(lat, lon, days_ahead=months_ahead*30)
                    future_exog = future_weather.resample("M").mean()[["T2M","PRECTOTCORR","SOILM_TOT","TSOIL0_10M"]]

                    # Pad if not enough months
                    if len(future_exog) < months_ahead:
                        last_vals = future_exog.iloc[-1]
                        while len(future_exog) < months_ahead:
                            future_exog.loc[future_exog.index[-1] + pd.offsets.MonthEnd(1)] = last_vals

                    forecast = fit.forecast(steps=months_ahead, exog=future_exog)
                    forecast_value = forecast.iloc[-1]

                    st.success(f"🌟 Forecasted Price for {commodity} in {market} on {user_date}: "
                               f"{forecast_value:.2f} INR/kg**")

                except Exception as e:
                    st.error(f"Model failed: {e}")