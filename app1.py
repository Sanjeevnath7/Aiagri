# agri_connect_prototype.py
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import datetime

# -------------------------------
# Dummy database for prototype
# -------------------------------
if 'users' not in st.session_state:
    st.session_state.users = {}  # username: password

if 'posts' not in st.session_state:
    st.session_state.posts = []  # social feed posts

if 'marketplace' not in st.session_state:
    st.session_state.marketplace = []  # marketplace listings

# -------------------------------
# Sidebar - Login / Register
# -------------------------------
st.sidebar.title("Farmer Login / Signup")
action = st.sidebar.radio("Select Action:", ["Login", "Register"])

username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

if action == "Register":
    if st.sidebar.button("Sign Up"):
        if username in st.session_state.users:
            st.sidebar.warning("Username already exists!")
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

# -------------------------------
# Generate synthetic dataset for prediction
# -------------------------------
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

# -------------------------------
# Main App
# -------------------------------
if 'logged_in' in st.session_state:
    st.title(f"AgriConnect - Welcome, {st.session_state.logged_in} 🌾")

    menu = st.radio("Choose Feature:", ["Social Feed", "Post Update", "Marketplace", "Price Prediction"])

    # -------------------------------
    # Social Feed
    # -------------------------------
    if menu == "Social Feed":
        st.subheader("Farmer Social Feed")
        if len(st.session_state.posts) == 0:
            st.info("No posts yet. Be the first to post!")
        else:
            for post in reversed(st.session_state.posts):
                st.write(f"{post['user']}** ({post['date']}): {post['content']}")

    # -------------------------------
    # Post Update
    # -------------------------------
    elif menu == "Post Update":
        st.subheader("Create a Post")
        content = st.text_area("Write your post here...")
        if st.button("Post"):
            st.session_state.posts.append({
                "user": st.session_state.logged_in,
                "content": content,
                "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            })
            st.success("Post created!")

    # -------------------------------
    # Marketplace
    # -------------------------------
    elif menu == "Marketplace":
        st.subheader("Marketplace")
        action_market = st.radio("Action:", ["List Commodity", "View Listings"])

        if action_market == "List Commodity":
            commodity = st.selectbox("Select Commodity", ["Banana", "Onion", "Maize"])
            qty = st.number_input("Quantity (Kg)", min_value=1)
            price = st.number_input("Expected Price (INR/Kg)", min_value=1)
            if st.button("Add Listing"):
                st.session_state.marketplace.append({
                    "user": st.session_state.logged_in,
                    "commodity": commodity,
                    "qty": qty,
                    "price": price
                })
                st.success("Listing added!")

        elif action_market == "View Listings":
            if len(st.session_state.marketplace) == 0:
                st.info("No listings yet.")
            else:
                df_market = pd.DataFrame(st.session_state.marketplace)
                st.dataframe(df_market)

    # -------------------------------
    # Price Prediction (SARIMAX)
    # -------------------------------
    elif menu == "Price Prediction":
        st.subheader("📈 Commodity Price Forecast (SARIMAX)")

        market = st.selectbox("Select Market", sorted(df["Market"].unique()))
        commodity = st.selectbox("Select Commodity", sorted(df["Commodity"].unique()))
        user_date = st.date_input("Enter future date (YYYY-MM-DD)")

        if st.button("Get Forecast"):
            filtered_df = df[(df["Market"] == market) & (df["Commodity"] == commodity)]
            monthly_df = filtered_df.groupby(pd.Grouper(key='Date', freq='M'))['Modal Price/Kg'].mean().reset_index()
            monthly_df.set_index('Date', inplace=True)

            last_date = monthly_df.index[-1]

            if pd.to_datetime(user_date) <= last_date:
                st.warning("⚠ Please enter a date after the dataset's last date.")
            else:
                months_ahead = (pd.to_datetime(user_date).year - last_date.year) * 12 + (pd.to_datetime(user_date).month - last_date.month)

                try:
                    sarima_model = SARIMAX(monthly_df['Modal Price/Kg'], order=(1,1,1), seasonal_order=(1,1,1,12))
                    sarima_fit = sarima_model.fit(disp=False)

                    forecast = sarima_fit.forecast(steps=months_ahead)
                    forecast_value = forecast.iloc[-1]

                    st.success(f"📢 Forecasted Price for {commodity} in {market} on {user_date}: *{forecast_value:.2f} INR/kg*")

                except Exception as e:
                    st.error(f"Model failed: {e}")