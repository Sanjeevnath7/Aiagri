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
    st.session_state.posts = []  # list of dicts: {"user":..., "content":..., "date":...}

if 'marketplace' not in st.session_state:
    st.session_state.marketplace = []  # list of dicts: {"user":..., "commodity":..., "qty":..., "price":...}

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
    # Price Prediction
    # -------------------------------
    elif menu == "Price Prediction":
        st.subheader("Predict Future Price for a Commodity")

        # Dummy historical price data (monthly average)
        dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq="M")
        np.random.seed(0)
        banana_prices = np.random.randint(30, 50, len(dates))
        df_prices = pd.DataFrame({"Date": dates, "Banana": banana_prices})
        df_prices.set_index("Date", inplace=True)

        commodity = st.selectbox("Select Commodity to Predict", ["Banana", "Onion", "Maize"])
        months_to_predict = st.slider("Months to predict", 1, 12, 3)

        if st.button("Predict"):
            # Fit SARIMAX model
            model = SARIMAX(df_prices[commodity], order=(1,1,1), seasonal_order=(1,1,1,12))
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(steps=months_to_predict)

            st.subheader(f"{commodity} Price Forecast (Next {months_to_predict} months)")
            for i, value in enumerate(forecast):
                next_month = df_prices.index[-1] + pd.DateOffset(months=i+1)
                st.write(f"{next_month.strftime('%Y-%m')}: {value:.2f} INR/kg")