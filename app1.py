import streamlit as st
import pandas as pd
import numpy as np
import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX

# -------------------------------
# Custom CSS with background
# -------------------------------
st.markdown("""
<style>
body {
    background-image: url('https://images.unsplash.com/photo-1606041008023-472dfb5e530a'); 
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
}
.post-card, .market-card {
    background: rgba(255,255,255,0.9);
    padding: 15px;
    margin: 10px 0;
    border-radius: 12px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
    transition: 0.3s;
}
.post-card:hover, .market-card:hover {
    transform: translateY(-3px);
    box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
}
.profile-box {
    background: linear-gradient(135deg, #4CAF50, #8BC34A);
    padding: 20px;
    color: white;
    border-radius: 12px;
    text-align: center;
}
.badge {
    display: inline-block;
    background: gold;
    padding: 3px 8px;
    border-radius: 8px;
    font-size: 12px;
    margin-left: 5px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Dummy database
# -------------------------------
if "users" not in st.session_state:
    st.session_state.users = {}
if "xp" not in st.session_state:
    st.session_state.xp = {}
if "posts" not in st.session_state:
    st.session_state.posts = []
if "marketplace" not in st.session_state:
    st.session_state.marketplace = {}

# -------------------------------
# Generate synthetic dataset for prediction
# -------------------------------
@st.cache_data
def generate_data():
    dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq="D")
    markets = ["Coimbatore", "Chennai", "Tiruppur", "Salem", "Erode"]
    commodities = ["Banana", "Onion", "Maize"]

    data = []
    np.random.seed(0)
    for date in dates:
        for market in markets:
            for crop in commodities:
                price = np.random.randint(30, 50)
                data.append({"Date": date, "Market": market, "Commodity": crop, "Modal Price/Kg": price})

    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = generate_data()

# -------------------------------
# If not logged in → Login/Register
# -------------------------------
if "logged_in" not in st.session_state:
    st.sidebar.title("🌾 Farmer Login / Signup")
    action = st.sidebar.radio("Select Action:", ["Login", "Register"])
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if action == "Register" and st.sidebar.button("Sign Up"):
        if username in st.session_state.users:
            st.sidebar.warning("❌ Username already exists!")
        else:
            st.session_state.users[username] = password
            st.session_state.xp[username] = 0
            st.sidebar.success("✅ User registered successfully!")

    if action == "Login" and st.sidebar.button("Login"):
        if username in st.session_state.users and st.session_state.users[username] == password:
            st.session_state.logged_in = username
            st.experimental_rerun()
        else:
            st.sidebar.error("❌ Invalid credentials")

# -------------------------------
# If logged in → show app
# -------------------------------
else:
    user = st.session_state.logged_in
    st.sidebar.success(f"👨‍🌾 Logged in as {user}")

    # 🚪 Logout button
    if st.sidebar.button("Logout"):
        del st.session_state.logged_in
        st.experimental_rerun()

    # Sidebar profile
    st.sidebar.markdown(f"""
    <div class='profile-box'>
        <h3>{user}</h3>
        <p>XP: {st.session_state.xp[user]} 🌟</p>
        <p>Level: {st.session_state.xp[user] // 100 + 1}</p>
    </div>
    """, unsafe_allow_html=True)

    # Main app menu
    st.title(f"🌱 AgriConnect - Welcome, {user}!")
    menu = st.radio("📌 Choose Feature:", ["Social Feed", "Post Update", "Marketplace", "Price Prediction"])

    # -------------------------------
    # Social Feed
    # -------------------------------
    if menu == "Social Feed":
        st.subheader("📢 Farmer Social Feed")
        if len(st.session_state.posts) == 0:
            st.info("No posts yet. Be the first to post!")
        else:
            for post in reversed(st.session_state.posts):
                st.markdown(f"""
                <div class='post-card'>
                    <b>{post['user']}</b> <span class='badge'>👨‍🌾 Farmer</span><br>
                    <i>{post['date']}</i><br>
                    <p>{post['content']}</p>
                </div>
                """, unsafe_allow_html=True)

    # -------------------------------
    # Post Update
    # -------------------------------
    elif menu == "Post Update":
        st.subheader("✍ Create a Post")
        content = st.text_area("Write your post here...")
        if st.button("Post"):
            if content.strip():
                st.session_state.posts.append({
                    "user": user,
                    "content": content,
                    "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                })
                st.session_state.xp[user] += 20
                st.balloons()
                st.success("✅ Post created! (+20 XP)")

    # -------------------------------
    # Marketplace
    # -------------------------------
    elif menu == "Marketplace":
        st.subheader("🛒 Marketplace")
        action_market = st.radio("Action:", ["List Commodity", "View Listings"])
        if action_market == "List Commodity":
            commodity = st.selectbox("Select Commodity", ["Banana", "Onion", "Maize"])
            qty = st.number_input("Quantity (Kg)", min_value=1)
            price = st.number_input("Expected Price (INR/Kg)", min_value=1)
            if st.button("Add Listing"):
                if "marketplace" not in st.session_state:
                    st.session_state.marketplace = []
                st.session_state.marketplace.append({
                    "user": user,
                    "commodity": commodity,
                    "qty": qty,
                    "price": price
                })
                st.session_state.xp[user] += 10
                st.snow()
                st.success("✅ Listing added! (+10 XP)")
        else:
            if len(st.session_state.marketplace) == 0:
                st.info("No listings yet.")
            else:
                for m in st.session_state.marketplace:
                    st.markdown(f"""
                    <div class='market-card'>
                        <b>{m['commodity']}</b> - {m['qty']} Kg<br>
                        Price: {m['price']} INR/Kg<br>
                        Seller: {m['user']}
                    </div>
                    """, unsafe_allow_html=True)

    # -------------------------------
    # Price Prediction
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
                    st.success(f"📢 Forecasted Price for {commodity} in {market} on {user_date}: {forecast_value:.2f} INR/kg")
                    st.session_state.xp[user] += 30
                except Exception as e:
                    st.error(f"Model failed: {e}")