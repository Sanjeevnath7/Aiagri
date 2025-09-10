import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Commodity Price Dashboard")

# Example: Load data (replace with your actual file path or DataFrame)
# df = pd.read_csv("your_data.csv")
# For demo, create a sample DataFrame
df = pd.DataFrame({
    "Date": pd.date_range("2023-01-01", periods=100, freq="D"),
    "Commodity": ["Banana"]*50 + ["Onion"]*50,
    "Price/Kg": pd.np.random.randint(20, 80, 100)
})

commodity = st.selectbox("Select Commodity", df["Commodity"].unique())
filtered_df = df[df["Commodity"] == commodity]

st.write(f"Showing data for: {commodity}")
st.dataframe(filtered_df)

st.line_chart(filtered_df.set_index("Date")["Price/Kg"])

# Matplotlib plot
fig, ax = plt.subplots()
ax.plot(filtered_df["Date"], filtered_df["Price/Kg"], marker="o")
ax.set_title(f"{commodity} Price Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Price/Kg")
st.pyplot(fig)