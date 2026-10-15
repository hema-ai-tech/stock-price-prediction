
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib

# Page configuration
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

# Project paths
DATA_FILE = "data/reliance_stock_data_cleaned.csv"
FEATURES_FILE = "data/stock_features.csv"
PREDICTIONS_FILE = "data/linear_regression_predictions.csv"
MODEL_FILE = "models/linear_regression_model.pkl"
FEATURE_INFO_FILE = "models/feature_info.pkl"

# Page title
st.title("📈 Stock Price Prediction Dashboard")
st.write(
    "Analyze historical Reliance Industries stock prices "
    "and explore machine-learning predictions."
)

st.info(
    "This project is for educational purposes. "
    "Predictions are estimates, not financial advice."
)

# Check required files
required_files = [
    DATA_FILE,
    FEATURES_FILE,
    PREDICTIONS_FILE,
    MODEL_FILE
]

missing_files = [file for file in required_files if not os.path.exists(file)]

if missing_files:
    st.error("Some project files are missing:")
    for file in missing_files:
        st.write(f"- {file}")
    st.stop()

# Load data
@st.cache_data
def load_data():
    historical = pd.read_csv(
        DATA_FILE,
        header=[0, 1],
        index_col=0,
        parse_dates=True
    )

    features = pd.read_csv(
        FEATURES_FILE,
        parse_dates=["Date"]
    )

    predictions = pd.read_csv(
        PREDICTIONS_FILE,
        parse_dates=["Date"]
    )

    return historical, features, predictions

historical, features, predictions = load_data()

# Historical closing price
st.header("1. Historical Closing Price")

close_column = next(
    (column for column in historical.columns
     if column[0] == "Close"),
    None
)

if close_column is None:
    st.error("The Close price column was not found in the historical dataset.")
    st.stop()

closing_prices = pd.to_numeric(
    historical[close_column],
    errors="coerce"
).dropna()

if closing_prices.empty:
    st.error("No valid closing-price data is available.")
    st.stop()

fig1, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(closing_prices.index, closing_prices.values)
ax1.set_title("Reliance Industries Historical Closing Price")
ax1.set_xlabel("Date")
ax1.set_ylabel("Closing Price (₹)")
ax1.grid(True)

st.pyplot(fig1)
plt.close(fig1)

# Dataset overview
st.header("2. Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Historical Records", len(closing_prices))
col2.metric(
    "First Closing Price",
    f"₹{closing_prices.iloc[0]:,.2f}"
)
col3.metric(
    "Latest Closing Price",
    f"₹{closing_prices.iloc[-1]:,.2f}"
)

# Model performance
st.header("3. Linear Regression Predictions")

col1, col2, col3 = st.columns(3)

col1.metric("MAE", "₹20.15")
col2.metric("RMSE", "₹26.02")
col3.metric("R² Score", "0.9339")

st.caption(
    "These metrics are from the model's held-out test dataset. "
    "Performance may differ on future data."
)

# Actual vs predicted
st.subheader("Actual vs Predicted Closing Prices")

fig2, ax2 = plt.subplots(figsize=(12, 5))

ax2.plot(
    predictions["Date"],
    predictions["Actual_Close"],
    label="Actual Closing Price"
)

ax2.plot(
    predictions["Date"],
    predictions["Predicted_Close"],
    label="Predicted Closing Price"
)

ax2.set_xlabel("Date")
ax2.set_ylabel("Closing Price (₹)")
ax2.set_title("Linear Regression: Actual vs Predicted")
ax2.legend()
ax2.grid(True)

st.pyplot(fig2)
plt.close(fig2)

# Prediction records
st.subheader("Prediction Results")
st.dataframe(predictions, use_container_width=True)

# Download results
csv_data = predictions.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Prediction Results (CSV)",
    data=csv_data,
    file_name="stock_price_predictions.csv",
    mime="text/csv"
)

st.success("Dashboard loaded successfully!")


# Next Trading Day Prediction
st.header("4. Predict the Next Trading Day's Closing Price")

st.write(
    "Use the most recent available historical prices "
    "to estimate the next trading day's closing price."
)

try:
    model = joblib.load(MODEL_FILE)
    feature_info = joblib.load(FEATURE_INFO_FILE)

    # Handle either a list of feature names or a dictionary
    if isinstance(feature_info, dict):
        feature_names = feature_info.get("features")

        if feature_names is None:
            feature_names = [
                "Previous_Day_Close",
                "Previous_2_Days_Close",
                "Previous_3_Days_Close",
                "Previous_5_Days_Close",
                "MA_5",
                "MA_10",
            ]
    else:
        feature_names = feature_info

    # Select the most recent available feature row
    latest_features = features[feature_names].iloc[[-1]]

    # Predict the next trading day's closing price
    next_day_prediction = model.predict(latest_features)[0]

    st.metric(
        label="Estimated Next Trading Day Closing Price",
        value=f"₹{next_day_prediction:,.2f}"
    )

    st.caption(
        "This is an estimate based on historical data, "
        "not a guaranteed future stock price."
    )

except FileNotFoundError:
    st.error(
        "A required model file is missing. "
        "Please check that train_model.py has run successfully."
    )

except Exception as error:
    st.error(f"Prediction error: {error}")
