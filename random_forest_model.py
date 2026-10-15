
import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# File paths
DATA_FILE = "data/stock_features.csv"
MODEL_DIR = "models"
PREDICTIONS_FILE = "data/random_forest_predictions.csv"

os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Input features
features = [
    "Previous_Day_Close",
    "Previous_2_Days_Close",
    "Previous_3_Days_Close",
    "Previous_5_Days_Close",
    "MA_5",
    "MA_10",
]

target = "Target"

X = df[features]
y = df[target]

# Split chronologically: first 80% for training, last 20% for testing
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Train Random Forest
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Generate predictions
predictions = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("=" * 55)
print("RANDOM FOREST - STOCK PRICE PREDICTION")
print("=" * 55)

print(f"\nTraining records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")

print("\nMODEL EVALUATION")
print("-" * 40)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.4f}")

# Save predictions
results = pd.DataFrame({
    "Date": df.loc[X_test.index, "Date"].values,
    "Actual_Close": y_test.values,
    "Predicted_Close": predictions
})

results.to_csv(PREDICTIONS_FILE, index=False)

# Save trained model and feature names
joblib.dump(model, os.path.join(MODEL_DIR, "random_forest_model.pkl"))
joblib.dump(features, os.path.join(MODEL_DIR, "random_forest_features.pkl"))

print(f"\nPredictions saved to: {PREDICTIONS_FILE}")
print(f"Model saved to: {MODEL_DIR}/random_forest_model.pkl")
print("\nSTEP 8 COMPLETED SUCCESSFULLY!")
