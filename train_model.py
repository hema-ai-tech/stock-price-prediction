
import pandas as pd
import joblib

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "stock_features.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = MODEL_DIR / "linear_regression_model.pkl"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load the feature dataset
# --------------------------------------------------

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Feature dataset not found: {DATA_FILE}"
    )

df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print("=" * 60)
print("LINEAR REGRESSION - STOCK PRICE PREDICTION")
print("=" * 60)

print("\nDataset shape:", df.shape)

# --------------------------------------------------
# 3. Define input features and target
# --------------------------------------------------

features = [
    "Previous_Day_Close",
    "Previous_2_Days_Close",
    "Previous_3_Days_Close",
    "Previous_5_Days_Close",
    "MA_5",
    "MA_10"
]

target = "Target"

X = df[features]
y = df[target]

# --------------------------------------------------
# 4. Split data chronologically
# --------------------------------------------------

split_index = int(len(df) * 0.80)

if split_index == 0 or split_index == len(df):
    raise ValueError("Not enough records to split the dataset.")

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

test_dates = df["Date"].iloc[split_index:]

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# --------------------------------------------------
# 5. Train the model
# --------------------------------------------------

model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel training completed successfully!")

# --------------------------------------------------
# 6. Generate predictions
# --------------------------------------------------

predictions = model.predict(X_test)

# --------------------------------------------------
# 7. Evaluate the model
# --------------------------------------------------

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\nMODEL EVALUATION")
print("-" * 40)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.4f}")

# --------------------------------------------------
# 8. Save test predictions
# --------------------------------------------------

results = pd.DataFrame({
    "Date": test_dates.to_numpy(),
    "Actual_Close": y_test.to_numpy(),
    "Predicted_Close": predictions
})

results_file = BASE_DIR / "data" / "linear_regression_predictions.csv"
results.to_csv(results_file, index=False)

print("\nSample predictions:")
print(results.head(10).round(2).to_string(index=False))

print("\nPredictions saved to:")
print(results_file)

# --------------------------------------------------
# 9. Save the trained model
# --------------------------------------------------

joblib.dump(model, MODEL_FILE)

# Save feature names for later application use
feature_info = {
    "features": features,
    "target": target
}

joblib.dump(feature_info, MODEL_DIR / "feature_info.pkl")

print("\nTrained model saved to:")
print(MODEL_FILE)

print("\nSTEP 7 COMPLETED SUCCESSFULLY!")
