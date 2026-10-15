
import pandas as pd
from pathlib import Path

# Define project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "reliance_stock_data_cleaned.csv"
OUTPUT_FILE = BASE_DIR / "data" / "stock_features.csv"

# Load the cleaned dataset
if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Cleaned dataset not found: {DATA_FILE}"
    )

df = pd.read_csv(DATA_FILE, header=[0, 1], index_col=0)

# Convert dates and sort chronologically
df.index = pd.to_datetime(df.index, errors="coerce")
df = df.loc[df.index.notna()].sort_index()

# Identify the closing price column
close_columns = [
    col for col in df.columns
    if (col[0] if isinstance(col, tuple) else col) == "Close"
]

if len(close_columns) != 1:
    raise ValueError(
        "Expected exactly one closing price column."
    )

# Create a simple DataFrame with a single closing price column
data = pd.DataFrame(index=df.index)
data["Close"] = pd.to_numeric(
    df[close_columns[0]], errors="coerce"
)

# Remove missing or invalid closing prices
data = data.dropna(subset=["Close"])
data = data[data["Close"] > 0]

# Create features using historical closing prices
data["Previous_Day_Close"] = data["Close"].shift(1)
data["Previous_2_Days_Close"] = data["Close"].shift(2)
data["Previous_3_Days_Close"] = data["Close"].shift(3)
data["Previous_5_Days_Close"] = data["Close"].shift(5)

# Moving averages use only information available before the prediction date
data["MA_5"] = data["Close"].shift(1).rolling(window=5).mean()
data["MA_10"] = data["Close"].shift(1).rolling(window=10).mean()

# Target: the next trading day's closing price
data["Target"] = data["Close"].shift(-1)

# Remove rows without enough historical data or a future target
data = data.dropna()

# Save the feature dataset
data.to_csv(OUTPUT_FILE, index_label="Date")

# Display results
print("=" * 55)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 55)

print("\nDataset shape:", data.shape)

print("\nFeature columns:")
print(data.columns.tolist())

print("\nFirst five records:")
print(data.head().to_string())

print("\nLast five records:")
print(data.tail().to_string())

print("\nSaved feature dataset to:")
print(OUTPUT_FILE)

print("\nFeature engineering completed successfully!")
