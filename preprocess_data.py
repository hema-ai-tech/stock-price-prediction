import pandas as pd
from pathlib import Path

# Define project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

INPUT_FILE = DATA_DIR / "reliance_stock_data.csv"
OUTPUT_FILE = DATA_DIR / "reliance_stock_data_cleaned.csv"

# Load the downloaded dataset
if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE, header=[0, 1], index_col=0)

# Convert index to datetime
df.index = pd.to_datetime(df.index, errors="coerce")

# Remove rows with invalid dates
df = df.loc[df.index.notna()]

# Convert column values to numeric
for column in df.columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Sort by date
df = df.sort_index()

print("Original dataset shape:", df.shape)

# Remove duplicate dates, keeping the last occurrence
df = df.loc[~df.index.duplicated(keep="last")]

# Check required stock price columns
required_columns = ["Close", "High", "Low", "Open", "Volume"]

available_columns = [
    column[0] if isinstance(column, tuple) else column
    for column in df.columns
]

missing_columns = [
    column for column in required_columns
    if column not in available_columns
]

if missing_columns:
    raise ValueError(
        f"Required columns are missing: {missing_columns}"
    )

# Remove rows where the closing price is missing
close_columns = [
    column for column in df.columns
    if (column[0] if isinstance(column, tuple) else column) == "Close"
]

if len(close_columns) != 1:
    raise ValueError(
        "Expected exactly one Close price column."
    )

close_column = close_columns[0]
df = df.dropna(subset=[close_column])

# Remove rows with invalid stock prices or volume
price_columns = [
    column for column in df.columns
    if (column[0] if isinstance(column, tuple) else column)
    in ["Open", "High", "Low", "Close"]
]

for column in price_columns:
    df = df[df[column] > 0]

volume_columns = [
    column for column in df.columns
    if (column[0] if isinstance(column, tuple) else column) == "Volume"
]

if volume_columns:
    df = df[df[volume_columns[0]] >= 0]

# Fill remaining missing values in non-target columns
# using the previous available trading day's values.
df = df.ffill()

# Remove rows that still contain missing values
df = df.dropna()

# Save the cleaned dataset
df.to_csv(OUTPUT_FILE)

# Display results
print("\nData cleaning completed successfully!")
print("Cleaned dataset shape:", df.shape)
print("Remaining missing values:", int(df.isna().sum().sum()))
print("Duplicate dates:", int(df.index.duplicated().sum()))
print("Saved file:", OUTPUT_FILE)

print("\nFirst five cleaned records:")
print(df.head().to_string())
