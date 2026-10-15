import pandas as pd
from pathlib import Path

# Locate the dataset
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "reliance_stock_data.csv"

# Check whether the dataset exists
if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_FILE}"
    )

# Load the dataset
df = pd.read_csv(DATA_FILE, header=[0, 1], index_col=0)

# Convert the index to dates
df.index = pd.to_datetime(df.index, errors="coerce")

# Remove any invalid date rows
df = df.loc[df.index.notna()]

# Convert stock values to numeric data
for column in df.columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Sort records by date
df = df.sort_index()

# Display dataset information
print("=" * 50)
print("STOCK PRICE DATASET EXPLORATION")
print("=" * 50)

print("\n1. Dataset shape:")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\n2. Column names:")
print(df.columns.tolist())

print("\n3. First five records:")
print(df.head().to_string())

print("\n4. Last five records:")
print(df.tail().to_string())

print("\n5. Data types:")
print(df.dtypes)

print("\n6. Missing values:")
print(df.isnull().sum())

print("\n7. Duplicate dates:")
print(df.index.duplicated().sum())

print("\n8. Date range:")
if not df.empty:
    print(f"Start date: {df.index.min().date()}")
    print(f"End date: {df.index.max().date()}")

print("\n9. Statistical summary:")
print(df.describe().to_string())

print("\nDataset exploration completed!")

