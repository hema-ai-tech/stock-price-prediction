import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Define the project directory
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "reliance_stock_data_cleaned.csv"

# Check whether the dataset exists
if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Cleaned dataset not found: {DATA_FILE}"
    )

# Load the cleaned dataset
df = pd.read_csv(DATA_FILE, header=[0, 1], index_col=0)

# Convert the index to datetime
df.index = pd.to_datetime(df.index, errors="coerce")
df = df.loc[df.index.notna()]
df = df.sort_index()

# Extract the required columns
close_columns = [
    column for column in df.columns
    if (column[0] if isinstance(column, tuple) else column) == "Close"
]

volume_columns = [
    column for column in df.columns
    if (column[0] if isinstance(column, tuple) else column) == "Volume"
]

if len(close_columns) != 1 or len(volume_columns) != 1:
    raise ValueError(
        "Could not identify exactly one Close and Volume column."
    )

closing_price = pd.to_numeric(df[close_columns[0]], errors="coerce")
trading_volume = pd.to_numeric(df[volume_columns[0]], errors="coerce")

# Remove rows with missing values in the selected columns
plot_data = pd.DataFrame({
    "Close": closing_price,
    "Volume": trading_volume
}).dropna()

# Graph 1: Historical closing price
plt.figure(figsize=(12, 6))
plt.plot(plot_data.index, plot_data["Close"], label="Closing Price")
plt.title("Reliance Industries - Historical Closing Price")
plt.xlabel("Date")
plt.ylabel("Price (INR)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the graph
price_graph = BASE_DIR / "data" / "historical_closing_price.png"
plt.savefig(price_graph, dpi=150)
plt.show()

print(f"Closing price graph saved to: {price_graph}")

# Graph 2: Trading volume
plt.figure(figsize=(12, 6))
plt.plot(plot_data.index, plot_data["Volume"], label="Trading Volume")
plt.title("Reliance Industries - Historical Trading Volume")
plt.xlabel("Date")
plt.ylabel("Number of Shares")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the graph
volume_graph = BASE_DIR / "data" / "historical_trading_volume.png"
plt.savefig(volume_graph, dpi=150)
plt.show()

print(f"Trading volume graph saved to: {volume_graph}")

print("\nData visualization completed successfully!")
