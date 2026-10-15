import yfinance as yf
from pathlib import Path

# Define the project directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Create the data folder if it doesn't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Stock symbol for Reliance Industries on NSE
STOCK_SYMBOL = "RELIANCE.NS"

# Download approximately five years of daily historical data
print("Downloading stock market data...")

data = yf.download(
    STOCK_SYMBOL,
    period="5y",
    interval="1d",
    auto_adjust=True,
    progress=False
)

# Check whether data was downloaded
if data.empty:
    print("ERROR: No data was downloaded.")
    print("Check your internet connection and try again.")
else:
    # Save the dataset as a CSV file
    file_path = DATA_DIR / "reliance_stock_data.csv"
    data.to_csv(file_path)

    # Display basic information
    print("\nDataset downloaded successfully!")
    print(f"Stock symbol: {STOCK_SYMBOL}")
    print(f"Number of records: {len(data)}")
    print(f"Dataset location: {file_path}")

    print("\nFirst five records:")
    print(data.head())

    print("\nDataset columns:")
    print(data.columns.tolist())

    print("\nMissing values in each column:")
    print(data.isnull().sum())

