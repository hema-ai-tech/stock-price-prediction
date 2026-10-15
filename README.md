# 📈 Stock Price Prediction Using Machine Learning

## Live Demo

Try the application here:
https://stock-price-prediction-jidutdaoe6qw4soyjzyxud.streamlit.app/

## About the Project

This project uses machine learning to analyze historical stock market data and estimate the next trading day's closing price of Reliance Industries.

The project demonstrates the complete machine-learning workflow, including data collection, preprocessing, exploratory data analysis, feature engineering, model training, evaluation, and visualization.

An interactive dashboard built with Streamlit allows users to explore historical prices and view model predictions.

## 🎯 Objectives

* Collect and analyze historical stock market data.
* Clean and preprocess the dataset.
* Visualize historical closing prices and trading volume.
* Create features from previous closing prices and moving averages.
* Train and evaluate machine-learning regression models.
* Compare Linear Regression and Random Forest.
* Display predictions through an interactive dashboard.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* yfinance
* Streamlit
* Joblib
* Git and GitHub

## 🤖 Machine Learning Models

### 1. Linear Regression

Used to estimate the next trading day's closing price based on historical price features.

### 2. Random Forest Regressor

Used as a second model to compare prediction performance.

## 📊 Model Evaluation

The models were evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the R² score.

| Metric   | Linear Regression | Random Forest |
| -------- | ----------------: | ------------: |
| MAE      |             20.15 |         21.68 |
| RMSE     |             26.02 |         27.35 |
| R² Score |            0.9339 |        0.9270 |

On the current test dataset, Linear Regression performed slightly better than Random Forest across these metrics.

These results describe the current test period and do not guarantee future prediction accuracy.

## ✨ Features

* Historical stock price visualization.
* Dataset overview.
* Linear Regression and Random Forest model training.
* Model performance comparison.
* Actual-versus-predicted price visualization.
* Prediction results table.
* CSV download of prediction results.
* Estimated next trading day's closing price.
* Interactive Streamlit dashboard.

## 📁 Project Structure

```text
Stock-Price-Prediction/
├── data/
├── models/
├── notebooks/
├── main.py
├── explore_data.py
├── preprocess_data.py
├── visualize_data.py
├── feature_engineering.py
├── train_model.py
├── random_forest_model.py
├── compare_models.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

The `data/` and `models/` folders contain local datasets, generated results, and trained model files. These files are excluded from GitHub by the `.gitignore` rules.

## ⚙️ Installation and Setup

### Prerequisites

* Python installed on your computer.
* Git installed on your computer.

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your repository URL.

### 2. Enter the project directory

```bash
cd Stock-Price-Prediction
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

Windows Command Prompt:

```bash
.venv\Scripts\activate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## 🚀 Running the Project

Run the scripts in this order:

```bash
python main.py
python explore_data.py
python preprocess_data.py
python visualize_data.py
python feature_engineering.py
python train_model.py
python random_forest_model.py
python compare_models.py
```

Then launch the interactive dashboard:

```bash
python -m streamlit run app.py
```

Streamlit will provide a local address to open the dashboard in your browser.

## 📚 Dataset

Historical stock data is collected using the `yfinance` Python library for Reliance Industries (`RELIANCE.NS`).

The data available from the source may change over time. Internet access is required to download fresh data.

## ⚠️ Disclaimer

This project is developed for educational purposes and machine-learning experimentation. Stock price predictions are estimates and should not be considered financial advice or guaranteed future prices.

## 👩‍💻 Author

Developed as a minor project to explore machine learning, financial data analysis, and interactive application development.
