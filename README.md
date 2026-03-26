# 🪙 Gold Price Prediction using Machine Learning & Power BI

## 📌 Project Overview
The **GoldSentinel AI** project fetches historical gold price data, performs feature engineering using technical indicators, trains an **XGBoost Regression** model to predict the next day's closing price, and exports a curated dataset specifically designed for **Power BI** Dashboards. It also includes a stunning, interactive **Streamlit** Web Application to visualize the predictions.

## 🗂️ Project Structure
```text
goldPricePrediction/
├── app.py                 # Streamlit UI Dashboard
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── powerbi_guide.md       # Step-by-step guide for Power BI
├── src/
│   ├── data_fetch.py      # Uses `yfinance` to get raw GC=F data
│   ├── preprocess.py      # Cleans and calculates Technical Indicators (TA)
│   ├── train.py           # Trains the XGBoost Regressor model
│   └── predict.py         # Generates final datasets for Power BI export
├── data/                  # Generates raw/ and processed/ datasets
└── models/                # Saved models and feature_importance.csv
```

## 🚀 Quick Start Guide

### 1️⃣ Installation
Ensure you have Python 3.8+ installed. Open your terminal in this directory and install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2️⃣ Run the Pipeline
Generate the data and train the model by running the following commands in order:
```bash
# 1. Fetch raw data (10 years history)
python src/data_fetch.py

# 2. Add Technical Indicators & Clean Data
python src/preprocess.py

# 3. Train Machine Learning Model (XGBoost)
python src/train.py

# 4. Generate Final Predictions for Power BI
python src/predict.py
```

### 3️⃣ Launch the Interactive Dashboard
After running the pipeline, you can view the predicted prices seamlessly in a vibrant Web App:
```bash
streamlit run app.py
```

### 4️⃣ Power BI Dashboard
Open the instructions in [powerbi_guide.md](powerbi_guide.md) to import the final dataset (`data/processed/powerbi_export.csv`) into Power BI and create stunning visuals!
