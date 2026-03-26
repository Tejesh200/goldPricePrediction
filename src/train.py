import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib

def train_model(data_path="data/processed/gold_processed.csv", model_dir="models"):
    print("Loading processed data...")
    if not os.path.exists(data_path):
        print(f"Error: Processed data {data_path} not found.")
        return
        
    df = pd.read_csv(data_path, parse_dates=['Date'])
    
    # Define features and target
    # Excluding 'Date' because it's not a numeric feature
    # Excluding 'Target_Next_Close' as it's the target
    feature_cols = [c for c in df.columns if c not in ['Date', 'Target_Next_Close']]
    
    X = df[feature_cols]
    y = df['Target_Next_Close']
    
    # Chronological Split (No random shuffling for time-series)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
    
    # Train XGBoost Regressor
    model = xgb.XGBRegressor(
        n_estimators=200, 
        learning_rate=0.05, 
        max_depth=5, 
        random_state=42
    )
    
    print("Training XGBoost Model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print(f"Model Evaluation => MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")
    
    # Creating output directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Save Model
    model_path = os.path.join(model_dir, "gold_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save Feature Importance for Power BI
    fi_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    fi_path = os.path.join(model_dir, "feature_importance.csv")
    fi_df.to_csv(fi_path, index=False)
    print(f"Feature importance saved to {fi_path}")

if __name__ == "__main__":
    train_model()
