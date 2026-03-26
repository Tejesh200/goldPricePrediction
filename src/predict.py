import os
import pandas as pd
import joblib

def generate_predictions(data_path="data/processed/gold_processed.csv", 
                        model_path="models/gold_model.pkl",
                        output_path="data/processed/powerbi_export.csv"):
    
    if not os.path.exists(data_path) or not os.path.exists(model_path):
        print("Error: Missing data or model files.")
        return
        
    print("Loading data and model...")
    df = pd.read_csv(data_path, parse_dates=['Date'])
    model = joblib.load(model_path)
    
    feature_cols = [c for c in df.columns if c not in ['Date', 'Target_Next_Close']]
    
    print("Generating predictions on history...")
    X = df[feature_cols]
    
    # Predict the Target (which is Next Day Close)
    predictions = model.predict(X)
    
    # Create final clean DataFrame for Power BI
    powerbi_df = df[['Date', 'Close', 'Target_Next_Close']].copy()
    powerbi_df.rename(columns={'Close': 'Actual_Close', 'Target_Next_Close': 'Actual_Next_Close'}, inplace=True)
    powerbi_df['Predicted_Next_Close'] = predictions
    
    # Calculate Error
    powerbi_df['Prediction_Error_Absolute'] = abs(powerbi_df['Actual_Next_Close'] - powerbi_df['Predicted_Next_Close'])
    powerbi_df['Prediction_Error_Percentage'] = (powerbi_df['Prediction_Error_Absolute'] / powerbi_df['Actual_Next_Close']) * 100
    
    # Add a column determining if model predicted direction correctly
    powerbi_df['Actual_Direction'] = (powerbi_df['Actual_Next_Close'] > powerbi_df['Actual_Close']).astype(int)
    powerbi_df['Predicted_Direction'] = (powerbi_df['Predicted_Next_Close'] > powerbi_df['Actual_Close']).astype(int)
    powerbi_df['Direction_Correct'] = (powerbi_df['Actual_Direction'] == powerbi_df['Predicted_Direction']).astype(int)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    powerbi_df.to_csv(output_path, index=False)
    print(f"Power BI dataset successfully saved to {output_path} ({len(powerbi_df)} rows)")

if __name__ == "__main__":
    generate_predictions()
