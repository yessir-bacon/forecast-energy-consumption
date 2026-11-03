import joblib
import pandas as pd
from pathlib import Path

from src.data_prep import data_prep_pipeline

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "best_hist_gradient_boosting_model.pkl"

expected_features =  [
    'day_of_week_num',
    'day',
    'total load actual_lag_1d',
    'total load actual_lag_2d',
    'week_of_year',
    'total load actual_lag_7d',
    'wind_deg',
    'temp_min',
    'temp_max',
    'temp_lag_1d',
    'month',
    'quarter',
    'wind_speed',
    'clouds_all',
    'pressure',
    'temp',
    'rain_3h',
    'is_month_start'
]

def load_production_model():
    """Loads the serialized champion model from the models directory."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Optimized model file not found at: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

def generate_forecast(raw_new_data: pd.DataFrame) -> pd.DataFrame:
    """
    Takes fresh raw input data, processes it, and returns a DataFrame 
    containing timestamps and predicted energy loads.
    """
    # Load model
    model = load_production_model()
    
    # Process raw incoming data (e.g., engineers lags, extracts datetime columns)
    # Ensure this function outputs a DataFrame with a datetime index or timestamp column
    processed_data = data_prep_pipeline(raw_new_data) 
    
    # Isolate and align only the columns the model was trained on
    # Using errors="ignore" or validating columns protects against crashes
    X_fresh = processed_data[expected_features]
    
    # Generate predictions
    predictions = model.predict(X_fresh)
    
    # Format the output cleanly for downstream use or visualization
    forecast_df = pd.DataFrame({
        "predicted_load": predictions
    }, index=processed_data.index) # Keeps the corresponding timestamps
    
    return forecast_df

if __name__ == "__main__":
    # Example usage / local testing block
    print("Testing inference pipeline layout...")
    
    # In a real run, you would load a fresh batch from data/raw/ or an API
    # raw_data_sample = pd.read_csv(ROOT_DIR / "data" / "raw" / "fresh_incoming_data.csv")
    
    # forecast = generate_forecast(raw_data_sample)
    # print(forecast.head())