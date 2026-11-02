from pathlib import Path
import numpy as np
import pandas as pd

def load_and_clean_data():
    # Resolve paths from the project root, regardless of the launch directory.
    project_root = Path(__file__).resolve().parent.parent
    raw_data_dir = project_root / "data" / "raw"

    energy_df = pd.read_csv(raw_data_dir / "energy_dataset.csv")
    weather_df = pd.read_csv(raw_data_dir / "weather_features.csv")

    # Convert time columns to datetime format
    time_col = 'timestamp'
    energy_df[time_col] = pd.to_datetime(energy_df['time'], format='ISO8601', utc=True)
    weather_df[time_col] = pd.to_datetime(weather_df['dt_iso'], format='ISO8601', utc=True)

    # Set time column as index
    energy_df = energy_df.set_index(time_col)
    weather_df = weather_df.set_index(time_col)

    # Drop original time column
    energy_df = energy_df.drop(columns=['time'])
    weather_df = weather_df.drop(columns=['dt_iso'])

    # Resample energy data to daily frequency and aggregate by sum
    daily_energy = energy_df.resample('D').agg({i: 'sum' for i in energy_df})

    # Resample weather data to daily frequency and aggregate by mean because it has multiple cities per day
    daily_weather = weather_df.resample('D').agg({i: 'mean' for i in weather_df.select_dtypes(include='number')})

    # Most frequent weather conditions
    daily_weather_conditions = (
        weather_df[['weather_main']]
        .resample('D')
        .apply(lambda x: x.mode().iloc[0] if not x.empty else np.nan)
    )

    # Merge datasets on the time index
    merged_df = pd.concat([
        daily_energy,
        daily_weather,
        daily_weather_conditions
    ], axis=1, join='inner')

    # One-hot encode the 'weather_main' column
    weather_main_encoded = pd.get_dummies(merged_df['weather_main'], prefix='weather_main', dtype=int)
    merged_df = pd.concat([merged_df, weather_main_encoded], axis=1).drop(columns=['weather_main'])

    # Data leakage columns to remove
    remove_cols = [i for i in merged_df if 'generation' in i or 'forecast' in i]
    remove_cols += ['price actual', 'price day ahead', 'total load forecast']
    merged_df = merged_df.drop(columns=remove_cols)

    # Lag the target variable to capture weekly seasonality
    target_col = 'total load actual'
    merged_df[f'{target_col}_lag_1d'] = merged_df[target_col].shift(1)
    merged_df[f'{target_col}_lag_2d'] = merged_df[target_col].shift(2)
    merged_df[f'{target_col}_lag_7d'] = merged_df[target_col].shift(7)

    # Lag the core weather features: temperature and humidity
    weather_metrics_to_lag = ['temp', 'humidity']
    for col in weather_metrics_to_lag:
        merged_df[f'{col}_lag_1d'] = merged_df[col].shift(1)
        merged_df[f'{col}_lag_2d'] = merged_df[col].shift(2)

    # Drop the first 7 rows with NaN values due to lagging
    merged_df = merged_df.dropna()

    # Save the cleaned data to a new CSV file
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "daily_energy_and_weather_data.csv"
    merged_df.to_csv(output_path)
    print(f"✨ Success! Base daily dataset exported to: {output_path}\n")

if __name__ == "__main__":
    load_and_clean_data()
