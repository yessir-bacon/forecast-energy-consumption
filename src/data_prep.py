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

    # Reset the index and rename the column
    merged_df = merged_df.reset_index().rename(columns={'index': 'timestamp'})
    
    return merged_df

def create_time_features(df):
    # Set 'timestamp' column as datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    # Specify the frequency as daily and sort the index
    df = df.set_index('timestamp').asfreq('D').sort_index()

    # Extract standard calendar components
    df['year'] = df.index.year
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['day'] = df.index.day

    # Label encode the year
    unique_years = sorted(df['year'].unique())
    year_mapping = {year: i for i, year in enumerate(unique_years)}
    df['year_encoded'] = df['year'].map(year_mapping)
    df = df.drop(columns=['year'])  # Drop the original year column after encoding

    # Day of the week features
    df['day_of_week_num'] = df.index.dayofweek  # 0 = Monday, 6 = Sunday
    df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)

    # Week-based features
    df['week_of_year'] = df.index.isocalendar().week.astype(int)

    # Week of the month (1 to 5)
    df['week_of_month'] = (df.index.day - 1) // 7 + 1

    # Business cycle markers
    df['is_month_start'] = df.index.is_month_start.astype(int)
    df['is_month_end'] = df.index.is_month_end.astype(int)
    df['is_quarter_end'] = df.index.is_quarter_end.astype(int)

    # Reset the index and rename the column
    df = df.reset_index()

    return df

def data_prep_pipeline():
    # Load and clean the data
    df = load_and_clean_data()

    # Create time-based features
    df_with_time = create_time_features(df)

    # Only keep useful features
    useful_features = [
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
    df_final = df_with_time[useful_features + ['total load actual']]

    # Save the processed dataset to a CSV file
    processed_file_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "daily_energy_and_weather_data.csv"
    df_final.to_csv(processed_file_path, index=False)
    print(f"Processed dataset saved to {processed_file_path}")

if __name__ == "__main__":
    data_prep_pipeline()