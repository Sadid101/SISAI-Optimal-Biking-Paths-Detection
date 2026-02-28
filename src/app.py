import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

from forecast_fetcher import FMIWeatherFetcher, FMIQueryLocation

# INSTRUCTIONS
#
#
# 1. SET THESE TWO TO THE DATE TIME RANGE YOU WANT TO PREDICT FOR 
# 2. AND JUST RUN THE APP.PY TO SEE THE PREDICTIONS AND GRAPH
#
# TEST RANGE FOR NOW IS 2026-02-25 00:00 to 2026-03-02 23:59 (full days with metrics), weather forecast includes only this date range.
#
# ENSURE THE FORMAT IS "YYYY-MM-DD HH:MM:SS" AND THE TIMEZONE IS LOCAL (Europe/Helsinki)

# FOR A FULL DAY START WITH MIDNIGHT AND END WITH MIDNIGHT OF THE NEXT DAY, OTHERWISE THE LAST HOUR MAY BE CUT OFF IN THE GRAPH (BECAUSE OF HOURLY ALIGNMENT)
TEST_DATE_START = "2026-03-02 00:00:00"

TEST_DATE_END = "2026-03-03 00:00:00"

#TEST_DATE_START = "2026-03-02 00:00:00"

#TEST_DATE_END = "2026-03-02 23:59:00"

# DO NOT TOUCH BELOW UNLESS YOU KNOW WHAT YOU ARE DOING
#
################################################################################################
# OLD HARDCODED TEST VALUES (FOR QUICK TESTING WITHOUT FORECAST)
TEST_TEMPERATURE = -20

TEST_SITE_NAME = "Ouluhalli"  # site id 100000647 at "Oulu_kaupunki" Domain

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent

# MODIFY THESE PATHS BELOW TWO TO POINT TO YOUR DATA JSON FILES

# ENSURE THE FILENAME ACTUALLY EXISTS
TRAINING_PEDESTRIAN_JSON_PATH = SCRIPT_DIR / "data" / "training" / "pedestrians_train.json"

# This is the historical data for TRAINING
TRAINING_HISTORICAL_WEATHER_PATH = SCRIPT_DIR / "data" / "training" / "weather_train.json"

# ENSURE THE FILENAME ACTUALLY EXISTS
TESTING_PEDESTRIAN_JSON_PATH = SCRIPT_DIR / "data" / "testing" / "pedestrians_test.json"

# This is the historical data for TESTING
TESTING_HISTORICAL_WEATHER_PATH = SCRIPT_DIR / "data" / "testing" / "weather_test.json"

# ENSURE THE FILENAME ACTUALLY EXISTS
VALIDATION_PEDESTRIAN_JSON_PATH = SCRIPT_DIR / "data" / "validation" / "pedestrians_val.json"

# This is the historical data for VALIDATION
VALIDATION_HISTORICAL_WEATHER_PATH = SCRIPT_DIR / "data" / "validation" / "weather_val.json"

# ENSURE THE FILENAME ACTUALLY EXISTS
WEATHER_JSON_PATH = SCRIPT_DIR / "data" / "weather_forecast.json"

# DO NOT TOUCH BELOW UNLESS YOU KNOW WHAT YOU ARE DOING

LOCAL_TZ = "Europe/Helsinki"
def ensure_forecast_data():
    """
    Triggers the forecast fetcher to update the weather_forecast.json file 
    before the model runs.
    """
    print("Updating Weather Forecast...")
    try:
        fmi = FMIWeatherFetcher()
        loc = FMIQueryLocation(place="Oulu")
        
        # Fetch data
        df_forecast = fmi.fetch_24h_forecast(loc)
        payload = fmi.to_json_payload(df_forecast, loc)
        
        # Ensure directory exists and save
        WEATHER_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(WEATHER_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            
        print(f"[+] Forecast updated successfully: {WEATHER_JSON_PATH.name}")
    except Exception as e:
        print(f"[!] Could not update forecast: {e}")
        print("Attempting to proceed with existing file if available...")

# Example of fetching bike rental stations, not used in baseline but can be useful for future work
def load_weather(weather_path: Path) -> pd.DataFrame:
    """
    Expects structure like:
    {
      "timezone": "Europe/Helsinki",
      "rows": [{"ts":"2026-02-20T14:00:00+02:00","temp_c":-15.5,...}, ...]
    }
    """
    data = json.loads(weather_path.read_text(encoding="utf-8"))
    rows = data["rows"]

    df_w = pd.DataFrame(rows)

    # Parse timestamp with timezone info from ISO string, then convert to LOCAL_TZ
    df_w["ts"] = pd.to_datetime(df_w["ts"], utc=False)
    if df_w["ts"].dt.tz is None:
        # If somehow the API gives naive timestamps, localize to the JSON timezone or LOCAL_TZ
        tz = data.get("timezone", LOCAL_TZ)
        df_w["ts"] = df_w["ts"].dt.tz_localize(tz)

    df_w["ts"] = df_w["ts"].dt.tz_convert(LOCAL_TZ)

    # Align to hour (important for joining)
    df_w["ts_hour"] = df_w["ts"].dt.floor("h", ambiguous="infer")

    # Keep what we need (rough model uses only temperature)
    df_w = df_w[["ts_hour", "temp_c"]].dropna()
    df_w = df_w.drop_duplicates(subset=["ts_hour"]).sort_values("ts_hour")

    return df_w

# Example of fetching bike rental stations, not used in baseline but can be useful for future work
def load_pedestrians(ped_path: Path) -> pd.DataFrame:
    """
    Expects structure like:
    {"ecoCounterSiteData":[{"date":"2021-01-31T22:00:00.000Z","counts":32740}, ...]}
    """
    data = json.loads(ped_path.read_text(encoding="utf-8"))
    rows = data["ecoCounterSiteData"]
    df_p = pd.DataFrame(rows)

    # Parse UTC Z timestamps and convert to LOCAL_TZ
    df_p["date"] = pd.to_datetime(df_p["date"], utc=True)
    df_p["ts_hour"] = df_p["date"].dt.tz_convert(LOCAL_TZ).dt.floor("h", ambiguous="infer")

    # Rename target to something clearer
    df_p = df_p.rename(columns={"counts": "pedestrians"})

    # Clean / de-dup
    df_p = df_p[["ts_hour", "pedestrians"]].dropna()
    df_p = df_p.drop_duplicates(subset=["ts_hour"]).sort_values("ts_hour")

    return df_p

# Simple time-based features for baseline model
def add_time_features(df: pd.DataFrame, ts_col: str = "ts_hour") -> pd.DataFrame:
    """
    Adds simple time/date features. Keep it minimal for baseline.
    """
    out = df.copy()
    out["hour"] = out[ts_col].dt.hour
    out["weekday"] = out[ts_col].dt.weekday  # Mon=0 ... Sun=6
    out["month"] = out[ts_col].dt.month
    out["is_weekend"] = (out["weekday"] >= 5).astype(int)

    # Optional: cyclical hour encoding (often better than raw hour)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)

    return out


def time_based_split(df: pd.DataFrame, ts_col: str = "ts_hour", train_ratio: float = 0.8):
    """
    Deterministic split: first 80% timestamps for train, last 20% for test.
    """
    df = df.sort_values(ts_col).reset_index(drop=True)
    cutoff = int(len(df) * train_ratio)
    return df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()


# Baseline model: linear regression on temperature + time features
def processData():
    """Loads data, trains a simple linear regression baseline, evaluates, and shows coefficients."""
    train_df = load_pedestrians(TRAINING_PEDESTRIAN_JSON_PATH)
    train_weather_df = load_weather(TRAINING_HISTORICAL_WEATHER_PATH)
 

    test_df = load_pedestrians(TESTING_PEDESTRIAN_JSON_PATH)
    test_weather_df = load_weather(TESTING_HISTORICAL_WEATHER_PATH)

       # Join on hourly timestamp
    train_merged = train_df.merge(train_weather_df, on="ts_hour", how="inner")
    test_merged = test_df.merge(test_weather_df, on="ts_hour", how="inner")

    # Add features
    train_merged = add_time_features(train_merged, "ts_hour")
    test_merged = add_time_features(test_merged, "ts_hour")

    # --- Baseline features: temperature + time/date features ---
    featureColumns = ["temp_c", "month", "weekday", "is_weekend", "hour_sin", "hour_cos"]
    target_col = "pedestrians"

    X_train = train_merged[featureColumns]
    y_train = train_merged[target_col]

    X_test = test_merged[featureColumns]
    y_test = test_merged[target_col]

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("Rows used (after merge):", len(train_merged) + len(test_merged))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))

    # Show learned coefficients (interpretation)
    coef = pd.Series(model.coef_, index=featureColumns).sort_values(key=np.abs, ascending=False)
    print("\nCoefficients:")
    print(coef)
    print("\nIntercept:", model.intercept_)

   
    return model, featureColumns, coef, model.intercept_
# Baseline model: linear regression on temperature + time features
def predictPedestrianCountAtHourWithTemp(requestedTime=TEST_DATE_START, requestedTemperature=TEST_TEMPERATURE):
    """Loads data, trains a simple linear regression baseline, evaluates, and shows coefficients."""
    model, featureColumns, coef, intercept = processData()

     # --- Example: predict for a chosen time + temperature ---
    # Suppose user asks: "At 2026-02-21 15:00 local time and temp is -7.0, how many pedestrians?"
    query_ts = pd.Timestamp(requestedTime, tz=LOCAL_TZ).floor("h")
    query_temp = requestedTemperature

    query_df = pd.DataFrame([{"ts_hour": query_ts, "temp_c": query_temp}])
    query_df = add_time_features(query_df, "ts_hour")
    y_hat = model.predict(query_df[featureColumns])[0]

    timestamp = query_ts
    temperature = query_temp
    pedestrianPrediction = y_hat

    print(f"\nPrediction for {timestamp} at temp {temperature}°C: {pedestrianPrediction:.2f} pedestrians")



    return pedestrianPrediction, timestamp, temperature


def visualize_predictions(query_df: pd.DataFrame, title: str = "Pedestrian Count Predictions", save_path: str = None):
    """
    Visualizes prediction results with a line plot.
    
    Args:
        query_df: DataFrame with columns ['ts_hour', 'predicted_pedestrians', 'temp_c']
        title: Title for the graph
        save_path: Optional path to save the figure (e.g., 'predictions.png')
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot pedestrian predictions on primary y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Predicted Pedestrians', color=color, fontsize=12)
    ax1.plot(query_df['ts_hour'], query_df['predicted_pedestrians'], 
             color=color, marker='o', linewidth=2, markersize=4, label='Pedestrian Count')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # Plot temperature on secondary y-axis
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Temperature (°C)', color=color, fontsize=12)
    ax2.plot(query_df['ts_hour'], query_df['temp_c'], 
             color=color, marker='s', linewidth=2, markersize=4, linestyle='--', label='Temperature')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Formatting
    plt.title(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    # Show an hourly grid/tick so each hour is distinguishable (force local TZ)
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=LOCAL_TZ))

    # Clamp x-axis to actual data range to avoid empty hours at the ends
    ax1.set_xlim(query_df['ts_hour'].min(), query_df['ts_hour'].max())

    # Rotate x-axis labels for readability
    fig.autofmt_xdate(rotation=45, ha='right')
    
    # Add legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graph saved to: {save_path}")
    
    plt.show()

def predictPedestrianCountAtTimeRangeWithTemp(startDateTime=TEST_DATE_START, endDateTime=TEST_DATE_END, requestedTemperature=TEST_TEMPERATURE):
    """Placeholder for future prediction code over a date range."""
   

    # First train the model on the historical data 
    model, featureColumns, coef, intercept = processData()

    # Then create a DataFrame for the requested date range and temperature, and predict pedestrian counts for each hour in that range
    date_range = pd.date_range(start=startDateTime, end=endDateTime, freq="h", tz=LOCAL_TZ)
    query_df = pd.DataFrame({"ts_hour": date_range, "temp_c": requestedTemperature})
    query_df = add_time_features(query_df, "ts_hour")
    query_df["predicted_pedestrians"] = model.predict(query_df[featureColumns])
    print(f"\nPredictions for {startDateTime} to {endDateTime} at temp {requestedTemperature}°C:")
    print(query_df[["ts_hour", "temp_c", "predicted_pedestrians"]])
    
    # Visualize the predictions
    visualize_predictions(query_df, 
                         title=f"Pedestrian ({startDateTime} to {endDateTime}) at {TEST_SITE_NAME}")
    


def predictPedestrianCountAtTimeRange(startDateTime=TEST_DATE_START, endDateTime=TEST_DATE_END):
    """Predicts pedestrian counts over a date range using weather forecast data."""
    
    # Load the forecast
    forecast_df = load_weather(WEATHER_JSON_PATH)

    # First train the model on the historical data 
    model, featureColumns, coef, intercept = processData()

    # Create a DataFrame for the requested date range
    date_range = pd.date_range(start=startDateTime, end=endDateTime, freq="h", tz=LOCAL_TZ)
    
    # For each hour in the range, find the matching forecast temperature and predict
    predictions = []
    
    for query_ts in date_range:
        query_ts = pd.Timestamp(query_ts).floor("h")
        
        # Find the matching temperature in the forecast
        matching_row = forecast_df[forecast_df['ts_hour'] == query_ts]
        
        if not matching_row.empty:
            query_temp = matching_row.iloc[0]['temp_c']
        else:
            print(f"Warning: {query_ts} not in forecast range. Assuming the temperature stays the same...")
            query_temp = forecast_df['temp_c'].iloc[-1]  # Use last known temp as fallback
            
        
        # Perform prediction using the forecast temperature
        query_df = pd.DataFrame([{"ts_hour": query_ts, "temp_c": query_temp}])
        query_df = add_time_features(query_df, "ts_hour")
        y_hat = model.predict(query_df[featureColumns])[0]
        
        predictions.append({
            "ts_hour": query_ts,
            "temp_c": query_temp,
            "predicted_pedestrians": y_hat
        })
    
    if not predictions:
        print(f"Warning: No weather forecast data available for the requested time range.")
        print(f"Forecast data covers: {forecast_df['ts_hour'].min()} to {forecast_df['ts_hour'].max()}")
        return
    
    prediction_df = pd.DataFrame(predictions)
    
    print(f"\nPredictions for {startDateTime} to {endDateTime} using weather forecast:")
    print(prediction_df[["ts_hour", "temp_c", "predicted_pedestrians"]])
    
    # Visualize the predictions
    visualize_predictions(prediction_df, 
                         title=f"Pedestrian Predictions ({startDateTime} to {endDateTime}) at {TEST_SITE_NAME}")


def main():
    predictPedestrianCountAtTimeRange()
if __name__ == "__main__":
    main()