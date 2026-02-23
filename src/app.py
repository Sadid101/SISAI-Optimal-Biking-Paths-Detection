import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

from forecast_fetcher import FMIWeatherFetcher, FMIQueryLocation

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent

#MODIFY THESE TWO TO POINT TO YOUR TRAINING DATA JSON FILES
# ENSURE THE FILENAME ACTUALLY EXISTS
WEATHER_JSON_PATH = SCRIPT_DIR / "data" / "training" / "weather_forecast.json"

# ENSURE THE FILENAME ACTUALLY EXISTS
PEDESTRIAN_JSON_PATH = SCRIPT_DIR / "data" / "training" / "2020-02-20_to_2020-02-22_hour_Oulu_kaupunki_100000647.json"

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
    df_w["ts_hour"] = df_w["ts"].dt.floor("h")

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
    df_p["ts_hour"] = df_p["date"].dt.tz_convert(LOCAL_TZ).dt.floor("h")

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
    df_w = load_weather(WEATHER_JSON_PATH)
    df_p = load_pedestrians(PEDESTRIAN_JSON_PATH)

    # Join on hourly timestamp
    df = df_p.merge(df_w, on="ts_hour", how="inner")

    # Add features
    df = add_time_features(df, "ts_hour")

    # --- Baseline features: temperature + time/date features ---
    feature_cols = ["temp_c", "month", "weekday", "is_weekend", "hour_sin", "hour_cos"]
    target_col = "pedestrians"

    # Train/test split (time-based)
    train_df, test_df = time_based_split(df, "ts_hour", train_ratio=0.8)

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("Rows used (after merge):", len(df))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))

    # Show learned coefficients (interpretation)
    coef = pd.Series(model.coef_, index=feature_cols).sort_values(key=np.abs, ascending=False)
    print("\nCoefficients:")
    print(coef)
    print("\nIntercept:", model.intercept_)

    # --- Example: predict for a chosen time + temperature ---
    # Suppose user asks: "At 2026-02-21 15:00 local time and temp is -7.0, how many pedestrians?"
    query_ts = pd.Timestamp("2026-02-21 15:00:00", tz=LOCAL_TZ).floor("h")
    query_temp = -7.0

    query_df = pd.DataFrame([{"ts_hour": query_ts, "temp_c": query_temp}])
    query_df = add_time_features(query_df, "ts_hour")
    y_hat = model.predict(query_df[feature_cols])[0]

    print(f"\nPrediction for {query_ts} at temp {query_temp}°C: {y_hat:.2f} pedestrians")
def main():
    ensure_forecast_data()
    processData()

if __name__ == "__main__":
    main()