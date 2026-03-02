import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from utils.forecast_fetcher import FMIWeatherFetcher, FMIQueryLocation, fetchAndSaveForecast
from utils.pedestrianDataFetcher import check_files_exist, fetchEcoCounterSiteData, getAndPrintListOfSites, storeSplitJson
from utils.weather_dataset_builder import check_if_all_historical_weather_data_exists, run_builder

# INSTRUCTIONS
# JUST RUN THIS APP.PY AND FOLLOW THE PROMPTS IN THE CONSOLE.
#
#
# DEFAULT MACHINE LEARNING MODEL
# 1 Random Forest (more accurate but slower)
# 2 Linear Regression
DEFAULT_MODEL = 1

################################################################################################
# DO NOT TOUCH ANYTHING BELOW UNLESS YOU KNOW WHAT YOU ARE DOING
################################################################################################
#
# TEST RANGE FOR NOW IS 2026-02-25 00:00 to 2026-03-02 23:59 (full days with metrics), weather forecast includes only this date range.
#
# ENSURE THE FORMAT IS "YYYY-MM-DD HH:MM:SS" AND THE TIMEZONE IS LOCAL (Europe/Helsinki)

# FOR A FULL DAY START WITH MIDNIGHT AND END WITH MIDNIGHT OF THE NEXT DAY, OTHERWISE THE LAST HOUR MAY BE CUT OFF IN THE GRAPH (BECAUSE OF HOURLY ALIGNMENT)
DEFAULT_TEST_DATE_START = "2026-03-05 00:00:00"

DEFAULT_TEST_DATE_END = "2026-03-06 00:00:00"

# DO NOT TOUCH BELOW UNLESS YOU KNOW WHAT YOU ARE DOING
#
################################################################################################

DEFAULT_SITE_ID = "100025213" # Kempele/Asemantie
DEFAULT_TEST_SITE_NAME = "Kempele/Asemantie" 

# OLD HARDCODED TEST VALUES (FOR QUICK TESTING WITHOUT FORECAST)
DEFAULT_TEST_TEMPERATURE = -20

# Get the directory where this script is located
ROOT_PARENT_FOLDER = Path(__file__).parent

# PEDESTRIAN DATA

PEDESTRIAN_DATA_BASE_PATH = ROOT_PARENT_FOLDER / "data" / "pedestrians"

# ENSURE THE FILENAME ACTUALLY EXISTS
DEFAULT_TRAINING_PEDESTRIAN_JSON_PATH = PEDESTRIAN_DATA_BASE_PATH / "default" / "training" / "pedestrians_train.json"

# ENSURE THE FILENAME ACTUALLY EXISTS
DEFAULT_TESTING_PEDESTRIAN_JSON_PATH = PEDESTRIAN_DATA_BASE_PATH /  "default" /  "testing" / "pedestrians_test.json"

# ENSURE THE FILENAME ACTUALLY EXISTS
DEFAULT_VALIDATION_PEDESTRIAN_JSON_PATH = PEDESTRIAN_DATA_BASE_PATH / "default" / "validation" / "pedestrians_val.json"



# WEATHER DATA
WEATHER_DATA_BASE_PATH = ROOT_PARENT_FOLDER / "data" / "weather" 
# This is the historical data for TRAINING
TRAINING_HISTORICAL_WEATHER_PATH = WEATHER_DATA_BASE_PATH /  "default" / "training" / "weather_train.json"

# This is the historical data for TESTING
TESTING_HISTORICAL_WEATHER_PATH = WEATHER_DATA_BASE_PATH /  "default" / "testing" / "weather_test.json"

# This is the historical data for VALIDATION
VALIDATION_HISTORICAL_WEATHER_PATH = WEATHER_DATA_BASE_PATH /  "default" / "validation" / "weather_val.json"

# Weather forecast data
# ENSURE THE FILENAME ACTUALLY EXISTS
WEATHER_FORECAST_JSON_PATH = WEATHER_DATA_BASE_PATH /  "default" / "weather_forecast.json"
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
        WEATHER_FORECAST_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(WEATHER_FORECAST_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            
        print(f"[+] Forecast updated successfully: {WEATHER_FORECAST_JSON_PATH.name}")
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

def load_pedestrians_training_data(site_id=DEFAULT_SITE_ID) -> pd.DataFrame:
    # ENSURE THE FILENAME ACTUALLY EXISTS
    path =  PEDESTRIAN_DATA_BASE_PATH / "sites" / site_id / "training" / "pedestrians_train.json"
    return load_pedestrians(path)
def load_pedestrians_validation_data(site_id=DEFAULT_SITE_ID) -> pd.DataFrame:
    # ENSURE THE FILENAME ACTUALLY EXISTS
    path =  PEDESTRIAN_DATA_BASE_PATH / "sites" / site_id / "validation" / "pedestrians_val.json"
    return load_pedestrians(path)
def load_pedestrians_testing_data(site_id=DEFAULT_SITE_ID) -> pd.DataFrame:
    # ENSURE THE FILENAME ACTUALLY EXISTS
    path =  PEDESTRIAN_DATA_BASE_PATH / "sites" / site_id / "testing" / "pedestrians_test.json"
    return load_pedestrians(path)
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
def processDataWithLinearRegression(site_id=DEFAULT_SITE_ID):
    """Loads data, trains a simple linear regression baseline, evaluates, and shows coefficients."""
    train_df = load_pedestrians_training_data(site_id=site_id)
    train_weather_df = load_weather(TRAINING_HISTORICAL_WEATHER_PATH)
 

    test_df = load_pedestrians_testing_data(site_id)
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

# Model: Random Forest regression on temperature + time features
def processDataWithRandomForestRegression(site_id=DEFAULT_SITE_ID):
    """Loads data, trains a Random Forest regressor, evaluates, and shows feature importances."""
    train_df = load_pedestrians_training_data(site_id=site_id)
    train_weather_df = load_weather(TRAINING_HISTORICAL_WEATHER_PATH)
 

    test_df = load_pedestrians_testing_data(site_id)
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

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("Rows used (after merge):", len(train_merged) + len(test_merged))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))

    # Show feature importances (interpretation)
    coef = pd.Series(model.feature_importances_, index=featureColumns).sort_values(ascending=False)
    print("\nFeature importances:")
    print(coef)
    intercept = None

   
    return model, featureColumns, coef, intercept
# Baseline model: linear regression on temperature + time features
def predictPedestrianCountAtHourWithTemp(site_id=DEFAULT_SITE_ID, requestedTime=DEFAULT_TEST_DATE_START, requestedTemperature=DEFAULT_TEST_TEMPERATURE, model    =DEFAULT_MODEL):
    """Loads data, trains a simple linear regression baseline, evaluates, and shows coefficients."""
    if model == 1:
        model, featureColumns, coef, intercept = processDataWithRandomForestRegression(site_id=site_id)
    else:
       
        model, featureColumns, coef, intercept = processDataWithLinearRegression(site_id=site_id)

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

def predictPedestrianCountAtTimeRangeWithTemp(site=None, startDateTime=DEFAULT_TEST_DATE_START, endDateTime=DEFAULT_TEST_DATE_END, requestedTemperature=DEFAULT_TEST_TEMPERATURE, model=DEFAULT_MODEL):
    """Predicts pedestrian counts over a date range for given site and temperature using the trained model."""
   

    # First train the model on the historical data 
    if model == 1:
        model, featureColumns, coef, intercept = processDataWithRandomForestRegression(site_id=site["siteId"])
    else:
        model, featureColumns, coef, intercept = processDataWithLinearRegression(site_id=site["siteId"])
        

    # Then create a DataFrame for the requested date range and temperature, and predict pedestrian counts for each hour in that range
    date_range = pd.date_range(start=startDateTime, end=endDateTime, freq="h", tz=LOCAL_TZ)
    query_df = pd.DataFrame({"ts_hour": date_range, "temp_c": requestedTemperature})
    query_df = add_time_features(query_df, "ts_hour")
    query_df["predicted_pedestrians"] = model.predict(query_df[featureColumns])
    print(f"\nPredictions for {startDateTime} to {endDateTime} at temp {requestedTemperature}°C:")
    print(query_df[["ts_hour", "temp_c", "predicted_pedestrians"]])
    
    # Visualize the predictions
    visualize_predictions(query_df, 
                         title=f"Pedestrian ({startDateTime} to {endDateTime}) at {site['name']}")
    


def predictPedestrianCountAtTimeRange(site, startDateTime=DEFAULT_TEST_DATE_START, endDateTime=DEFAULT_TEST_DATE_END, model=DEFAULT_MODEL, saveJson=False):
    """Predicts pedestrian counts over a date range for given site using weather forecast data."""
    
    # Load the forecast
    forecast_df = load_weather(WEATHER_FORECAST_JSON_PATH)

    # First train the model on the historical data 
    if model == 1:
        model, featureColumns, coef, intercept = processDataWithRandomForestRegression(site_id=site["siteId"])
    else:
        model, featureColumns, coef, intercept = processDataWithLinearRegression(site_id=site["siteId"])

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
    
   
    return prediction_df
def getSiteDetailsByIndex(site_index, sites_list):
    """Returns the site details from the sites list based on the provided index. The index is expected to be 0-based."""
    if 0 <= site_index < len(sites_list):
        return sites_list[site_index]
    return None

def promptSiteSelection(sites):
    """Prompts the user to select a site by number from the list of sites."""
    selectedSite = None
     # Keep asking until a valid site number is entered
    while(selectedSite is None):
    # Ask for the user to input a site number from the list of sites printed above
        siteNumber = input("Enter the site number you want to fetch data for (Default: 1): ").strip()
        if(siteNumber == ""):
            print("No site number entered. Defaulting to site number 1 (Kempele/Asemantie)...")
            siteNumber = 1
        selectedSite = getSiteDetailsByIndex(int(siteNumber) - 1, sites)
        if not selectedSite or selectedSite["siteId"] is None:
            print(f"Number {siteNumber} not found. Please check the list and try again.\n")
            continue
    return selectedSite

def checkForPedestrianDataAndPromptRefetch(site):
    """Checks if pedestrian data files already exist for the given site. If they do, prompts the user to decide whether to refetch the data or use existing files.
    """
    if check_files_exist(site["siteId"]):
        choice = input(f"Pedestrian files already exist for the requested site {site['name']} ({site['siteId']}). Do you want to refetch? (y/[N]): ").strip().lower()
        if choice != 'y':
            print("Using existing files. Proceeding to prediction...")
        else:
            print(f"Refetching data for the site {site['siteId']}...")
            data = fetchEcoCounterSiteData(site["siteId"], site["domain"], 'hour')
            storeSplitJson(data, site_id=site["siteId"])
    else:
        print(f"Fetching data for the site {site['siteId']}...")
        data = fetchEcoCounterSiteData(site["siteId"], site["domain"], 'hour')
        storeSplitJson(data, site_id=site["siteId"])

DEFAULT_DATE_PROMPT = "Enter the date and time"
def promptForDate(promptMessage=DEFAULT_DATE_PROMPT,minDate=pd.Timestamp.now(tz=LOCAL_TZ).floor("h"), maxDate=None, defaultDate=pd.Timestamp.now(tz=LOCAL_TZ).floor("d") + pd.Timedelta(hours=24, minutes=0)):
    """Prompts the user to enter a date and time in the format 'YYYY-MM-DD HH:MM:SS' (Helsinki timezone, UTC+2). Validates the input and ensures it falls within the specified min and max date range if provided. If the user presses enter without inputting a date, it defaults to the provided defaultDate."""
    date = None
    while(date is None):
        dateInput = input(f"{promptMessage} \n(Default: {defaultDate}, format: 'YYYY-MM-DD HH:MM:SS', timezone: Helsinki UTC+2): ").strip()
        try:
            #default to provided defaultDate if user just presses enter without inputting a date
            if(dateInput == ""):
                if defaultDate is not None:
                    date = defaultDate
                    return date
                else:
                    print("No date entered and no default date provided. Please enter a valid date.")
                    continue
            dateSplit = dateInput.split(" ")
            # fill out the time part if the user only inputs a date without time
            if dateSplit and len(dateSplit) == 1:
                print("No time part detected in the input. Assuming 00:00:00 for the time...")
                dateInput += " 00:00:00"  # Append default time
            
            # Parse the naive datetime string and localize it to Helsinki timezone
            date = pd.to_datetime(dateInput)
            # Localize naive datetime to Helsinki timezone (Europe/Helsinki)
            if date.tz is None:
                date = date.tz_localize(LOCAL_TZ)
        except ValueError:
            date = None
            print("Invalid date format. Please ensure the format is 'YYYY-MM-DD HH:MM:SS'.")
            continue
        # Check if date is within the specified range        
        if minDate and date < minDate:
            date = None
            print(f"Date must be on or after {minDate}. Please try again.")
        if maxDate and date > maxDate:
            date = None
            print(f"Date must be on or before {maxDate}. Please try again.")
    return date
def check_historica_weather_data_and_fetch_if_needed():
    print("Checking for existing historical weather data...")
    if check_if_all_historical_weather_data_exists():
        print("Existing weather data found. No need to fetch.")
        return True
    else:
        print("No existing weather data found. Preparing to fetch data...")
        run_builder()
def save_predictions_to_json(predictions_df: pd.DataFrame, site_id: str, startDateTime: pd.Timestamp, endDateTime: pd.Timestamp, path: str = "./results"):
    """Saves prediction results to a JSON file at the specified path.
    
    Args:
        predictions_df: DataFrame with prediction results
        site_id: Site identifier
        startDateTime: Start date of predictions
        endDateTime: End date of predictions
        path: Directory path where the file will be saved (default: current directory)
    """
    # Format timestamps to be filename-safe (remove colons and timezone info)
    start_str = startDateTime.strftime("%Y-%m-%d_%H-%M-%S")
    end_str = endDateTime.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"predictions_{site_id}_{start_str}_{end_str}.json"
    output_file = Path(path) / filename
    output_file.parent.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
    predictions_df.to_json(output_file, orient="records", indent=2)
    print(f"Saved predictions to {output_file}")
def main():
    try:
        

        # 1. Prompt user to select a start date and end date for the prediction (ensure the format is "YYYY-MM-DD HH:MM:SS" and the timezone is local, e.g., "2026-02-25 00:00:00")
        currentDate = pd.Timestamp.now(tz=LOCAL_TZ).floor("h")
        print(f"Current date and time: {currentDate}")
        startDateTime = promptForDate(promptMessage="Enter a start date and time", minDate=currentDate)
        maxDateForEnd = startDateTime + pd.Timedelta(days=7)  # Limit end date to 7 days after start date
        endDateTime = promptForDate(promptMessage="Enter an end date and time", minDate=startDateTime, maxDate=maxDateForEnd, defaultDate=startDateTime + pd.Timedelta(hours=24, minutes=0)  )
        print(f"Selected date range: {startDateTime} to {endDateTime}\n")
       
        # 2.a Get and print the list of sites,
        sites = getAndPrintListOfSites()
        # 2.b ask user to select one by number, and fetch data for that site
        selectedSite = promptSiteSelection(sites)
        print(f"Fetching data for site ID: {selectedSite['siteId']}...")
    
        # 3. Check if files already exist for the selected site, if yes ask if user wants to refetch or use existing files
        checkForPedestrianDataAndPromptRefetch(selectedSite)

        # 4.a Ensure the historical weather data exists and fetch it if needed
        check_historica_weather_data_and_fetch_if_needed()
        # 4.b update weather forecast data before proceeding
        fetchAndSaveForecast()

        # 5. Run the prediction for the selected site and the default test date range using the default model
        results = predictPedestrianCountAtTimeRange(selectedSite, startDateTime, endDateTime, model=DEFAULT_MODEL)

        choice = input("Save the predicted results to a json? (y/[N]): ").strip().lower()
        if choice == 'y':
            save_predictions_to_json(results, selectedSite['siteId'], startDateTime, endDateTime)
        else:
            print("Skipping saving to JSON.")
        # Visualize the predictions
        visualize_predictions(results, 
                         title=f"Pedestrian Predictions ({startDateTime} to {endDateTime}) at {selectedSite['name']}")
            
    except Exception as e:
        print(f"An error occurred: {e}")
   
if __name__ == "__main__":
    main()