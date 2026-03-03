import requests
import pandas as pd
import xml.etree.ElementTree as ET
import json
from datetime import datetime, timedelta
import time
from pathlib import Path

# --- CONFIGURATION ---
STATION_PLACE = "Oulu"
START_DATE = datetime(2014, 12, 31, 22, 0, 0)
END_DATE = datetime(2026, 1, 1, 0, 0, 0)
DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
FMI_WFS_BASE = "https://opendata.fmi.fi/wfs"
OBS_STORED_QUERY = "fmi::observations::weather::timevaluepair"
PARAM_MAP = {"t2m": "temp_c"} 
# Get the root parent directory
ROOT_PARENT_FOLDER = Path(__file__).parent.parent
WEATHER_DATA_BASE_PATH = ROOT_PARENT_FOLDER / "data" / "weather"

def fetch_chunk(starttime: datetime, endtime: datetime) -> pd.DataFrame:
    params = {
        "service": "WFS", "version": "2.0.0", "request": "getFeature",
        "storedquery_id": OBS_STORED_QUERY, "place": STATION_PLACE,
        "starttime": starttime.strftime(DATE_FORMAT),
        "endtime": endtime.strftime(DATE_FORMAT),
        "timestep": 60 
    }
    try:
        response = requests.get(FMI_WFS_BASE, params=params, timeout=60)
        response.raise_for_status()
        root = ET.fromstring(response.text)
        ns = {"wml2": "http://www.opengis.net/waterml/2.0"}
        series_list = []
        
        for ts_node in root.findall(".//wml2:MeasurementTimeseries", ns):
            series_id = ts_node.attrib.get("{http://www.opengis.net/gml/3.2}id", "")
            param_raw = series_id.split("-")[-1] if "-" in series_id else series_id
            if param_raw not in PARAM_MAP: continue
                
            points = []
            for p in ts_node.findall(".//wml2:point", ns):
                t_elem = p.find(".//wml2:time", ns)
                v_elem = p.find(".//wml2:value", ns)
                if t_elem is not None and v_elem is not None:
                    val_str = v_elem.text.strip() if v_elem.text else ""
                    try:
                        val = float(val_str) if val_str not in ("", "NaN") else None
                    except ValueError: val = None
                    
                    utc_dt = pd.to_datetime(t_elem.text)
                    points.append((utc_dt.strftime(DATE_FORMAT), val))
            
            if points:
                df_p = pd.DataFrame(points, columns=["ts", PARAM_MAP[param_raw]])
                df_p.set_index("ts", inplace=True)
                series_list.append(df_p)
                
        return pd.concat(series_list, axis=1) if series_list else pd.DataFrame()
    except Exception as e:
        print(f"   [!] API Error: {e}")
        return pd.DataFrame()

def repair_nan_values():
    """Finds NaN values in existing JSONs and tries to refetch them."""
    base_dir = WEATHER_DATA_BASE_PATH / "default"
    folders = ["training", "validation", "testing"]
    
    print("--- Starting Weather Data Repair ---")
    
    for folder in folders:
        file_path = base_dir / folder / f"weather_{folder[:5] if folder != 'testing' else 'test'}.json"
        if not file_path.exists(): continue
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        df = pd.DataFrame(data["rows"])
        df.set_index("ts", inplace=True)
        
        # Identify rows where temp_c is NaN or None
        nan_mask = df["temp_c"].isna()
        if not nan_mask.any():
            print(f"✅ {folder}: No NaN values found.")
            continue
            
        nan_timestamps = df.index[nan_mask].tolist()
        print(f"⚠️ {folder}: Found {len(nan_timestamps)} NaN values. Attempting refetch...")

        for ts_str in nan_timestamps:
            # Convert string back to datetime for the API call
            ts_dt = datetime.strptime(ts_str, DATE_FORMAT)
            # Fetch just this specific hour (1 hour window)
            new_val_df = fetch_chunk(ts_dt, ts_dt + timedelta(minutes=1))
            
            if not new_val_df.empty and not pd.isna(new_val_df["temp_c"].iloc[0]):
                new_temp = new_val_df["temp_c"].iloc[0]
                df.at[ts_str, "temp_c"] = new_temp
                print(f"   Fixed {ts_str}: {new_temp}°C")
                time.sleep(0.2) # Avoid hitting API too hard
            else:
                print(f"   Failed to recover {ts_str} (Station might have been offline)")

        # Save back to file
        rows = [{"ts": ts, "temp_c": row["temp_c"]} for ts, row in df.iterrows()]
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({"location": {"place": STATION_PLACE}, "rows": rows}, f, indent=2)
        print(f"💾 {folder}: Saved updated file.")

def check_if_all_historical_weather_data_exists():
    base_dir = WEATHER_DATA_BASE_PATH / "default"
    trainingDataExists = (base_dir / "training" / "weather_train.json").exists()
    validationDataExists = (base_dir / "validation" / "weather_val.json").exists()
    testingDataExists = (base_dir / "testing" / "weather_test.json").exists()
    return trainingDataExists and validationDataExists and testingDataExists


        
def run_builder(start_date: datetime = START_DATE, end_date: datetime = END_DATE):
    base_dir = WEATHER_DATA_BASE_PATH / "default"
    
    if (check_if_all_historical_weather_data_exists()):
        print("Existing data found.")
        print("1. Full Refetch (Delete and start over)")
        print("2. Repair NaNs (Keep existing, try to fill holes)")
        print("3. Exit")
        choice = input("Select an option (1-3): ").strip()
        
        if choice == '2':
            repair_nan_values()
            return
        elif choice != '1':
            return

    print("--- Starting Weather Data Build ---")
    current_start = start_date
    all_chunks = []

    while current_start < end_date:
        current_end = min(current_start + timedelta(hours=168), end_date)
        print(f" > Fetching: {current_start.strftime('%Y-%m-%d')} ...", end="\r")
        df_chunk = fetch_chunk(current_start, current_end)
        if not df_chunk.empty:
            all_chunks.append(df_chunk)
        time.sleep(0.4)
        current_start = current_end

    print("\nProcessing and splitting data...")
    full_df = pd.concat(all_chunks).sort_index()
    full_df = full_df[~full_df.index.duplicated(keep='first')]
    full_df = full_df.loc["2015-01-01T00:00:00Z":]

    total = len(full_df)
    train_end = int(total * 0.6)
    val_end = int(total * 0.8)
    
    splits = {
        "training": (full_df.iloc[:train_end], "weather_train.json"),
        "validation": (full_df.iloc[train_end:val_end], "weather_val.json"),
        "testing": (full_df.iloc[val_end:], "weather_test.json")
    }

    for folder, (df, filename) in splits.items():
        target_dir = base_dir / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        rows = [{"ts": ts, "temp_c": row["temp_c"]} for ts, row in df.iterrows()]
        with open(target_dir / filename, "w", encoding="utf-8") as f:
            json.dump({"location": {"place": STATION_PLACE}, "rows": rows}, f, indent=2)
        print(f"Saved {len(rows)} rows to {folder}/{filename}")

if __name__ == "__main__":
    run_builder()