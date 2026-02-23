import os
import requests
import pandas as pd
import xml.etree.ElementTree as ET
import json
from datetime import datetime, timedelta
import time
from pathlib import Path

# --- CONFIGURATION ---
STATION_PLACE = "Oulu"
START_DATE = datetime(2015, 1, 1)
END_DATE = datetime(2021, 1, 1) 
FMI_WFS_BASE = "https://opendata.fmi.fi/wfs"
OBS_STORED_QUERY = "fmi::observations::weather::timevaluepair"
PARAM_MAP = {"t2m": "temp_c"} # Maps FMI parameter to our JSON key

def fetch_chunk(starttime: datetime, endtime: datetime) -> pd.DataFrame:
    """Fetches a single chunk of weather data from the FMI WFS API."""
    params = {
        "service": "WFS", "version": "2.0.0", "request": "getFeature",
        "storedquery_id": OBS_STORED_QUERY, "place": STATION_PLACE,
        "starttime": starttime.strftime('%Y-%m-%dT%H:%M:%SZ'),
        "endtime": endtime.strftime('%Y-%m-%dT%H:%M:%SZ'),
        "timestep": 60 # Hourly data
    }
    try:
        response = requests.get(FMI_WFS_BASE, params=params, timeout=60)
        response.raise_for_status()
        root = ET.fromstring(response.text)
        ns = {"wml2": "http://www.opengis.net/waterml/2.0"}
        series_list = []
        
        # Parse XML response to extract time-value pairs
        for ts in root.findall(".//wml2:MeasurementTimeseries", ns):
            series_id = ts.attrib.get("{http://www.opengis.net/gml/3.2}id", "")
            param_raw = series_id.split("-")[-1] if "-" in series_id else series_id
            
            if param_raw not in PARAM_MAP: 
                continue
                
            points = []
            for p in ts.findall(".//wml2:point", ns):
                t_elem = p.find(".//wml2:time", ns)
                v_elem = p.find(".//wml2:value", ns)
                if t_elem is not None and v_elem is not None:
                    val_str = v_elem.text.strip() if v_elem.text else ""
                    try:
                        # Convert string value to float, handling NaNs
                        val = float(val_str) if val_str not in ("", "NaN") else None
                    except ValueError: 
                        val = None
                    points.append((t_elem.text, val))
            
            if points:
                df_p = pd.DataFrame(points, columns=["ts", PARAM_MAP[param_raw]])
                df_p.set_index("ts", inplace=True)
                series_list.append(df_p)
                
        return pd.concat(series_list, axis=1) if series_list else pd.DataFrame()
    except Exception as e:
        print(f"  [!] Error fetching chunk: {e}")
        return pd.DataFrame()

def run_builder():
    """Main orchestration function to build the historical weather dataset."""
    
    # Path setup: src/data/training/historical_weather_data
    base_dir = Path(__file__).parent
    output_dir = base_dir / "data" / "training" / "historical_weather_data"
    
    # --- CHECK IF DATA ALREADY EXISTS ---
    # Prevents re-downloading if the work is already done
    required_files = ["train_final.json", "val_final.json", "test_final.json"]
    if output_dir.exists():
        files_exist = all((output_dir / f).exists() for f in required_files)
        if files_exist:
            print(f"--- Weather data already exists in {output_dir}. Skipping build. ---")
            return True

    # Ensure local directory structure exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Starting Weather Data Build for {STATION_PLACE} ---")
    print(f"Target Directory: {output_dir}")
    print(f"Target Period: {START_DATE.date()} to {END_DATE.date()}")
    
    current_start = START_DATE
    all_chunks = []

    # Loop through time range in 1-week increments (API limit safety)
    while current_start < END_DATE:
        current_end = min(current_start + timedelta(hours=168), END_DATE)
        
        # '\r' allows the print to update on the same line
        print(f" > Fetching: {current_start.strftime('%Y-%m-%d')} ...", end="\r")
        
        df_chunk = fetch_chunk(current_start, current_end)
        if not df_chunk.empty:
            all_chunks.append(df_chunk)
        
        time.sleep(0.4) # Small delay to respect FMI rate limits
        current_start = current_end

    print("\nDownload complete. Processing data...")

    if not all_chunks:
        print("Error: No data was collected.")
        return False

    # Merge all weekly DataFrames and clean duplicates
    full_df = pd.concat(all_chunks).sort_index()
    full_df = full_df[~full_df.index.duplicated(keep='first')]
    
    # Define split points (60% Train, 20% Val, 20% Test)
    total = len(full_df)
    train_end = int(total * 0.6)
    val_end = int(total * 0.8)
    
    datasets = {
        "train_final.json": full_df.iloc[:train_end],
        "val_final.json": full_df.iloc[train_end:val_end],
        "test_final.json": full_df.iloc[val_end:]
    }

    # Save each subset as a structured JSON file
    for name, df in datasets.items():
        rows = [{"ts": ts, "temp_c": row["temp_c"]} for ts, row in df.iterrows()]
        file_path = output_dir / name
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({"location": {"place": STATION_PLACE}, "rows": rows}, f, indent=2)
        print(f"Successfully saved: {name} ({len(rows)} records)")
            
    print("--- Weather Data Build Finished ---\n")
    return True

# --- EXECUTION ---
if __name__ == "__main__":
    run_builder()