import os
import requests
import pandas as pd
import xml.etree.ElementTree as ET
import json
from datetime import datetime, timedelta
import time
from pathlib import Path
import sys

# --- CONFIGURATION ---
STATION_PLACE = "Oulu"
START_DATE = datetime(2014, 12, 31, 22, 0, 0)
END_DATE = datetime(2020, 12, 31, 22, 0, 0) 
DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
FMI_WFS_BASE = "https://opendata.fmi.fi/wfs"
OBS_STORED_QUERY = "fmi::observations::weather::timevaluepair"
PARAM_MAP = {"t2m": "temp_c"} 

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
                    
                    # FIX: Data is already UTC-aware, so just convert it
                    utc_dt = pd.to_datetime(t_elem.text).tz_convert('Europe/Helsinki')
                    local_ts = utc_dt.strftime(DATE_FORMAT)
                    points.append((local_ts, val))
            
            if points:
                df_p = pd.DataFrame(points, columns=["ts", PARAM_MAP[param_raw]])
                df_p.set_index("ts", inplace=True)
                series_list.append(df_p)
                
        return pd.concat(series_list, axis=1) if series_list else pd.DataFrame()
    except Exception as e:
        print(f"  [!] Error: {e}")
        return pd.DataFrame()

def check_files_exist():
    base_dir = Path(__file__).parent / "data"
    files = [
        base_dir / "training" / "weather_train.json",
        base_dir / "validation" / "weather_val.json",
        base_dir / "testing" / "weather_test.json"
    ]
    return all(f.exists() for f in files)

def run_builder():
    base_dir = Path(__file__).parent / "data"
    
    if check_files_exist():
        choice = input("Weather files already exist. Refetch? (y/[N]): ").strip().lower()
        if choice != 'y':
            print("Using existing files. Execution complete!")
            return

    print("--- Starting Weather Data Build ---")
    current_start = START_DATE
    all_chunks = []

    while current_start < END_DATE:
        current_end = min(current_start + timedelta(hours=168), END_DATE)
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