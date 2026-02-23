import requests
import json
import time
from datetime import datetime
from pathlib import Path

START_DATE = "2014-12-31T22:00:00Z" # EARLIEST DATE TO CONTAIN SENSIBLE DATA = 2010-12-31T22:00:00.000Z
END_DATE = "2020-12-31T22:00:00Z" # LATEST DATE TO CONTAIN SENSIBLE DATA = 2021-05-31T23:59:59

# The format for parsing and saving timestamps (ISO 8601 UTC)
DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

PRECISION = "hour"
LOCATION = "Oulu_kaupunki" 
ID = "100000647" 

URL = "https://api.oulunliikenne.fi/proxy/graphql"

def fetchAllEcoCounterSites(id=ID, domain=LOCATION, step=PRECISION, begin=START_DATE, end=END_DATE):
    start_dt = datetime.strptime(begin, DATE_FORMAT)
    end_dt = datetime.strptime(end, DATE_FORMAT)
    
    all_counts = []
    current_start = start_dt

    print(f"Starting chunked fetch from {begin} to {end}...")

    while current_start < end_dt:
        # Move forward by 1 year
        try:
            current_end = min(current_start.replace(year=current_start.year + 1), end_dt)
        except ValueError: # Handle Feb 29 leap year issues
            current_end = min(current_start.replace(year=current_start.year + 1, day=28), end_dt)
        
        start_str = current_start.strftime(DATE_FORMAT)
        end_str = current_end.strftime(DATE_FORMAT)
        
        constructedQuery = """query GetEcoCounterSiteData {{
          ecoCounterSiteData(
            id: "{id}",
            domain: {domain},
            step: {step},
            begin: "{begin}",
            end: "{end}"
          ) {{
            date
            counts
          }}
        }}"""
        
        payload = {"query": constructedQuery.format(id=id, domain=domain, step=step, begin=start_str, end=end_str)}
        headers = {"Content-Type": "application/json"}

        print(f"  -> Fetching chunk: {start_str} to {end_str}")
        
        r = requests.post(URL, json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        
        data = r.json()
        if "errors" in data:
            raise RuntimeError(f"GraphQL errors: {data['errors']}")
            
        chunk_data = data["data"]["ecoCounterSiteData"]
        all_counts.extend(chunk_data)
        
        current_start = current_end
        time.sleep(0.5)

    print(f"Total rows retrieved: {len(all_counts)}")
    return {"ecoCounterSiteData": all_counts}

from datetime import timedelta

from datetime import timedelta

def storeJson(data):
    """
    Converts timestamps to Helsinki Time, adds the 'Z' suffix to match 
    weather data, and splits data into subfolders.
    """
    base_dir = Path(__file__).parent / "data"
    rows = data["ecoCounterSiteData"]
    
    for row in rows:
        clean_date = row["date"].split('.')[0].replace('Z', '')
        utc_dt = datetime.strptime(clean_date, "%Y-%m-%dT%H:%M:%S")
        local_dt = (utc_dt + timedelta(hours=2)).replace(microsecond=0)
        row["date"] = f"{local_dt.isoformat()}Z" 

    total_rows = len(rows)
    if total_rows == 0:
        print("No data found to save.")
        return

    # 60/20/20 split
    train_end = int(total_rows * 0.6)
    val_end = int(total_rows * 0.8)

    splits = {
        "training": (rows[:train_end], "pedestrians_train.json"),
        "validation": (rows[train_end:val_end], "pedestrians_val.json"),
        "testing": (rows[val_end:], "pedestrians_test.json")
    }

    for folder, (subset, filename) in splits.items():
        target_dir = base_dir / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = target_dir / filename
        payload = {"ecoCounterSiteData": subset}
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(subset)} rows to {target_dir.name}/{filename}")

if __name__ == "__main__":
    try:
        eco_counter_sites = fetchAllEcoCounterSites()
        storeJson(eco_counter_sites)
    except Exception as e:
        print(f"An error occurred: {e}")