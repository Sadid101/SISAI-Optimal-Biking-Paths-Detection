import requests
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys

START_DATE = "2014-12-31T22:00:00Z" # EARLIEST DATE TO CONTAIN SENSIBLE DATA = 2010-12-31T22:00:00.000Z
END_DATE = "2020-12-31T22:00:00Z" # LATEST DATE TO CONTAIN SENSIBLE DATA = 2021-05-31T23:59:59

# The format for parsing and saving timestamps (ISO 8601 UTC)
DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

#PRECISION
STEP = "hour"

#LOCATION
DOMAIN = "Oulu_kaupunki" 

# SITE ID
ID = "100000647" 

URL = "https://api.oulunliikenne.fi/proxy/graphql"

def fetchAllEcoCounterSites():
    """Fetches metadata for all eco counter sites from the API."""
    # Query to fetch all eco counter sites (no filtering - API returns all sites)
    constructedQuery = """query GetAllEcoCounterSites {
  ecoCounterSites {
    id
    siteId
    name
    domain
    userType
    timezone
    interval
    sens
    channels {
      id
      siteId
      name
      domain
      userType
      timezone
      interval
      sens
      lat
      lon
    }
  }
}"""

    payload = {"query": constructedQuery}
    headers = {
        "Content-Type": "application/json",
        # Add auth headers here if the API needs them, e.g.:
        # "Authorization": "Bearer YOUR_TOKEN"
    }

    print("Fetching eco counter sites...")
    r = requests.post(URL, json=payload, headers=headers, timeout=10)
    
    if not r.ok:
        print("\n" + "="*80)
        print("ERROR: Non-2xx response received")
        print("="*80)
        print(f"Status Code: {r.status_code}")
        print(f"Reason: {r.reason}")
        print(f"\nRequest URL: {r.url}")
        print(f"\nRequest Payload:\n{json.dumps(payload, indent=2)}")
        print(f"\nResponse Headers:\n{dict(r.headers)}")
        print(f"\nResponse Body:\n{r.text}")
        print("="*80 + "\n")
        r.raise_for_status()
    
    data = r.json()

    # GraphQL returns {"data": {...}, "errors": [...]}
    if "errors" in data:
        print("\n" + "="*80)
        print("GraphQL ERRORS DETECTED")
        print("="*80)
        print(f"Number of errors: {len(data['errors'])}\n")
        print("Full Response Data:")
        print(json.dumps(data, indent=2, default=str))
        print("\n" + "-"*80)
        for i, error in enumerate(data['errors'], 1):
            print(f"\nError {i}:")
            print(json.dumps(error, indent=2, default=str))
        print("\n" + "="*80 + "\n")
        raise RuntimeError(f"GraphQL errors: {data['errors']}")

    return data["data"]

def fetchEcoCounterSiteData(id=ID, domain=DOMAIN, step=STEP, begin=START_DATE, end=END_DATE):
    """Fetches eco counter site data for given site in chunks to avoid timeouts and memory issues."""
    start_dt = datetime.strptime(begin, DATE_FORMAT)
    end_dt = datetime.strptime(end, DATE_FORMAT)
    all_counts = []
    current_start = start_dt

    print(f"Starting chunked fetch from {begin} to {end}...")
    while current_start < end_dt:
        try:
            current_end = min(current_start.replace(year=current_start.year + 1), end_dt)
        except ValueError:
            current_end = min(current_start.replace(year=current_start.year + 1, day=28), end_dt)
        
        payload = {
            "query": """query GetEcoCounterSiteData {{
                ecoCounterSiteData(id: "{id}", domain: {domain}, step: {step}, begin: "{begin}", end: "{end}") 
                {{ date counts }}
            }}""".format(id=id, domain=domain, step=step, begin=current_start.strftime(DATE_FORMAT), end=current_end.strftime(DATE_FORMAT))
        }
        
        r = requests.post(URL, json=payload, headers={"Content-Type": "application/json"}, timeout=60)
        r.raise_for_status()

        if not r.ok:
            print("\n" + "="*80)
            print("ERROR: Non-2xx response received")
            print("="*80)
            print(f"Status Code: {r.status_code}")
            print(f"Reason: {r.reason}")
            print(f"\nRequest URL: {r.url}")
            print(f"\nRequest Payload:\n{json.dumps(payload, indent=2)}")
            print(f"\nResponse Headers:\n{dict(r.headers)}")
            print(f"\nResponse Body:\n{r.text}")
            print("="*80 + "\n")
            r.raise_for_status()
    
        data = r.json()

        # GraphQL returns {"data": {...}, "errors": [...]}
        if "errors" in data:
            print("\n" + "="*80)
            print("GraphQL ERRORS DETECTED")
            print("="*80)
            print(f"Number of errors: {len(data['errors'])}\n")
            print("Full Response Data:")
            print(json.dumps(data, indent=2, default=str))
            print("\n" + "-"*80)
            for i, error in enumerate(data['errors'], 1):
                print(f"\nError {i}:")
                print(json.dumps(error, indent=2, default=str))
            print("\n" + "="*80 + "\n")
            raise RuntimeError(f"GraphQL errors: {data['errors']}")
    
       
       
            
        all_counts.extend(data["data"]["ecoCounterSiteData"])
        current_start = current_end
        time.sleep(0.5)

    # Convert all retrieved timestamps to Finnish Local Time (+2h)
    # This makes the first entry 2015-01-01T00:00:00Z
    for row in all_counts:
        clean_date = row["date"].split('.')[0].replace('Z', '')
        utc_dt = datetime.strptime(clean_date, "%Y-%m-%dT%H:%M:%S")
        local_dt = (utc_dt + timedelta(hours=2)).replace(microsecond=0)
        row["date"] = local_dt.strftime(DATE_FORMAT)

    print(f"Total rows retrieved: {len(all_counts)}")
    return all_counts

def storeSplitJson(rows):
    """
    Splits the fetched data into 60/20/20 datasets.
    """
    base_dir = Path(__file__).parent / "data"
    
    total_rows = len(rows)
    if total_rows == 0:
        print("No data found to save.")
        return

    # Calculate split indices
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
        
        print(f"Saved {len(subset)} rows to {folder}/{filename} (First: {subset[0]['date']})")

def check_files_exist():
    base_dir = Path(__file__).parent / "data"
    files = [
        base_dir / "training" / "pedestrians_train.json",
        base_dir / "validation" / "pedestrians_val.json",
        base_dir / "testing" / "pedestrians_test.json"
    ]
    return all(f.exists() for f in files)

if __name__ == "__main__":
    try:
        if check_files_exist():
            choice = input("Pedestrian files already exist. Do you want to refetch? (y/[N]): ").strip().lower()
            if choice != 'y':
                print("Using existing files. Execution complete!")
                sys.exit(0)

        data = fetchEcoCounterSiteData()
        storeSplitJson(data)
    except Exception as e:
        print(f"An error occurred: {e}")