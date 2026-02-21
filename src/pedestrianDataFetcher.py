import requests
import json
from pathlib import Path

# MODIFY THESE TO SET THE DATE RANGE
START_DATE = "2020-02-20T14:00:00"
END_DATE = "2020-02-22T15:00:00"

PRECISION = "hour"  # "hour" or "month"

LOCATION = "Oulu_kaupunki" 

ID="100000647"  # Oulu city center, change if you want another site


# DO NOT TOUCH BELOW UNLESS YOU KNOW WHAT YOU ARE DOING

# API URL
URL = "https://api.oulunliikenne.fi/proxy/graphql"

# EARLIEST DATE TO CONTAIN SENSIBLE DATA
EARLIEST_DATE = "2010-12-31T22:00:00.000Z"

# LATEST DATE TO CONTAIN SENSIBLE DATA
LATEST_DATE = "2021-05-31T23:59:59"

# LATEST DATE WITH ANY DATA, LOOKS INCOMPLETE AND OFTER SIMPLY 0s
LATEST_DATE_WITH_DATA = "2022-07-25T21:00:00.000Z"

def fetchAllEcoCounterSites(id=ID, domain=LOCATION, step=PRECISION, begin=START_DATE, end=END_DATE):
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
        
    
    payload = {"query": constructedQuery.format(id=id, domain=domain, step=step, begin=begin, end=end)}
    headers = {
        "Content-Type": "application/json",
        # Add auth headers here if the API needs them, e.g.:
        # "Authorization": "Bearer YOUR_TOKEN"
    }

    print("Fetching eco counter sites...")
    r = requests.post(URL, json=payload, headers=headers, timeout=10)
    r.raise_for_status()  # raises if non-2xx
    data = r.json()

    # GraphQL returns {"data": {...}, "errors": [...]}
    if "errors" in data:
        raise RuntimeError(f"GraphQL errors: {data['errors']}")

    return data["data"]

def storeJson(data, filename=f'{START_DATE}_to_{END_DATE}_{PRECISION}_{LOCATION}_{ID}.json'):
        # Create rawData/pedestrians directory if it doesn't exist
    output_dir = Path("rawData") / "pedestrians"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract date portions for filename
    start_date_str = START_DATE.split('T')[0]
    end_date_str = END_DATE.split('T')[0]
    filename = output_dir / f'{start_date_str}_to_{end_date_str}_{PRECISION}_{LOCATION}_{ID}.json'
    
    # Save the JSON response to a file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Data successfully saved to {filename}")

if __name__ == "__main__":
    eco_counter_sites = fetchAllEcoCounterSites(ID, LOCATION, PRECISION, START_DATE, END_DATE)
    storeJson(eco_counter_sites)
