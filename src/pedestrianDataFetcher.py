import requests
import json
from pathlib import Path

# MODIFY THESE TO SET THE DATE RANGE
START_DATE = "2025-01-01T00:00:00"
END_DATE = "2025-02-28T23:59:00"

PRECISION = "hour"  # "hour" or "month"

DOMAIN = "Oulu_kaupunki" 

ID="100025213"  # change if you want another site


# DO NOT TOUCH BELOW UNLESS YOU KNOW WHAT YOU ARE DOING

# API URL
URL = "https://api.oulunliikenne.fi/proxy/graphql"

# EARLIEST DATE TO CONTAIN SENSIBLE DATA
EARLIEST_DATE = "2010-12-31T22:00:00.000Z"

# LATEST DATE TO CONTAIN SENSIBLE DATA
LATEST_DATE = "2021-05-31T23:59:59"

# LATEST DATE WITH ANY DATA, LOOKS INCOMPLETE AND OFTER SIMPLY 0s
LATEST_DATE_WITH_DATA = "2022-07-25T21:00:00.000Z"

def fetchAllEcoCounterSites():
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


def fetchEcoCounterSite(id=ID, domain=DOMAIN, step=PRECISION, begin=START_DATE, end=END_DATE):
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

def storeJson(data, filename=f'{START_DATE}_to_{END_DATE}_{PRECISION}_{DOMAIN}_{ID}_TEST2.json'):
        # Create rawData/pedestrians directory if it doesn't exist
    output_dir = Path("rawData") / "pedestrians"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract date portions for filename
    start_date_str = START_DATE.split('T')[0]
    end_date_str = END_DATE.split('T')[0]
    filename = output_dir / f'{start_date_str}_to_{end_date_str}_{PRECISION}_{DOMAIN}_{ID}.json'
    
    # Save the JSON response to a file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Data successfully saved to {filename}")

if __name__ == "__main__":
    eco_counter_sites = fetchEcoCounterSite(ID, DOMAIN, PRECISION, START_DATE, END_DATE)
    #eco_counter_sites = fetchAllEcoCounterSites()
    storeJson(eco_counter_sites)
