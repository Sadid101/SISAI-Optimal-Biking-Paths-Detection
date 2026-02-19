import requests

URL = "https://api.oulunliikenne.fi/proxy/graphql"

QUERY1 = """
query GetAllBikeRentalStations {
  bikeRentalStations {
    id
    stationId
    name
    bikesAvailable
    spacesAvailable
    state
    allowDropoff
    lat
    lon
  }
}
"""

QUERY2 = """
query GetAllEcoCounterSites {
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
}
"""

LATEST_DATE = "2022-07-25T21:00:00.000Z"
START_DATE = "2022-01-01T00:00:00"
END_DATE = "2022-07-25T21:00:00.000Z"
QUERYTEST = """query GetEcoCounterSiteData {
  ecoCounterSiteData(
    id: "100000647",
    domain: Oulu_kaupunki,
    step: month,
    begin: "2022-01-01T00:00:00"
  ) {
    date
    counts
  }
}"""

def fetch_bike_rental_stations():
    payload = {"query": QUERY1}
    headers = {
        "Content-Type": "application/json",
        # Add auth headers here if the API needs them, e.g.:
        # "Authorization": "Bearer YOUR_TOKEN"
    }

    print("Fetching bike rental stations...")
    r = requests.post(URL, json=payload, headers=headers, timeout=10)
    r.raise_for_status()  # raises if non-2xx
    data = r.json()

    # GraphQL returns {"data": {...}, "errors": [...]}
    if "errors" in data:
        raise RuntimeError(f"GraphQL errors: {data['errors']}")

    return data["data"]["bikeRentalStations"]

def fetchAllEcoCounterSites(id='100000647', domain='Oulu_kaupunki', step='month', begin='2022-01-01T00:00:00', end=END_DATE):
    constructedQuery = """query GetEcoCounterSiteData {{
  ecoCounterSiteData(
    id: "{id}",
    domain: {domain},
    step: {step},
    begin: "{begin}"
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

if __name__ == "__main__":
    eco_counter_sites = fetchAllEcoCounterSites('100000647', 'Oulu_kaupunki', 'month', START_DATE, END_DATE)
    print(eco_counter_sites)