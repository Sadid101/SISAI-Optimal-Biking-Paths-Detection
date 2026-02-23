from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import json
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

# --- CONFIGURATION ---
FMI_WFS_BASE = "https://opendata.fmi.fi/wfs"

@dataclass
class FMIQueryLocation:
    """Helper class to handle location parameters for the FMI API."""
    place: Optional[str] = None
    latlon: Optional[str] = None

    def to_params(self) -> Dict[str, str]:
        """Converts location data to API request parameters."""
        if (self.place is None) == (self.latlon is None):
            raise ValueError("Provide exactly one of: place OR latlon")
        return {"place": self.place} if self.place else {"latlon": self.latlon}

class FMIWeatherFetcher:
    """Class to fetch and parse weather forecast data from Finnish Meteorological Institute (FMI)."""
    
    def __init__(self, base_url: str = FMI_WFS_BASE):
        self.base_url = base_url

    def fetch_24h_forecast(self, location: FMIQueryLocation) -> pd.DataFrame:
        """
        Fetches the next 24 hours of hourly temperature forecasts.
        
        Returns:
            pd.DataFrame: Indexed by local time with a 'temp_c' column.
        """
        # Calculate 24h window from the current moment in UTC
        now_utc = datetime.now(timezone.utc)
        end_utc = now_utc + timedelta(hours=24)

        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "getFeature",
            "storedquery_id": "fmi::forecast::harmonie::surface::point::timevaluepair",
            "timestep": 60, # Hourly steps
            "starttime": now_utc.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "endtime": end_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
        }
        params.update(location.to_params())

        # Perform the HTTP GET request
        r = requests.get(self.base_url, params=params, timeout=60)
        r.raise_for_status()
        
        return self._parse_xml(r.text)

    def _parse_xml(self, xml_text: str) -> pd.DataFrame:
        """Parses the FMI WFS XML response into a Pandas DataFrame."""
        root = ET.fromstring(xml_text)
        ns = {"wml2": "http://www.opengis.net/waterml/2.0"}

        # The API returns multiple timeseries (wind, humidity, etc.)
        # We isolate only the one containing 'Temperature' in its ID
        for ts in root.findall(".//wml2:MeasurementTimeseries", ns):
            series_id = ts.attrib.get("{http://www.opengis.net/gml/3.2}id", "")
            if "Temperature" not in series_id:
                continue

            points = []
            for p in ts.findall(".//wml2:point", ns):
                t = p.find(".//wml2:time", ns)
                v = p.find(".//wml2:value", ns)
                if t is not None and v is not None:
                    try:
                        # Handle potential missing or NaN values in the XML
                        val = float(v.text) if v.text not in ("NaN", "") else None
                        points.append((t.text, val))
                    except ValueError:
                        continue
            
            if points:
                # Convert UTC strings to local Helsinki time
                idx = pd.to_datetime([p[0] for p in points], utc=True).tz_convert("Europe/Helsinki")
                return pd.DataFrame(data=[p[1] for p in points], index=idx, columns=["temp_c"])

        raise ValueError("Temperature forecast data not found in response.")

    def to_json_payload(self, df: pd.DataFrame, location: FMIQueryLocation) -> Dict[str, Any]:
        """Converts the DataFrame into a dictionary ready for JSON serialization."""
        rows = []
        for ts, row in df.iterrows():
            rows.append({
                "ts": ts.isoformat(),
                "temp_c": float(row["temp_c"]) if pd.notna(row["temp_c"]) else None
            })

        return {
            "kind": "forecast_24h",
            "location": {"place": location.place} if location.place else {"latlon": location.latlon},
            "rows": rows
        }

if __name__ == "__main__":
    # --- SETUP PATHS ---
    # Path(__file__).parent points to 'src'
    base_dir = Path(__file__).parent
    output_dir = base_dir / "data" / "training"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = output_dir / "weather_forecast.json"

    fmi = FMIWeatherFetcher()
    loc = FMIQueryLocation(place="Oulu")

    try:
        # Fetch data
        df_forecast = fmi.fetch_24h_forecast(loc)
        print("Next 24 Hours Forecast (Local Time):")
        print(df_forecast)

        # Convert to JSON structure
        payload = fmi.to_json_payload(df_forecast, loc)
        
        # Save to file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            
        print(f"\nSuccessfully saved forecast to: {file_path}")
            
    except Exception as e:
        print(f"Error fetching forecast: {e}")