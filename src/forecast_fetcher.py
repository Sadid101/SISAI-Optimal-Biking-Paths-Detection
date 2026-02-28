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
DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

@dataclass
class FMIQueryLocation:
    place: Optional[str] = None
    latlon: Optional[str] = None

    def to_params(self) -> Dict[str, str]:
        if (self.place is None) == (self.latlon is None):
            raise ValueError("Provide exactly one of: place OR latlon")
        return {"place": self.place} if self.place else {"latlon": self.latlon}

class FMIWeatherFetcher:
    def __init__(self, base_url: str = FMI_WFS_BASE):
        self.base_url = base_url

    def fetch_forecast(self, location: FMIQueryLocation, start_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Fetches the official 7-day hourly forecast.
        """
        # If no start_time provided, use 'now'
        if start_time is None:
            start_time = (datetime.now(timezone.utc) + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    
        end_time = start_time + timedelta(days=7)

        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "getFeature",
            "storedquery_id": "fmi::forecast::edited::weather::scandinavia::point::timevaluepair",
            "timestep": 60,
            "starttime": start_time.strftime(DATE_FORMAT),
            "endtime": end_time.strftime(DATE_FORMAT)
        }
        params.update(location.to_params())

        print(f"Requesting 7-day forecast for: {location.place or location.latlon}")
        r = requests.get(self.base_url, params=params, timeout=60)
        r.raise_for_status()
        
        return self._parse_xml(r.text)
    
    def _parse_xml(self, xml_text: str) -> pd.DataFrame:
        """
        Parses FMI XML response into a Pandas DataFrame.
        """
        root = ET.fromstring(xml_text)
        ns = {"wml2": "http://www.opengis.net/waterml/2.0"}

        for ts in root.findall(".//wml2:MeasurementTimeseries", ns):
            series_id = ts.attrib.get("{http://www.opengis.net/gml/3.2}id", "").lower()
            if "temperature" in series_id or "t2m" in series_id:
                points = []
                for p in ts.findall(".//wml2:point", ns):
                    t = p.find(".//wml2:time", ns)
                    v = p.find(".//wml2:value", ns)
                    if t is not None and v is not None:
                        try:
                            val = round(float(v.text), 1) if v.text not in ("NaN", "") else None
                            points.append((t.text, val))
                        except ValueError:
                            continue
                
                if points:
                    idx = pd.to_datetime([p[0] for p in points], utc=True).tz_convert("Europe/Helsinki")
                    return pd.DataFrame(data=[p[1] for p in points], index=idx, columns=["temp_c"]).dropna()

        raise ValueError("Temperature data not found. Check if the location is correct.")

    def to_json_payload(self, df: pd.DataFrame, location: FMIQueryLocation) -> Dict[str, Any]:
        """
        Converts the DataFrame into a JSON dictionary using the place name.
        """
        rows = [{"ts": ts.isoformat(), "temp_c": float(row["temp_c"])} 
                for ts, row in df.iterrows()]
        
        return {
            "kind": "forecast_7d_hourly",
            "location": {"place": location.place} if location.place else {"latlon": location.latlon},
            "rows": rows
        }

if __name__ == "__main__":
    # Setup directories
    base_dir = Path(__file__).parent
    output_dir = base_dir / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / "weather_forecast.json"

    # Configure fetcher for Oulu
    fmi = FMIWeatherFetcher()
    loc = FMIQueryLocation(place="Oulu")

    try:
        df_forecast = fmi.fetch_forecast(loc)
        total_records = len(df_forecast)
        start_str = df_forecast.index.min().strftime('%d.%m. %H:%M')
        end_str = df_forecast.index.max().strftime('%d.%m. %H:%M')

        print(f"\nSuccess! Found {total_records} hourly records.")
        print(f"Coverage: {start_str} —> {end_str}")
        print("\nUpcoming forecast snippet:")
        print(df_forecast.head(3))

        # Save to JSON
        payload = fmi.to_json_payload(df_forecast, loc)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            
        print(f"\n[OK] Forecast saved to: {file_path}")
            
    except Exception as e:
        print(f"Error occurred: {e}")