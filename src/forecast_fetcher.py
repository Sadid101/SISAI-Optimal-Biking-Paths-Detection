from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import json
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

FMI_WFS_BASE = "https://opendata.fmi.fi/wfs"

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

    def fetch_24h_forecast(self, location: FMIQueryLocation) -> pd.DataFrame:
        # Calculate 24h window from the current moment
        now_utc = datetime.now(timezone.utc)
        end_utc = now_utc + timedelta(hours=24)

        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "getFeature",
            "storedquery_id": "fmi::forecast::harmonie::surface::point::timevaluepair",
            "timestep": 60,
            "starttime": now_utc.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "endtime": end_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
        }
        params.update(location.to_params())

        r = requests.get(self.base_url, params=params, timeout=60)
        r.raise_for_status()
        
        return self._parse_xml(r.text)

    def _parse_xml(self, xml_text: str) -> pd.DataFrame:
        root = ET.fromstring(xml_text)
        ns = {"wml2": "http://www.opengis.net/waterml/2.0"}

        # Isolate only the 'Temperature' timeseries
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
                        val = float(v.text) if v.text not in ("NaN", "") else None
                        points.append((t.text, val))
                    except ValueError:
                        continue
            
            if points:
                idx = pd.to_datetime([p[0] for p in points], utc=True).tz_convert("Europe/Helsinki")
                return pd.DataFrame(data=[p[1] for p in points], index=idx, columns=["temp_c"])

        raise ValueError("Temperature forecast data not found in response.")

    def to_json_payload(self, df: pd.DataFrame, location: FMIQueryLocation) -> Dict[str, Any]:
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
    fmi = FMIWeatherFetcher()
    loc = FMIQueryLocation(place="Oulu")

    try:
        df_forecast = fmi.fetch_24h_forecast(loc)
        print("Next 24 Hours Forecast (Local Time):")
        print(df_forecast)

        payload = fmi.to_json_payload(df_forecast, loc)
        with open("weather_forecast.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print("\nSuccessfully saved forecast to weather_forecast.json")
            
    except Exception as e:
        print(f"Error fetching forecast: {e}")