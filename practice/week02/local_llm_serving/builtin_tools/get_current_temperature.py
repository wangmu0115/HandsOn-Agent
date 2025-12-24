from dataclasses import dataclass
from datetime import datetime

import httpx

datetime.now()


@dataclass
class LocationGeo:
    location: str
    latitude: float
    longitude: float
    country: str


def get_current_temperature(location: str, unit: str = "celsius") -> dict:
    """
    Get current temperature using Open-Meteo free weather API
    No API key required - https://open-meteo.com/
    """
    # First, get location geo
    geo = get_location_geo(location)
    if geo is None:
        return {
            "location": location,
            "error": f"Location `{location}` not found",
            "timestamp": datetime.now(tz=datetime.now().astimezone().tzinfo).strftime("%Y-%m-%d %H:%M:%S%z"),
        }
    # Second, get location weather
    url = "https://api.open-meteo.com/v1/forecast"


def get_location_geo(location: str):
    """Geocoding location"""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {
        "name": location,
        "count": 1,
        "language": "en",
        "format": "json",
    }
    resp = httpx.get(url, params=params, timeout=5.0)
    resp.raise_for_status()  # Raise the `HTTPStatusError` if not 2xx
    geo_data = resp.json()
    if not geo_data.get("results"):
        return None
    else:
        result = geo_data["results"][0]
        return LocationGeo(
            location=result.get("name", location),
            latitude=result["latitude"],
            longitude=result["longitude"],
            country=result.get("country", ""),
        )


if __name__ == "__main__":
    print(get_location_geo("Beijing"))
