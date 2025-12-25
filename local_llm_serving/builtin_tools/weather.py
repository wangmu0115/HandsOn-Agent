from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx


@dataclass
class LocationTempError:
    location: str
    error: str
    timestamp: str = datetime.now(tz=datetime.now().astimezone().tzinfo).strftime("%Y-%m-%d %H:%M:%S%z")


# WMO Weather interpretation codes
WeatherCodes = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "foggy",
    48: "foggy",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "light rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "light snow",
    73: "moderate snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light rain showers",
    81: "moderate rain showers",
    82: "heavy rain showers",
    85: "light snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with light hail",
    99: "thunderstorm with heavy hail",
}


def get_current_temperature(location: str, unit: str = "celsius") -> dict:
    """
    Get current temperature using Open-Meteo free weather API
    No API key required - https://open-meteo.com/
    """
    try:
        # get location geo
        geo = _get_location_geo(location)
        if geo is None:
            return LocationTempError(location, f"Location `{location}` not found")
        location_name = f"{geo['location']}, {geo['country']}"
        # get location weather
        temp_unit = "fahrenheit" if unit is not None and unit.lower() == "fahrenheit" else "celsius"
        weather = _get_location_weather(geo["latitude"], geo["longitude"], temp_unit)
        if weather is None:
            return LocationTempError(location_name, "Weather data not available")
        weather_code = weather.get("weather_code", 0)
        conditions = WeatherCodes.get(weather_code, "unknown")
        unit_symbol = "°F" if temp_unit == "fahrenheit" else "°C"
        tz = ZoneInfo(weather["timezone"])
        if "time" in weather:
            weather_time = datetime.strptime(weather["time"], "%Y-%m-%dT%H:%M").replace(tzinfo=tz)
        else:
            weather_time = datetime.now(tz)
        return {
            "location": location_name,
            "temperature": round(weather["temperature_2m"], 1),
            "unit": unit_symbol,
            "conditions": conditions,
            "humidity": weather.get("relative_humidity_2m"),
            "wind_speed": round(weather.get("wind_speed_10m", 0), 1),
            "wind_unit": "km/h",
            "coordinates": {"latitude": geo["latitude"], "longitude": geo["longitude"]},
            "timestamp": weather_time.strftime("%Y-%m-%d %H:%M%z"),
            "source": "Open-Meteo, https://open-meteo.com/",
        }
    except Exception as e:
        import logging

        logging.error(f"Open-Meteo API error: {e}")
        return LocationTempError(location, str(e))


def _get_location_geo(location: str):
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
        return {
            "location": result.get("name", location),
            "latitude": result["latitude"],
            "longitude": result["longitude"],
            "country": result.get("country", ""),
        }


def _get_location_weather(latitude: float, longitude: float, temp_unit: str):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
        "temperature_unit": temp_unit,
        "timezone": "auto",
    }
    resp = httpx.get(url, params=params, timeout=5.0)
    resp.raise_for_status()
    weather_data = resp.json()
    if "current" not in weather_data:
        return None
    else:
        weather = weather_data["current"]
        weather["timezone"] = weather_data["timezone"]
        return weather


if __name__ == "__main__":
    print(get_current_temperature("Beijing"))
