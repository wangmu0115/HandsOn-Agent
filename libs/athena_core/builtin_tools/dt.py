from datetime import datetime
from zoneinfo import ZoneInfo

timezone_aliases = {
    "EST": "America/New_York",
    "EDT": "America/New_York",
    "PST": "America/Los_Angeles",
    "PDT": "America/Los_Angeles",
    "CST": "America/Chicago",
    "CDT": "America/Chicago",
    "MST": "America/Denver",
    "MDT": "America/Denver",
    "GMT": "Europe/London",
    "BST": "Europe/London",
    "CET": "Europe/Paris",
    "CEST": "Europe/Paris",
    "JST": "Asia/Tokyo",
    "IST": "Asia/Kolkata",
    "AEST": "Australia/Sydney",
    "AEDT": "Australia/Sydney",
    "SGT": "Asia/Singapore",
    "HKT": "Asia/Hong_Kong",
    "UTC+1": "Etc/GMT-1",  # Note: signs are inverted in Etc/GMT
    "UTC-1": "Etc/GMT+1",
    "UTC+8": "Etc/GMT-8",
    "UTC-8": "Etc/GMT+8",
}


def get_current_time(timezone: str = "UTC") -> dict:
    """
    Get current date and time in specified timezone using zoneinfo (Python 3.9+)
    """
    tz_name = timezone_aliases.get(timezone.upper(), timezone)
    try:
        tz = ZoneInfo(tz_name)
        current_dt = datetime.now(tz)
        return {
            "timezone": tz_name,
            "datetime": current_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "date": current_dt.strftime("%Y-%m-%d"),
            "time": current_dt.strftime("%H:%M:%S"),
            "day_of_week": current_dt.strftime("%A"),
            "utc_offset": current_dt.strftime("%z"),
            "timestamp": current_dt.strftime("%Y-%m-%d %H:%M:%S%z"),
        }
    except Exception as e:
        return {
            "error": str(e),
            "timezone": timezone,
            "timestamp": datetime.now(tz=datetime.now().astimezone().tzinfo).strftime("%Y-%m-%d %H:%M:%S%z"),
        }
