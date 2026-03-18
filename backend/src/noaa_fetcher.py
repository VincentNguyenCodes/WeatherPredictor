"""
NOAA Climate Data Online (CDO) API fetcher.
Fetches TMAX, TMIN, PRCP for San Jose Intl Airport and saves CSV
matching the existing schema: year,month,day,day_of_year,tmax,tmin,precip,rained

Usage:
    python src/noaa_fetcher.py --year 2024
    python src/noaa_fetcher.py --year 2024 --token YOUR_TOKEN --output-dir data/
"""

import csv
import os
import argparse
from datetime import date, timedelta

try:
    import requests
except ImportError:
    raise ImportError("Install the 'requests' library: pip install requests")

STATION_ID = "GHCND:USW00023293"  # Norman Y. Mineta San Jose Intl Airport
API_BASE = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
DEFAULT_OUTPUT_DIR = "data"


def fetch_weather_year(year: int, token: str, output_dir: str = DEFAULT_OUTPUT_DIR) -> str:
    """
    Fetch one year of weather data from NOAA CDO and write a CSV file.

    Returns the path of the written CSV file.
    Requires a free NOAA CDO token: https://www.ncdc.noaa.gov/cdo-web/token
    """
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    params = {
        "datasetid": "GHCND",
        "datatypeid": "TMAX,TMIN,PRCP",
        "stationid": STATION_ID,
        "startdate": start_date,
        "enddate": end_date,
        "limit": 1000,
        "units": "standard",  # returns °F for temperature, inches for precip
    }
    headers = {"token": token}

    print(f"Fetching NOAA data for {year}...")
    response = requests.get(API_BASE, params=params, headers=headers, timeout=30)

    if response.status_code == 429:
        raise RuntimeError("NOAA API rate limit exceeded. Wait and retry.")
    if response.status_code != 200:
        raise RuntimeError(f"NOAA API error {response.status_code}: {response.text}")

    payload = response.json()
    results = payload.get("results", [])
    if not results:
        raise RuntimeError(f"No data returned for {year}. Check station ID and token.")

    # Group observations by date
    by_date: dict[str, dict[str, float]] = {}
    for obs in results:
        obs_date = obs["date"][:10]  # "YYYY-MM-DD"
        datatype = obs["datatype"]
        value = obs["value"]
        by_date.setdefault(obs_date, {})[datatype] = value

    # Build rows in schema order, filling missing values with empty string
    rows = []
    current = date(year, 1, 1)
    end = date(year, 12, 31)
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        obs = by_date.get(date_str, {})

        tmax = obs.get("TMAX", "")
        tmin = obs.get("TMIN", "")
        precip_raw = obs.get("PRCP", "")

        # Round temps to nearest integer (matches existing CSV format)
        if tmax != "":
            tmax = round(float(tmax))
        if tmin != "":
            tmin = round(float(tmin))

        if precip_raw != "":
            precip = round(float(precip_raw), 2)
            rained = 1 if precip > 0 else 0
        else:
            precip = ""
            rained = ""

        rows.append({
            "year": current.year,
            "month": current.month,
            "day": current.day,
            "day_of_year": current.timetuple().tm_yday,
            "tmax": tmax,
            "tmin": tmin,
            "precip": precip,
            "rained": rained,
        })
        current += timedelta(days=1)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"SanJoseWeather{year}.csv")

    fieldnames = ["year", "month", "day", "day_of_year", "tmax", "tmin", "precip", "rained"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Fetch NOAA weather data for San Jose, CA")
    parser.add_argument("--year", type=int, required=True, help="Year to fetch (e.g. 2024)")
    parser.add_argument(
        "--token",
        default=os.environ.get("NOAA_CDO_TOKEN", ""),
        help="NOAA CDO API token (or set NOAA_CDO_TOKEN env var). "
             "Get one free at https://www.ncdc.noaa.gov/cdo-web/token",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for CSV")
    args = parser.parse_args()

    if not args.token:
        parser.error(
            "NOAA CDO token required. Pass --token or set the NOAA_CDO_TOKEN environment variable.\n"
            "Get a free token at: https://www.ncdc.noaa.gov/cdo-web/token"
        )

    fetch_weather_year(year=args.year, token=args.token, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
