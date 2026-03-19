import csv
import os
import argparse
import urllib.request
import json
import time
from datetime import date, timedelta
from pathlib import Path

LATITUDE  = 37.3382
LONGITUDE = -121.8863
API_BASE  = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parents[1] / "data")


def fetch_year(year: int, output_dir: str = DEFAULT_OUTPUT_DIR) -> str:
    start_date = f"{year}-01-01"
    end_date   = f"{year}-12-31"

    url = (
        f"{API_BASE}?latitude={LATITUDE}&longitude={LONGITUDE}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
        f"&temperature_unit=fahrenheit"
        f"&timezone=America%2FLos_Angeles"
    )

    req  = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=30)
    data = json.loads(resp.read())

    daily      = data["daily"]
    dates      = daily["time"]
    tmax_vals  = daily["temperature_2m_max"]
    tmin_vals  = daily["temperature_2m_min"]
    precip_vals = daily["precipitation_sum"]

    rows = []
    for date_str, tmax, tmin, precip in zip(dates, tmax_vals, tmin_vals, precip_vals):
        d = date.fromisoformat(date_str)
        tmax   = round(float(tmax))   if tmax   is not None else ""
        tmin   = round(float(tmin))   if tmin   is not None else ""
        precip = round(float(precip), 2) if precip is not None else ""
        rained = (1 if precip > 0 else 0) if precip != "" else ""
        rows.append({
            "year":       d.year,
            "month":      d.month,
            "day":        d.day,
            "day_of_year": d.timetuple().tm_yday,
            "tmax":       tmax,
            "tmin":       tmin,
            "precip":     precip,
            "rained":     rained,
        })

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"SanJoseWeather{year}.csv")
    fieldnames  = ["year", "month", "day", "day_of_year", "tmax", "tmin", "precip", "rained"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  {year}: {len(rows)} rows -> {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=1950)
    parser.add_argument("--end-year",   type=int, default=2014)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    print(f"Fetching Open-Meteo data for San Jose, CA ({args.start_year}-{args.end_year})...")
    for year in range(args.start_year, args.end_year + 1):
        try:
            fetch_year(year, args.output_dir)
            time.sleep(0.5)
        except Exception as e:
            print(f"  {year}: ERROR - {e}")

    print("Done.")


if __name__ == "__main__":
    main()
