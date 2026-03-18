"""
Django management command to fetch and store actual daily temperatures.

Fetches confirmed actual high/low/precip from the Open-Meteo archive API
(free, no API key required) and updates the local CSV file for that year.
The views layer auto-detects the file change and reloads on next request.

Usage:
    # Update yesterday's actuals (default)
    python manage.py update_actuals

    # Update a specific date
    python manage.py update_actuals --date 2026-03-15

    # Backfill a range of dates
    python manage.py update_actuals --date 2026-03-01 --end-date 2026-03-17
"""

import csv
import json
from datetime import date, timedelta
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

from django.core.management.base import BaseCommand
from django.conf import settings

LAT, LON = 37.3622, -121.9289


def fetch_actuals(start: date, end: date) -> dict:
    """
    Fetch confirmed daily actuals from Open-Meteo archive API.
    Returns {date: (tmax, tmin, precip)} in °F.
    Note: archive data is typically available with a 1–2 day delay.
    """
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date={start.isoformat()}"
        f"&end_date={end.isoformat()}"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
        f"&timezone=America%2FLos_Angeles"
        f"&temperature_unit=fahrenheit"
    )
    with urlopen(url, timeout=15) as resp:
        data = json.loads(resp.read())

    daily = data.get('daily', {})
    dates  = daily.get('time', [])
    tmaxes = daily.get('temperature_2m_max', [])
    tmins  = daily.get('temperature_2m_min', [])
    precips = daily.get('precipitation_sum', [])

    result = {}
    for i, d_str in enumerate(dates):
        tx = tmaxes[i] if i < len(tmaxes) else None
        tn = tmins[i]  if i < len(tmins)  else None
        pr = precips[i] if i < len(precips) else 0.0
        if tx is None or tn is None:
            continue
        result[date.fromisoformat(d_str)] = (round(tx), round(tn), round(pr or 0.0, 2))
    return result


def update_csv(data_dir: Path, actuals: dict) -> list:
    """
    Update CSV files with actual data. Creates new year files if needed.
    Returns list of (date, tmax, tmin, precip) for each updated row.
    """
    # Group by year
    by_year: dict[int, dict] = {}
    for d, vals in actuals.items():
        by_year.setdefault(d.year, {})[d] = vals

    updated = []
    fieldnames = ['year', 'month', 'day', 'day_of_year', 'tmax', 'tmin', 'precip', 'rained']

    for yr, year_actuals in sorted(by_year.items()):
        csv_path = data_dir / f"SanJoseWeather{yr}.csv"

        # Read existing rows
        existing = {}
        if csv_path.exists():
            with open(csv_path, newline='') as f:
                for row in csv.DictReader(f):
                    doy = int(row['day_of_year'])
                    existing[doy] = row

        # Apply updates
        for d, (tmax, tmin, precip) in year_actuals.items():
            doy = d.timetuple().tm_yday
            rained = 1 if precip > 0 else 0
            existing[doy] = {
                'year': d.year, 'month': d.month, 'day': d.day,
                'day_of_year': doy,
                'tmax': tmax, 'tmin': tmin,
                'precip': precip, 'rained': rained,
            }
            updated.append((d, tmax, tmin, precip))

        # Write back sorted by day_of_year
        rows = [existing[k] for k in sorted(existing)]
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return updated


class Command(BaseCommand):
    help = 'Fetch actual daily temperatures from Open-Meteo and update CSV data files.'

    def add_arguments(self, parser):
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        parser.add_argument(
            '--date', default=yesterday,
            help='Start date to fetch actuals for (YYYY-MM-DD). Defaults to yesterday.'
        )
        parser.add_argument(
            '--end-date', default=None,
            help='End date for a range (YYYY-MM-DD). Defaults to same as --date.'
        )

    def handle(self, *args, **options):
        start = date.fromisoformat(options['date'])
        end   = date.fromisoformat(options['end_date']) if options['end_date'] else start
        data_dir = Path(settings.DATA_DIR)

        self.stdout.write(f"Fetching actuals from Open-Meteo: {start} → {end} ...")
        try:
            actuals = fetch_actuals(start, end)
        except URLError as e:
            self.stderr.write(self.style.ERROR(f"Network error: {e}"))
            return
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"Failed to fetch data: {e}"))
            return

        if not actuals:
            self.stdout.write(self.style.WARNING(
                "No data returned. Archive data may not be available yet "
                "(Open-Meteo archive has a 1–2 day delay)."
            ))
            return

        updated = update_csv(data_dir, actuals)

        for d, tmax, tmin, precip in updated:
            self.stdout.write(
                self.style.SUCCESS(
                    f"  {d.strftime('%b %d, %Y')}  High: {tmax}°F  Low: {tmin}°F  "
                    f"Precip: {precip}\""
                )
            )

        self.stdout.write(self.style.SUCCESS(
            f"\nUpdated {len(updated)} day(s). "
            f"The API will auto-reload data on next request."
        ))
