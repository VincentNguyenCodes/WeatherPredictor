import csv
import json
from datetime import date, timedelta
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

import torch
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.conf import settings

from .ml.model import WeatherNet, build_features, SEQ_DAYS

HIST_YEARS = 7

# San Jose International Airport coordinates
LAT, LON = 37.3622, -121.9289

_model     = None
_all_data  = None   # {year: {doy: (tmax, tmin, precip)}}
_data_mtime = None  # track CSV changes so cache auto-refreshes after update_actuals


def _load_data():
    global _all_data, _data_mtime
    data_dir = Path(settings.DATA_DIR)
    # Auto-refresh if any CSV has been updated since last load
    current_mtime = max(
        (f.stat().st_mtime for f in data_dir.glob('SanJoseWeather*.csv')),
        default=0
    )
    if _all_data is not None and current_mtime == _data_mtime:
        return
    _data_mtime = current_mtime
    _all_data = {}
    for f in sorted(data_dir.glob('SanJoseWeather*.csv')):
        with open(f, newline='') as fp:
            for row in csv.DictReader(fp):
                yr  = int(row['year'])
                doy = int(row['day_of_year'])
                tx, tn = row['tmax'], row['tmin']
                if tx == '' or tn == '':
                    continue
                precip = float(row['precip']) if row.get('precip', '') != '' else 0.0
                _all_data.setdefault(yr, {})[doy] = (float(tx), float(tn), precip)


def _get_model():
    global _model
    if _model is None:
        m = WeatherNet()
        weights = Path(settings.MODEL_PATH)
        if not weights.exists():
            raise FileNotFoundError(f"Model weights not found at {weights}. Run training first.")
        m.load_state_dict(torch.load(weights, map_location='cpu'))
        m.eval()
        _model = m
    return _model


def _fetch_todays_weather():
    """
    Fetch today's actual high/low from Open-Meteo forecast API (free, no key).
    Returns (tmax, tmin, precip) in °F or None if the request fails.
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
        f"&timezone=America%2FLos_Angeles"
        f"&forecast_days=1"
        f"&temperature_unit=fahrenheit"
    )
    try:
        with urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        daily = data.get('daily', {})
        tmax   = daily.get('temperature_2m_max', [None])[0]
        tmin   = daily.get('temperature_2m_min', [None])[0]
        precip = daily.get('precipitation_sum',  [0.0])[0] or 0.0
        if tmax is None or tmin is None:
            return None
        return round(tmax), round(tmin), round(precip, 2)
    except (URLError, KeyError, IndexError, json.JSONDecodeError):
        return None


WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
MONTHS   = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def _predict_day(model, target_date, predicted_cache):
    """
    Predict tmax/tmin for target_date.
    predicted_cache: {date: (tmax, tmin, precip)} — real or predicted values for prior days.
    """
    _load_data()
    doy = target_date.timetuple().tm_yday
    yr  = target_date.year

    # Same-day historical from CSV data
    same_day = []
    for past_yr in sorted(_all_data.keys()):
        if past_yr >= yr:
            continue
        if doy in _all_data.get(past_yr, {}):
            entry = _all_data[past_yr][doy]
            same_day.append((entry[0], entry[1]))
    same_day = same_day[-HIST_YEARS:]

    # Sequential: last SEQ_DAYS days (prefer real data from cache, then CSV)
    recent = []
    precip_seq = []
    for offset in range(1, SEQ_DAYS + 1):
        prev = target_date - timedelta(days=offset)
        prev_doy = prev.timetuple().tm_yday
        prev_yr  = prev.year
        if prev in predicted_cache:
            entry = predicted_cache[prev]
            recent.append((entry[0], entry[1]))
            precip_seq.append(entry[2] if len(entry) > 2 else 0.0)
        elif prev_yr in _all_data and prev_doy in _all_data[prev_yr]:
            entry = _all_data[prev_yr][prev_doy]
            recent.append((entry[0], entry[1]))
            precip_seq.append(entry[2])

    feats = build_features(same_day, recent, doy, precip_seq).unsqueeze(0)
    with torch.no_grad():
        pred = _get_model()(feats).squeeze(0)
    return round(pred[0].item()), round(pred[1].item())


@api_view(['GET'])
def predict(request):
    """
    GET /api/predict/?date=YYYY-MM-DD
    Returns a single-day prediction for any arbitrary date.
    """
    date_str = request.query_params.get('date', '').strip()
    if not date_str:
        return Response({'error': 'Provide a ?date=YYYY-MM-DD query parameter.'}, status=400)
    try:
        target = date.fromisoformat(date_str)
    except ValueError:
        return Response({'error': f'Invalid date format: {date_str!r}. Use YYYY-MM-DD.'}, status=400)

    _load_data()

    cache = {}
    yr = target.year
    if yr in _all_data:
        base = date(yr, 1, 1)
        for doy, entry in _all_data[yr].items():
            d = base + timedelta(days=doy - 1)
            if d < target:
                cache[d] = entry

    tmax, tmin = _predict_day(_get_model(), target, cache)
    doy = target.timetuple().tm_yday
    hist_years = [
        y for y in sorted(_all_data.keys())
        if y < target.year and doy in _all_data.get(y, {})
    ][-HIST_YEARS:]

    return Response({
        'date':            target.isoformat(),
        'label':           WEEKDAYS[target.weekday()],
        'short_date':      f"{MONTHS[target.month - 1]} {target.day}, {target.year}",
        'tmax':            tmax,
        'tmin':            tmin,
        'based_on_years':  hist_years,
    })


@api_view(['GET'])
def forecast(request):
    _load_data()
    today = date.today()
    predicted_cache = {}

    # Pre-load confirmed CSV data for the current year as baseline
    yr = today.year
    if yr in _all_data:
        base = date(yr, 1, 1)
        for doy, entry in _all_data[yr].items():
            d = base + timedelta(days=doy - 1)
            predicted_cache[d] = entry

    # Fetch today's real temperature from Open-Meteo — use as D-1 seed for model
    todays_actual = _fetch_todays_weather()
    is_actual = todays_actual is not None
    if is_actual:
        predicted_cache[today] = todays_actual

    results = []
    for offset in range(8):  # today + 7 days
        d = today + timedelta(days=offset)
        doy = d.timetuple().tm_yday

        if offset == 0 and is_actual:
            # Use real API data for today
            tmax, tmin = todays_actual[0], todays_actual[1]
        else:
            tmax, tmin = _predict_day(_get_model(), d, predicted_cache)
            predicted_cache[d] = (tmax, tmin, 0.0)

        hist_years_used = [
            y for y in sorted(_all_data.keys())
            if y < d.year and doy in _all_data.get(y, {})
        ][-HIST_YEARS:]

        results.append({
            'offset':         offset,
            'date':           d.isoformat(),
            'label':          'Today' if offset == 0 else WEEKDAYS[d.weekday()],
            'short_date':     f"{MONTHS[d.month - 1]} {d.day}",
            'tmax':           tmax,
            'tmin':           tmin,
            'is_actual':      offset == 0 and is_actual,
            'based_on_years': hist_years_used,
        })

    return Response({
        'location': 'San Jose, CA',
        'forecast': results,
    })
