# San Jose Weather Predictor

A full-stack weather prediction application that uses a PyTorch neural network trained on 12 years of historical San Jose, CA temperature data (2015–2026) to forecast high and low temperatures for the upcoming week and any arbitrary future date.

---

## Features

- **Real-time today** - today's high/low is pulled live from the Open-Meteo API (no key required), not predicted
- **6-day model forecast** - days 1–6 predicted by WeatherNet, seeded with the last 7 days of real observed temps fetched live from the API. No predictions ever build on other predictions.
- **Daily actuals pipeline** - `python manage.py update_actuals` fetches yesterday's confirmed temps and writes them to the CSV, keeping training data current
- **Date lookup** - predict the high/low for any date by typing it in
- **Transparent predictions** - each forecast shows which historical years it was based on
- **iOS-inspired UI** - glassmorphism weather card built in React
- **REST API** - Django backend exposable to any client

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    React Frontend                        │
│         Forecast Tab  │  Predict a Date Tab             │
└──────────────┬──────────────────────────┬───────────────┘
               │ HTTP                     │ HTTP
               ▼                          ▼
┌─────────────────────────────────────────────────────────┐
│               Django REST API (port 8000)                │
│    GET /api/forecast/     GET /api/predict/?date=...     │
└──────────────────────────┬──────────────────────────────┘
                           │
               ┌───────────▼───────────┐
               │     WeatherNet (MLP)  │
               │   40 inputs → 2 out   │
               │   (tmax, tmin in °F)  │
               └───────────┬───────────┘
                           │
               ┌───────────▼───────────┐
               │   data/ (CSV files)   │
               │   2015 – 2025         │
               │   San Jose, CA        │
               └───────────────────────┘
```

---

## Neural Network Model

**Architecture:** 5-layer MLP (40 → 128 → 256 → 128 → 64 → 2)

**Input features (40 total):**
| Feature Group | Size | Description |
|---|---|---|
| Same-day historical temps (normalized) | 14 | tmax + tmin for the same calendar date across the past 7 years, divided by 100 |
| Presence flags | 7 | 1 if historical data exists for that year slot, 0 otherwise |
| Sequential prior days (normalized) | 14 | tmax + tmin from the 7 days immediately preceding the target date, divided by 100 |
| Temperature deltas | 2 | (yesterday − 2 days ago) for tmax and tmin; captures the warming/cooling trend |
| 7-day rolling precipitation | 1 | Sum of prior 7 days of precipitation, normalized; wet/dry streaks affect temps |
| Cyclical day-of-year | 2 | sin and cos encoding of day-of-year (captures seasonal patterns) |

**Output:** tmax, tmin (°F) for the target date

**Training:** Adam optimizer, Huber loss, 1,000 epochs, batch size 64

**Hold-out accuracy (tested on 2023–2025, never seen during training):**
- High temp MAE: **4.53°F** vs baseline 5.33°F
- Low temp MAE: **2.94°F** vs baseline 3.87°F

> **v1 → v2 improvements:** normalized inputs, extended history window (5 → 7 years),
> extended sequential window (3 → 7 days), added temperature delta features,
> added precipitation rolling sum, switched MSE → Huber loss.
> Low temp MAE improved by **0.51°F**; see [`docs/accuracy-report.md`](docs/accuracy-report.md) for full before/after breakdown.

See [`docs/accuracy-report.md`](docs/accuracy-report.md) for the full evaluation.

---

## Project Structure

```
.
├── backend/                    # Django REST API
│   ├── weather/
│   │   ├── views.py            # /api/forecast/ and /api/predict/ endpoints
│   │   ├── urls.py
│   │   └── ml/
│   │       ├── model.py        # WeatherNet definition + feature builder (40 inputs)
│   │       ├── train.py        # Training script
│   │       └── model_weights.pth
│   ├── weather_project/
│   │   └── settings.py
│   ├── data/                   # Historical CSV files (2015–2025)
│   │   └── SanJoseWeather{year}.csv
│   ├── src/
│   │   ├── evaluate.py         # Hold-out accuracy evaluation
│   │   ├── noaa_fetcher.py     # Optional: fetch data via NOAA CDO API
│   │   └── csv-reader.py       # CSV utility functions
│   ├── requirements.txt
│   └── manage.py
├── frontend/                   # React app
│   └── src/
│       └── components/
│           ├── WeatherApp.js   # Forecast + Predict tabs
│           └── WeatherApp.css
├── docs/
│   └── accuracy-report.md
└── README.md
```

---

## Setup & Running

### Prerequisites

- Python 3.9+
- Node.js 18+

### 1. Install Python dependencies

```bash
pip install -r backend/requirements.txt
```

### 2. Start the Django backend

```bash
cd backend
python manage.py migrate
python manage.py runserver 8000
```

API will be available at `http://127.0.0.1:8000/api/`

### 3. Start the React frontend

```bash
cd frontend
npm install
npm start
```

App will open at `http://localhost:3000`

---

## API Reference

### `GET /api/forecast/`

Returns today + 7-day predicted forecast.

**Response:**
```json
{
  "location": "San Jose, CA",
  "forecast": [
    {
      "offset": 0,
      "date": "2026-03-12",
      "label": "Today",
      "short_date": "Mar 12",
      "tmax": 65,
      "tmin": 48,
      "based_on_years": [2021, 2022, 2023, 2024, 2025]
    }
  ]
}
```

### `GET /api/predict/?date=YYYY-MM-DD`

Returns a prediction for any specific date.

**Example:** `GET /api/predict/?date=2027-06-15`

**Response:**
```json
{
  "date": "2027-06-15",
  "label": "Tuesday",
  "short_date": "Jun 15, 2027",
  "tmax": 81,
  "tmin": 57,
  "based_on_years": [2021, 2022, 2023, 2024, 2025]
}
```

---

## Retrain the Model

```bash
cd backend/weather/ml
python train.py
# Options:
python train.py --data-dir ../../data --epochs 1000
```

## Evaluate Accuracy

```bash
cd backend
python src/evaluate.py
```

## Update Daily Actuals

Run this each day to append yesterday's confirmed temperatures to the CSV.
The API auto-reloads data on next request, so no server restart is needed.

```bash
cd backend

# Update yesterday (default)
python manage.py update_actuals

# Update a specific date
python manage.py update_actuals --date 2026-03-17

# Backfill a range
python manage.py update_actuals --date 2026-01-01 --end-date 2026-03-15
```

> **Note:** Open-Meteo archive data has a 1–2 day delay, so running this for yesterday is reliable. Same-day actuals are fetched live by the forecast endpoint automatically.

Trains a fresh hold-out model on 2015–2022, evaluates on 2023–2025, and prints MAE, RMSE, per-year breakdowns, worst predictions, and monthly error breakdown.

## Fetch New Data

Data is sourced from the [Open-Meteo Historical API](https://open-meteo.com/) (free, no token required) for San Jose International Airport (37.3622°N, 121.9289°W).

Optionally, use the NOAA CDO API fetcher (requires a [free token](https://www.ncdc.noaa.gov/cdo-web/token)):

```bash
export NOAA_CDO_TOKEN=your_token
python backend/src/noaa_fetcher.py --year 2026
```

---

## Data Format

Each CSV in `data/` follows this schema:

| Column | Type | Description |
|---|---|---|
| `year` | int | Calendar year |
| `month` | int | Month (1–12) |
| `day` | int | Day of month |
| `day_of_year` | int | Day of year (1–366) |
| `tmax` | int | Daily high temperature (°F) |
| `tmin` | int | Daily low temperature (°F) |
| `precip` | float | Precipitation (inches) |
| `rained` | int | 1 if precip > 0, else 0 |

---



---

## System Design at Scale

The current implementation runs on a single server and retrains on every request. Here's how it would evolve to handle 100k+ users across multiple cities:

### Bottleneck 1 - Single city, single model
**Problem:** WeatherNet is trained on San Jose data only. Generalizing to more cities requires a new model per city, which doesn't scale.

**Solution:**
- Train a **shared global model** with city as an additional input feature (latitude, longitude, elevation, climate zone one-hot).
- One model serves all cities; city-specific fine-tuning can be layered on top for higher accuracy.
- Store per-city historical data in a structured data lake (S3 + Parquet) rather than flat CSVs.

### Bottleneck 2 - No retraining pipeline
**Problem:** New weather data arrives daily but the model is static. Predictions degrade as time passes without retraining.

**Solution:**
- Schedule a nightly **Celery + Redis** job that fetches yesterday's actuals from Open-Meteo, appends to the dataset, and retrains the model.
- Version model weights with timestamps and keep the last 3 versions for rollback.
- Compare new model MAE against the current deployed model before promoting. Only deploy if the new model improves or holds steady.

### Bottleneck 3 - No uncertainty quantification
**Problem:** The model returns a single point prediction (e.g., 72°F) with no confidence interval. Real forecasting systems communicate uncertainty.

**Solution:**
- Replace the single MLP output with a **Monte Carlo Dropout** inference pass: run the same input through the model N times with dropout enabled and report the mean and standard deviation as a confidence range.
- Surface this in the UI: "High: 72°F ± 4°F" instead of just "72°F."

### Bottleneck 4 - Synchronous prediction on every request
**Problem:** At high traffic, running inference on every API call blocks the server.

**Solution:**
- **Pre-compute and cache** the 7-day forecast once per day per city in Redis. Serve cached results instantly; only recompute when the cache expires or a new model is deployed.
- For the arbitrary date prediction endpoint, inference is fast enough to stay synchronous but should be rate-limited per IP.

### Revised architecture at scale

```
                         ┌──────────────────┐
                         │  Open-Meteo API  │  ← nightly data fetch
                         └────────┬─────────┘
                                  │
                         ┌────────▼─────────┐
                         │  Celery Worker   │  ← retrains model nightly
                         │  (retraining     │     versions weights to S3
                         │   pipeline)      │
                         └────────┬─────────┘
                                  │
┌──────────┐   HTTPS   ┌──────────▼─────────┐
│  Client  │──────────▶│    NGINX / LB       │
└──────────┘           └──────────┬──────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │ Django   │ │ Django   │ │ Django   │  ← Gunicorn workers
              └────┬─────┘ └────┬─────┘ └────┬─────┘
                   └────────────┼────────────┘
                                │
               ┌────────────────┼────────────────┐
               ▼                ▼                ▼
        ┌────────────┐  ┌────────────┐  ┌────────────┐
        │ PostgreSQL │  │   Redis    │  │    S3      │
        │ (city +    │  │ (forecast  │  │ (model     │
        │  actuals)  │  │  cache)    │  │  weights)  │
        └────────────┘  └────────────┘  └────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, CSS (glassmorphism) |
| Backend | Python 3.9, Django 4.2, Django REST Framework |
| ML | PyTorch 2.8 |
| Data | Open-Meteo Historical API, NOAA CDO API |
| Database | SQLite (Django default) |
