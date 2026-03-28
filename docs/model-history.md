# WeatherNet Model History

---

## v3 — Overfitting Fixes (Current)

### What Changed from v2

**1. Expanded dataset: 11 years to 76 years**
Dataset expanded from 2015-2026 to 1950-2026 using the Open-Meteo historical archive API (free, no key required). Going from 3,729 to 27,470 samples is the single biggest factor in reducing overfitting. The model can no longer memorize the training set when the dataset is 7x larger.

**2. Early stopping**
Training now monitors cross validation loss each epoch and stops when it has not improved for 50 consecutive epochs, saving the best weights. Previously the model trained for all 1000 epochs even after CV loss started climbing, causing the train/CV gap to widen to 1.22. Stopped at epoch 234.

**3. Weight decay (L2 regularization)**
Added `weight_decay=1e-4` to the Adam optimizer. Penalizes large weights, discouraging the model from fitting noise in the training set.

**4. Recency weighting on sequential features**
Prior day temperatures are now weighted linearly by closeness to the target date. D-1 gets weight 1.0, D-2 gets 6/7, down to D-7 which gets 1/7. Same applied to the rolling precipitation sum. This makes the model more sensitive to recent weather momentum.

### Architecture

5-layer MLP: 40 inputs → 128 → 256 → 128 → 64 → 2 outputs (tmax, tmin in °F)

### Input Features (40 total)

| Feature Group | Size | Description |
|---|---|---|
| Same-day historical temps (normalized) | 14 | tmax + tmin for the same calendar date across the past 7 years, divided by 100 |
| Presence flags | 7 | 1 if historical data exists for that year slot, 0 otherwise |
| Sequential prior days (normalized) | 14 | tmax + tmin from the 7 days preceding the target date, divided by 100 with recency weighting |
| Temperature deltas | 2 | (yesterday minus 2 days ago) for tmax and tmin, normalized |
| 7-day rolling precipitation | 1 | Sum of prior 7 days of precipitation, normalized with recency weighting |
| Cyclical day-of-year | 2 | sin/cos encoding of day-of-year |

---

## v2 — Improved Model

### What Changed from v1

**1. Input normalization**
All temperature values are divided by 100 before being fed into the network. Previously, raw °F values (30-105) sat alongside cyclical sin/cos features clamped to [-1, 1]. This 100x scale difference caused the optimizer to over-weight temperature features while the cyclical seasonal signal was drowned out.

**2. Extended historical years: 5 to 7**
The original model only looked back 5 years on the same calendar date. Increasing to 7 gives the model more data points to estimate typical conditions for each day of year, reducing variance in the historical signal.

**3. Extended sequential window: 3 to 7 days**
Weather systems have momentum that extends beyond 3 days. A cold front, heat wave, or marine layer typically persists for a week or more. Extending the sequential window lets the model see the full arc of the current weather pattern.

**4. Temperature delta features**
Two new features: tmax_delta and tmin_delta, defined as yesterday's temperature minus the day before's temperature (normalized). This gives the model an explicit signal for whether temperatures are trending up or down.

**5. Precipitation features**
The CSV data includes daily precipitation but it was never used. A 7-day rolling precipitation sum (normalized) is now included. Wet stretches suppress high temps and moderate lows; dry spells allow temperatures to swing further. This particularly improved low temperature accuracy.

**6. Huber loss (replaces MSE)**
MSE penalizes large errors quadratically, causing the model to over-prioritize rare extreme events at the expense of normal-day accuracy. Huber loss behaves like MSE for small errors but clips large errors linearly, making the model more robust on typical days.

### Architecture

5-layer MLP: 40 inputs → 128 → 256 → 128 → 64 → 2 outputs (tmax, tmin in °F)

### Input Features (40 total)

| Feature Group | Size | Description |
|---|---|---|
| Same-day historical temps (normalized) | 14 | tmax + tmin for the same calendar date across the past 7 years, divided by 100 |
| Presence flags | 7 | 1 if historical data exists for that year slot, 0 otherwise |
| Sequential prior days (normalized) | 14 | tmax + tmin from the 7 days preceding the target date, divided by 100 |
| Temperature deltas | 2 | (yesterday minus 2 days ago) for tmax and tmin, normalized |
| 7-day rolling precipitation | 1 | Sum of prior 7 days of precipitation, normalized |
| Cyclical day-of-year | 2 | sin/cos encoding of day-of-year |

---

## v1 — Original Model

### Architecture

5-layer MLP: 23 inputs → 128 → 256 → 128 → 64 → 2 outputs (tmax, tmin in °F)

### Input Features (23 total)

| Feature Group | Size | Description |
|---|---|---|
| Same-day historical temps | 10 | Raw tmax + tmin for the same calendar date across the past 5 years |
| Presence flags | 5 | 1 if historical data exists for that year slot, 0 otherwise |
| Sequential prior days | 6 | tmax + tmin from the 3 days immediately preceding the target date |
| Cyclical day-of-year | 2 | sin/cos encoding of day-of-year |

### Known Weaknesses
- Raw °F values (30-105) mixed with normalized sin/cos features (-1 to 1), large input scale imbalance
- Only 3 prior days of context, too short to capture weather momentum
- Only 5 years of same-day history
- Precipitation data available in CSVs but completely ignored
- No trend signal, model could not tell if temperatures were rising or falling

---

## Data Tables

### Training Configuration Comparison

| Parameter | v1 | v2 | v3 |
|---|---|---|---|
| Dataset | 2015-2025 | 2015-2026 | 1950-2026 |
| Samples | ~2,700 | 3,729 | 27,470 |
| Loss function | MSE | HuberLoss | HuberLoss |
| Optimizer | Adam | Adam | Adam (weight_decay=1e-4) |
| Early stopping | No | No | Yes (patience=50, stopped epoch 234) |
| Historical years | 5 | 7 | 7 |
| Sequential days | 3 | 7 | 7 (recency weighted) |
| Input features | 23 | 40 | 40 |

### Overfitting Comparison (Train/CV Gap)

| Metric | v2 | v3 | Change |
|---|---|---|---|
| Dataset size | 3,729 | 27,470 | +636% |
| Epochs trained | 1000 | 234 | -766 |
| Final train loss | 1.8390 | 2.5654 | higher (less memorization) |
| Final CV loss | 3.0590 | 2.5855 | -0.4735 |
| Train/CV gap | 1.2200 | 0.0201 | -1.1999 |

### Test Set Accuracy Comparison

All versions evaluated using the same hold-out methodology: tested on completely unseen 2023-2025 data (1,096 samples). v1 and v2 trained on 2015-2022; v3 trained on 1950-2022.

| Model | MAE High | MAE Low | RMSE High | RMSE Low |
|---|---|---|---|---|
| v1 | 4.61°F | 3.45°F | 6.00°F | 4.38°F |
| v2 | 4.53°F | 2.94°F | 5.81°F | 3.89°F |
| v3 | 3.94°F | 2.45°F | 5.24°F | 3.22°F |
| Baseline (same-day average) | 5.31°F | 3.83°F | 6.81°F | 4.94°F |

