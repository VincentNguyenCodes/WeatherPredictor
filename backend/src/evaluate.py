"""
Accuracy evaluation for the WeatherNet model.

Strategy: hold out the last 3 years (2023, 2024, 2025) as the test set.
Train a fresh model on 2015-2022 only, then evaluate on each held-out year.
Also evaluates the current saved model (trained on all years) for comparison.
A naive "historical average" baseline is included for reference.

Run: python src/evaluate.py
"""

import csv, math, sys
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR   = Path(__file__).resolve().parents[1] / "data"
MODEL_PATH = Path(__file__).resolve().parents[1] / "backend/weather/ml/model_weights.pth"

HIST_YEARS = 5
SEQ_DAYS   = 3
TEST_YEARS = [2023, 2024, 2025]
TRAIN_YEARS = [y for y in range(2015, 2026) if y not in TEST_YEARS]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all():
    """Return {year: {doy: (tmax, tmin)}}"""
    all_data = {}
    for f in sorted(DATA_DIR.glob("SanJoseWeather*.csv")):
        with open(f, newline="") as fp:
            for row in csv.DictReader(fp):
                yr  = int(row["year"])
                doy = int(row["day_of_year"])
                tx, tn = row["tmax"], row["tmin"]
                if tx == "" or tn == "":
                    continue
                all_data.setdefault(yr, {})[doy] = (float(tx), float(tn))
    return all_data


# ── Feature builder ───────────────────────────────────────────────────────────

def build_features(same_day_hist, recent_seq, doy):
    hist_tx = [0.] * HIST_YEARS
    hist_tn = [0.] * HIST_YEARS
    flags   = [0.] * HIST_YEARS
    for i, (tx, tn) in enumerate(same_day_hist[-HIST_YEARS:]):
        slot = HIST_YEARS - len(same_day_hist) + i
        hist_tx[slot] = tx; hist_tn[slot] = tn; flags[slot] = 1.
    seq_tx = [0.] * SEQ_DAYS; seq_tn = [0.] * SEQ_DAYS
    for i, (tx, tn) in enumerate(recent_seq[:SEQ_DAYS]):
        seq_tx[i] = tx; seq_tn[i] = tn
    angle = 2 * math.pi * doy / 365.
    vec = (
        [v for p in zip(hist_tx, hist_tn) for v in p]
        + flags
        + [v for p in zip(seq_tx, seq_tn) for v in p]
        + [math.sin(angle), math.cos(angle)]
    )
    return vec


def make_dataset(all_data, use_years, target_years):
    """Build (X, y, meta) for given target years, using use_years as history."""
    features, targets, meta = [], [], []
    for target_year in target_years:
        if target_year not in all_data:
            continue
        past_years = [y for y in sorted(use_years) if y < target_year]
        for doy in sorted(all_data[target_year]):
            tmax_t, tmin_t = all_data[target_year][doy]
            same_day = [
                all_data[py][doy]
                for py in past_years[-HIST_YEARS:]
                if doy in all_data.get(py, {})
            ]
            if not same_day:
                continue
            recent = [
                all_data[target_year][doy - off]
                for off in range(1, SEQ_DAYS + 1)
                if (doy - off) >= 1 and (doy - off) in all_data[target_year]
            ]
            features.append(build_features(same_day, recent, doy))
            targets.append([tmax_t, tmin_t])
            meta.append((target_year, doy, same_day))
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(targets,  dtype=torch.float32)
    return X, y, meta


# ── Model ─────────────────────────────────────────────────────────────────────

def make_model():
    return nn.Sequential(
        nn.Linear(23, 128), nn.ReLU(),
        nn.Linear(128, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 2),
    )


def train_model(X, y, epochs=1000, lr=1e-3, verbose=False):
    model  = make_model()
    opt    = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
    loss_fn = nn.MSELoss()
    for epoch in range(1, epochs + 1):
        total = 0.
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        if verbose and epoch % 200 == 0:
            print(f"    epoch {epoch:4d}  loss={total/len(loader):.4f}")
    model.eval()
    return model


# ── Metrics ───────────────────────────────────────────────────────────────────

def metrics(preds, actuals):
    """Returns (mae_tmax, mae_tmin, rmse_tmax, rmse_tmin)"""
    n = len(preds)
    mae_tx = sum(abs(p[0] - a[0]) for p, a in zip(preds, actuals)) / n
    mae_tn = sum(abs(p[1] - a[1]) for p, a in zip(preds, actuals)) / n
    rmse_tx = math.sqrt(sum((p[0]-a[0])**2 for p,a in zip(preds,actuals)) / n)
    rmse_tn = math.sqrt(sum((p[1]-a[1])**2 for p,a in zip(preds,actuals)) / n)
    return mae_tx, mae_tn, rmse_tx, rmse_tn


def predict_batch(model, X):
    with torch.no_grad():
        out = model(X)
    return [(round(r[0].item(), 1), round(r[1].item(), 1)) for r in out]


# ── Baseline: historical same-day average ─────────────────────────────────────

def baseline_preds(meta):
    """Average of past same-day temps (no neural net)."""
    preds = []
    for (yr, doy, same_day) in meta:
        avg_tx = sum(tx for tx, tn in same_day) / len(same_day)
        avg_tn = sum(tn for tx, tn in same_day) / len(same_day)
        preds.append((avg_tx, avg_tn))
    return preds


# ── Per-year breakdown ────────────────────────────────────────────────────────

def per_year_metrics(preds, actuals, meta):
    by_year = defaultdict(lambda: {"preds": [], "actuals": []})
    for pred, actual, (yr, doy, _) in zip(preds, actuals, meta):
        by_year[yr]["preds"].append(pred)
        by_year[yr]["actuals"].append(actual)
    results = {}
    for yr in sorted(by_year):
        results[yr] = metrics(by_year[yr]["preds"], by_year[yr]["actuals"])
    return results


# ── Worst predictions ─────────────────────────────────────────────────────────

def worst_predictions(preds, actuals, meta, n=10):
    errors = []
    for pred, actual, (yr, doy, _) in zip(preds, actuals, meta):
        err = abs(pred[0] - actual[0]) + abs(pred[1] - actual[1])
        errors.append((err, yr, doy, pred, actual))
    return sorted(errors, reverse=True)[:n]


# ── Main ──────────────────────────────────────────────────────────────────────

def divider(char="─", width=64):
    print(char * width)

def section(title):
    divider()
    print(f"  {title}")
    divider()

def main():
    all_data = load_all()
    years_available = sorted(all_data.keys())
    print(f"\n  Data loaded: {years_available[0]}–{years_available[-1]}")
    print(f"  Train years: {TRAIN_YEARS}")
    print(f"  Test years:  {TEST_YEARS}\n")

    # ── 1. Build train and test sets ──────────────────────────────────────
    all_years = list(range(2015, 2026))
    X_train, y_train, _     = make_dataset(all_data, all_years, TRAIN_YEARS)
    X_test,  y_test,  meta  = make_dataset(all_data, all_years, TEST_YEARS)
    actuals = [(float(y[0]), float(y[1])) for y in y_test]

    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples:     {len(X_test)}\n")

    # ── 2. Train a fresh hold-out model ──────────────────────────────────
    section("Training hold-out model (2015–2022 only)")
    holdout_model = train_model(X_train, y_train, epochs=1000, verbose=True)

    # ── 3. Load the production model (trained on all years) ───────────────
    prod_model = make_model()
    prod_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    prod_model.eval()

    # ── 4. Predictions ────────────────────────────────────────────────────
    preds_holdout = predict_batch(holdout_model, X_test)
    preds_prod    = predict_batch(prod_model,    X_test)
    preds_base    = baseline_preds(meta)

    # ── 5. Overall metrics ────────────────────────────────────────────────
    section("Overall accuracy on 2023–2025 test set")
    header = f"  {'Model':<28}  {'MAE High':>8}  {'MAE Low':>8}  {'RMSE High':>10}  {'RMSE Low':>9}"
    print(header)
    divider("-")

    for label, preds in [
        ("Hold-out model (2015–2022 train)", preds_holdout),
        ("Production model (all years)",      preds_prod),
        ("Baseline (same-day avg, no NN)",    preds_base),
    ]:
        mx, mn, rx, rn = metrics(preds, actuals)
        print(f"  {label:<28}  {mx:>7.2f}°  {mn:>7.2f}°  {rx:>9.2f}°  {rn:>8.2f}°")

    # ── 6. Per-year breakdown (hold-out model) ────────────────────────────
    section("Per-year breakdown — hold-out model")
    print(f"  {'Year':<6}  {'Samples':>7}  {'MAE High':>9}  {'MAE Low':>9}  {'RMSE High':>10}  {'RMSE Low':>9}")
    divider("-")
    by_year = per_year_metrics(preds_holdout, actuals, meta)
    by_year_counts = defaultdict(int)
    for (yr, doy, _) in meta:
        by_year_counts[yr] += 1
    for yr, (mx, mn, rx, rn) in by_year.items():
        print(f"  {yr:<6}  {by_year_counts[yr]:>7}  {mx:>8.2f}°  {mn:>8.2f}°  {rx:>9.2f}°  {rn:>8.2f}°")

    # ── 7. Worst individual predictions ───────────────────────────────────
    section("10 largest errors — hold-out model (tmax + tmin combined)")
    print(f"  {'Year':>4}  {'DOY':>4}  {'Pred High':>9}  {'Act High':>8}  {'Pred Low':>9}  {'Act Low':>8}  {'Total Err':>9}")
    divider("-")
    from datetime import date, timedelta
    for err, yr, doy, pred, actual in worst_predictions(preds_holdout, actuals, meta):
        d = date(yr, 1, 1) + timedelta(days=doy - 1)
        date_str = d.strftime("%b %d")
        print(
            f"  {yr:>4}  {date_str:>6}  "
            f"{pred[0]:>8.1f}°  {actual[0]:>7.1f}°  "
            f"{pred[1]:>8.1f}°  {actual[1]:>7.1f}°  "
            f"{err:>8.1f}°"
        )

    # ── 8. Monthly MAE breakdown ──────────────────────────────────────────
    section("Monthly MAE — hold-out model (averaged over 2023–2025)")
    from datetime import date as dt, timedelta as td
    MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    by_month = defaultdict(lambda: {"preds": [], "actuals": []})
    for pred, actual, (yr, doy, _) in zip(preds_holdout, actuals, meta):
        d = dt(yr, 1, 1) + td(days=doy - 1)
        by_month[d.month]["preds"].append(pred)
        by_month[d.month]["actuals"].append(actual)
    print(f"  {'Month':<6}  {'Samples':>7}  {'MAE High':>9}  {'MAE Low':>9}")
    divider("-")
    for m in range(1, 13):
        if m not in by_month:
            continue
        mx, mn, _, _ = metrics(by_month[m]["preds"], by_month[m]["actuals"])
        n = len(by_month[m]["preds"])
        print(f"  {MONTHS[m-1]:<6}  {n:>7}  {mx:>8.2f}°  {mn:>8.2f}°")

    divider()
    print()


if __name__ == "__main__":
    main()
