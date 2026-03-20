import math
from pathlib import Path

import torch
import torch.nn as nn

from model import WeatherNet

MODEL_PATH     = Path(__file__).resolve().parent / "model_weights.pth"
TEST_SPLIT_PATH = Path(__file__).resolve().parent / "test_split.pt"
REPORT_PATH    = Path(__file__).resolve().parent / "training-report.md"


def mae_rmse(preds, actuals):
    n = len(preds)
    mae_tx  = sum(abs(p[0] - a[0]) for p, a in zip(preds, actuals)) / n
    mae_tn  = sum(abs(p[1] - a[1]) for p, a in zip(preds, actuals)) / n
    rmse_tx = math.sqrt(sum((p[0] - a[0]) ** 2 for p, a in zip(preds, actuals)) / n)
    rmse_tn = math.sqrt(sum((p[1] - a[1]) ** 2 for p, a in zip(preds, actuals)) / n)
    return mae_tx, mae_tn, rmse_tx, rmse_tn


def main():
    split = torch.load(TEST_SPLIT_PATH, map_location="cpu")
    X_test, y_test = split["X"], split["y"]

    model = WeatherNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        preds_tensor = model(X_test)

    preds   = [(float(r[0]), float(r[1])) for r in preds_tensor]
    actuals = [(float(r[0]), float(r[1])) for r in y_test]

    mae_tx, mae_tn, rmse_tx, rmse_tn = mae_rmse(preds, actuals)

    print(f"Test samples : {len(X_test)}")
    print(f"MAE  tmax    : {mae_tx:.4f} F")
    print(f"MAE  tmin    : {mae_tn:.4f} F")
    print(f"RMSE tmax    : {rmse_tx:.4f} F")
    print(f"RMSE tmin    : {rmse_tn:.4f} F")

    test_section = f"""
## Test Set Results

| Metric | Value |
|---|---|
| Test samples | {len(X_test)} |
| MAE tmax | {mae_tx:.4f} F |
| MAE tmin | {mae_tn:.4f} F |
| RMSE tmax | {rmse_tx:.4f} F |
| RMSE tmin | {rmse_tn:.4f} F |
"""

    report = REPORT_PATH.read_text()
    REPORT_PATH.write_text(report + test_section)
    print(f"Results appended to {REPORT_PATH}")


if __name__ == "__main__":
    main()
