import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import WeatherNet, build_features, HIST_YEARS, SEQ_DAYS

DEFAULT_DATA_DIR   = Path(__file__).resolve().parents[2] / "data"
DEFAULT_MODEL_OUT  = Path(__file__).resolve().parent / "model_weights.pth"
DEFAULT_REPORT_OUT = Path(__file__).resolve().parents[3] / "docs" / "accuracy-report.md"
TEST_SPLIT_OUT     = Path(__file__).resolve().parent / "test_split.pt"


def load_data(data_dir: Path) -> dict:
    all_data: dict = {}
    for csv_file in sorted(data_dir.glob("SanJoseWeather*.csv")):
        with open(csv_file, newline="") as f:
            for row in csv.DictReader(f):
                yr  = int(row["year"])
                doy = int(row["day_of_year"])
                tx, tn = row["tmax"], row["tmin"]
                if tx == "" or tn == "":
                    continue
                precip = float(row["precip"]) if row.get("precip", "") != "" else 0.0
                all_data.setdefault(yr, {})[doy] = (float(tx), float(tn), precip)
    return all_data


def build_dataset(all_data: dict):
    years = sorted(all_data.keys())
    features, targets, sample_years = [], [], []

    for y_idx, target_year in enumerate(years):
        if y_idx == 0:
            continue
        past_years = years[:y_idx]

        for doy, (tmax_t, tmin_t, _) in sorted(all_data[target_year].items()):
            same_day = [
                (all_data[py][doy][0], all_data[py][doy][1])
                for py in past_years[-HIST_YEARS:]
                if doy in all_data.get(py, {})
            ]
            if not same_day:
                continue

            recent = []
            precip_seq = []
            for off in range(1, SEQ_DAYS + 1):
                prev_doy = doy - off
                if prev_doy >= 1 and prev_doy in all_data[target_year]:
                    tx, tn, pr = all_data[target_year][prev_doy]
                    recent.append((tx, tn))
                    precip_seq.append(pr)

            features.append(build_features(same_day, recent, doy, precip_seq).tolist())
            targets.append([tmax_t, tmin_t])
            sample_years.append(target_year)

    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(targets,  dtype=torch.float32)
    return X, y, sample_years


def split_data(X, y, sample_years, train_end=1995, val_end=2010):
    years_t = torch.tensor(sample_years)
    train_idx = (years_t <= train_end).nonzero(as_tuple=True)[0]
    val_idx   = ((years_t > train_end) & (years_t <= val_end)).nonzero(as_tuple=True)[0]
    test_idx  = (years_t > val_end).nonzero(as_tuple=True)[0]
    return (
        X[train_idx], y[train_idx],
        X[val_idx],   y[val_idx],
        X[test_idx],  y[test_idx],
    )


def train(data_dir: Path, output_path: Path, report_path: Path, epochs: int, lr: float, patience: int = 50):
    print(f"Loading data from {data_dir} ...")
    all_data = load_data(data_dir)
    years = sorted(all_data.keys())
    print(f"  Years: {years[0]}-{years[-1]}")

    X, y, sample_years = build_dataset(all_data)
    print(f"  Total samples: {len(X)}")

    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, sample_years)
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    torch.save({'X': X_test, 'y': y_test}, TEST_SPLIT_OUT)
    print(f"  Test split saved -> {TEST_SPLIT_OUT}")

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val),     batch_size=64, shuffle=False)

    model   = WeatherNet()
    opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.HuberLoss()

    history       = []
    best_val_loss = float('inf')
    best_weights  = None
    epochs_no_improve = 0
    stopped_epoch = epochs

    print(f"Training for up to {epochs} epochs (early stopping patience={patience})...")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += loss_fn(model(xb), yb).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights  = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 100 == 0:
            history.append((epoch, train_loss, val_loss))
            print(f"  Epoch {epoch:5d}  train={train_loss:.4f}  val={val_loss:.4f}")

        if epochs_no_improve >= patience:
            stopped_epoch = epoch
            print(f"  Early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(best_weights)
    torch.save(model.state_dict(), output_path)
    print(f"Saved best model weights -> {output_path}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write("# WeatherNet Training Report\n\n")

        f.write("## Data Split\n\n")
        f.write("| Split | Years | Samples |\n|---|---|---|\n")
        f.write(f"| Train | 1950-1995 | {len(X_train)} |\n")
        f.write(f"| Validation | 1996-2010 | {len(X_val)} |\n")
        f.write(f"| Test | 2011-2026 | {len(X_test)} |\n\n")

        f.write("## Training Configuration\n\n")
        f.write("| Parameter | Value |\n|---|---|\n")
        f.write(f"| Max epochs | {epochs} |\n")
        f.write(f"| Stopped at epoch | {stopped_epoch} |\n")
        f.write(f"| Early stopping patience | {patience} |\n")
        f.write(f"| Learning rate | {lr} |\n")
        f.write("| Batch size | 64 |\n")
        f.write("| Loss function | HuberLoss |\n")
        f.write("| Optimizer | Adam |\n")
        f.write("| Weight decay | 1e-4 |\n\n")

        f.write("## Loss History\n\n")
        f.write("| Epoch | Train Loss | Val Loss |\n|---|---|---|\n")
        for ep, tl, vl in history:
            f.write(f"| {ep} | {tl:.4f} | {vl:.4f} |\n")

        final_train = history[-1][1]
        final_val   = history[-1][2]
        f.write("\n## Final Results\n\n")
        f.write("| Metric | Value |\n|---|---|\n")
        f.write(f"| Final train loss | {final_train:.4f} |\n")
        f.write(f"| Final val loss | {final_val:.4f} |\n")

    print(f"Report saved -> {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output",   type=Path, default=DEFAULT_MODEL_OUT)
    parser.add_argument("--report",   type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--epochs",   type=int,  default=1000)
    parser.add_argument("--patience", type=int,  default=50)
    parser.add_argument("--lr",       type=float, default=1e-3)
    args = parser.parse_args()
    train(args.data_dir, args.output, args.report, args.epochs, args.lr, args.patience)
