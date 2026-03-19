# WeatherNet Training Report

## Overview

WeatherNet is a 4-layer MLP (40 inputs → 128 → 256 → 128 → 64 → 2 outputs) trained to predict daily high and low temperatures for San Jose, CA using 11 years of NOAA historical data (2015-2026).

This report covers training and validation loss only. The held-out test set (20%) has not been evaluated.

## Data Split

3,729 total samples, randomly split with seed 42.

| Split | Samples | Fraction |
|---|---|---|
| Train | 2237 | 60% |
| Validation | 745 | 20% |
| Test | 747 | 20% |

## Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 1000 |
| Learning rate | 0.001 |
| Batch size | 64 |
| Loss function | HuberLoss |
| Optimizer | Adam |

## Loss History

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 100 | 2.6265 | 2.8541 |
| 200 | 2.3824 | 2.7113 |
| 300 | 2.4375 | 2.6857 |
| 400 | 2.4017 | 2.7399 |
| 500 | 2.1979 | 2.7589 |
| 600 | 2.2036 | 2.8409 |
| 700 | 2.0559 | 2.8138 |
| 800 | 1.9963 | 2.8608 |
| 900 | 1.8950 | 2.9314 |
| 1000 | 1.8390 | 3.0590 |

## Final Results

| Metric | Value |
|---|---|
| Final train loss | 1.8390 |
| Final val loss | 3.0590 |

## Observations

Val loss bottoms out around epoch 300 (2.6857) while train loss continues falling to 1.8390 by epoch 1000. The gap of ~1.22 between final train and val loss is a sign of overfitting. The model would likely generalize better if stopped around epoch 300 rather than running the full 1000 epochs.

Potential next steps: add early stopping based on val loss, add dropout to the architecture, or reduce model capacity.
