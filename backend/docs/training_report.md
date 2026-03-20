# WeatherNet Training Report

## Improvements Made

| Change | Detail |
|---|---|
| More data | Extended dataset from 2015-2026 (11 years) to 1950-2026 (76 years) via Open-Meteo |
| Early stopping | Stops training when cross validation loss stops improving (patience=50) |
| Weight decay | Added L2 regularization to Adam optimizer (weight_decay=1e-4) |
| Recency weighting | Sequential prior day features weighted linearly by closeness to target date |

---

## Before Improvements

**Dataset:** 2015-2026 (3,729 total samples)

| Split | Samples |
|---|---|
| Train (60%) | 2237 |
| Cross Validation (20%) | 745 |
| Test (20%) | 747 |

**Training Configuration**

| Parameter | Value |
|---|---|
| Max epochs | 1000 |
| Stopped at epoch | 1000 (no early stopping) |
| Learning rate | 0.001 |
| Batch size | 64 |
| Loss function | HuberLoss |
| Optimizer | Adam (no weight decay) |

**Loss History**

| Epoch | Train Loss | Cross Validation Loss |
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

**Final Results**

| Metric | Value |
|---|---|
| Final train loss | 1.8390 |
| Final cross validation loss | 3.0590 |
| Train/CV gap | 1.2200 (overfitting) |

---

## After Improvements

**Dataset:** 1950-2026 (27,470 total samples)

| Split | Samples |
|---|---|
| Train (60%) | 16482 |
| Cross Validation (20%) | 5494 |
| Test (20%) | 5494 |

**Training Configuration**

| Parameter | Value |
|---|---|
| Max epochs | 1000 |
| Stopped at epoch | 222 |
| Early stopping patience | 50 |
| Learning rate | 0.001 |
| Batch size | 64 |
| Loss function | HuberLoss |
| Optimizer | Adam |
| Weight decay | 1e-4 |

**Loss History**

| Epoch | Train Loss | Cross Validation Loss |
|---|---|---|
| 100 | 2.6616 | 2.6756 |
| 200 | 2.6400 | 2.6075 |
| 222 | — | 2.6075 (best, early stop) |

**Final Results**

| Metric | Value |
|---|---|
| Final train loss | 2.6400 |
| Final cross validation loss | 2.6075 |
| Train/CV gap | 0.0325 (overfitting resolved) |

---

## Summary

| Metric | Before | After | Change |
|---|---|---|---|
| Dataset size | 3,729 | 27,470 | +636% |
| Epochs trained | 1000 | 222 | -778 |
| Final train loss | 1.8390 | 2.6400 | higher (less memorization) |
| Final CV loss | 3.0590 | 2.6075 | -0.4515 |
| Train/CV gap | 1.2200 | 0.0325 | -1.1875 |

The train/CV gap dropped from 1.22 to 0.03, meaning the model is no longer memorizing the training data. CV loss improved by 0.45 despite training for far fewer epochs.

Test set evaluation has not been run yet.

## Test Set Results

| Metric | Value |
|---|---|
| Test samples | 5494 |
| MAE tmax | 3.6525 F |
| MAE tmin | 2.3914 F |
| RMSE tmax | 4.7170 F |
| RMSE tmin | 3.0857 F |
