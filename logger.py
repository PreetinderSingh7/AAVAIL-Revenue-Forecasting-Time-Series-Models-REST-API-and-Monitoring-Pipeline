"""CSV-style logging for training and prediction. test=True writes to logs/test (isolated from production)."""
import csv
import os
import time

BASE = os.path.dirname(os.path.abspath(__file__))
TRAIN_HEADER = ["timestamp", "country", "version", "model", "rmse", "baseline_rmse", "runtime"]
PREDICT_HEADER = ["timestamp", "country", "version", "target_date", "y_pred", "baseline", "runtime"]


def log_dir(test=False):
    d = os.path.join(BASE, "logs", "test") if test else os.path.join(BASE, "logs")
    os.makedirs(d, exist_ok=True)
    return d


def log_path(kind, version, test=False):
    return os.path.join(log_dir(test), f"{kind}-v{version}.log")


def _write(path, header, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def update_train_log(country, version, model, rmse, baseline_rmse, runtime, test=False):
    _write(log_path("train", version, test), TRAIN_HEADER,
           [time.time(), country, version, model, round(rmse, 4), round(baseline_rmse, 4), round(runtime, 4)])


def update_predict_log(country, version, target_date, y_pred, baseline, runtime, test=False):
    _write(log_path("predict", version, test), PREDICT_HEADER,
           [time.time(), country, version, target_date, round(y_pred, 4), round(baseline, 4), round(runtime, 4)])


def read_log(kind, version, test=False):
    path = log_path(kind, version, test)
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))
