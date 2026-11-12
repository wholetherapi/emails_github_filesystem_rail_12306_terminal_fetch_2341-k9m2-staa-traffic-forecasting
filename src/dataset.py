"""Dataset loaders for the traffic forecasting experiments."""

import numpy as np
import pandas as pd


def load_series(path: str):
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    values = frame.values.astype("float32")
    return np.nan_to_num(values, nan=0.0)


def windowed(series, history=12, horizon=12):
    samples = []
    targets = []
    for start in range(len(series) - history - horizon):
        samples.append(series[start:start + history])
        targets.append(series[start + history:start + history + horizon])
    return np.stack(samples), np.stack(targets)
