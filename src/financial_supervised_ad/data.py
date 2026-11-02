"""Label validation, deterministic samples and independent holdouts."""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def synthetic_data(n_samples=5000, seed=42):
    if n_samples < 300:
        raise ValueError("Synthetic smoke data requires at least 300 rows")
    x, y = make_classification(n_samples=n_samples, n_features=20, n_informative=10,
        n_redundant=4, weights=[0.97, 0.03], flip_y=0.002, class_sep=1.3, random_state=seed)
    frame = pd.DataFrame(x, columns=[f"feature_{i}" for i in range(x.shape[1])])
    frame["Class"] = y
    return frame


def validate_frame(frame, label):
    if not frame.columns.is_unique:
        raise ValueError("Duplicate column names are not supported")
    if label not in frame:
        raise ValueError(f"Missing label column: {label}")
    if frame[label].isna().any() or set(frame[label].unique()) != {0, 1}:
        raise ValueError("Labels must contain both 0 (normal) and 1 (fraud), without missing values")
    return frame


def split_indices(frame, label="Class", mode="temporal", time_column="Time", seed=42):
    """60/20/20 split; temporal boundaries never divide equal timestamps."""
    validate_frame(frame, label)
    ids = np.arange(len(frame))
    if mode == "stratified":
        train, rest = train_test_split(ids, test_size=0.4, stratify=frame[label], random_state=seed)
        val, test = train_test_split(rest, test_size=0.5, stratify=frame.iloc[rest][label], random_state=seed)
    elif mode == "temporal":
        if time_column not in frame or frame[time_column].isna().any():
            raise ValueError("Temporal split requires a nonmissing time column")
        times = frame[time_column]
        if not pd.api.types.is_numeric_dtype(times):
            times = pd.to_datetime(times, utc=True, errors="raise")
        order = np.argsort(times.to_numpy(), kind="stable")
        ordered_times = times.iloc[order].to_numpy()
        # Every observation with a boundary timestamp goes into the later split.
        boundaries = [np.searchsorted(ordered_times, ordered_times[int(len(order)*q)], side="left")
                      for q in (0.6, 0.8)]
        train, val, test = np.split(order, boundaries)
    else:
        raise ValueError(f"Unknown split mode: {mode}")
    for name, group in zip(("train", "validation", "test"), (train, val, test)):
        if len(group) == 0 or frame.iloc[group][label].nunique() != 2:
            raise ValueError(f"{name} split lacks one class; use more data or a different split")
    return {"train": train, "validation": val, "test": test}


def prepare_features(frame, label, time_column, drop_columns=()):
    dropped = {label, time_column, *drop_columns}
    missing = set(drop_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Unknown columns requested for dropping: {sorted(missing)}")
    features = frame.drop(columns=[c for c in dropped if c in frame])
    if not features.shape[1]:
        raise ValueError("No features remain")
    # Infinity is missing, never a finite artificial sentinel learned from test data.
    features = features.replace([np.inf, -np.inf], np.nan)
    for col in features.select_dtypes(exclude="number"):
        features[col] = features[col].map(lambda v: str(v) if pd.notna(v) else np.nan)
    return features
