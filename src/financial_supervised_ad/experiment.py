"""Shared leakage-aware experiment runner and auditable artifacts."""
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from .data import prepare_features, split_indices, validate_frame
from .metrics import evaluate, select_threshold
from .models import METHODS, build_model


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, obj):
    # Estimator parameters legitimately include np.nan (e.g. imputer missing_values).
    # Preserve these as named strings while keeping metric JSON strictly standard.
    def sanitize(value):
        if isinstance(value, dict):
            return {str(key): sanitize(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [sanitize(item) for item in value]
        if isinstance(value, (float, np.floating)) and not math.isfinite(value):
            return str(value)
        return value
    Path(path).write_text(json.dumps(sanitize(obj), indent=2, ensure_ascii=False, default=str,
                                    allow_nan=False) + "\n", encoding="utf-8")


def preprocessing(features):
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in features if c not in numeric]
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                               ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ], remainder="drop")


def score_model(estimator, x):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(x)[:, 1]
    return estimator.decision_function(x)


def run_experiment(frame, output, *, label="Class", time_column="Time", split="temporal",
                   methods=METHODS, seed=42, balance="none", threshold="f1", max_fpr=0.01,
                   n_jobs=1, quick=False, save_models=False, drop_columns=(), provenance=None):
    validate_frame(frame, label)
    if balance not in {"none", "balanced"}:
        raise ValueError("balance must be none or balanced")
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(METHODS):
        raise ValueError("Specify nonempty, unique, known methods")
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {out}. Use a new folder to preserve results.")
    out.mkdir(parents=True, exist_ok=True)
    # Remove repeated observations before splitting, and record how many were removed.
    # Conflicting labels for identical predictors are rejected rather than silently resolved.
    raw_rows = len(frame)
    predictors = [c for c in frame if c != label]
    if frame.groupby(predictors, dropna=False, observed=True)[label].nunique().gt(1).any():
        raise ValueError("Identical observations have conflicting labels")
    frame = frame.drop_duplicates().reset_index(drop=True)
    groups = split_indices(frame, label, split, time_column, seed)
    features = prepare_features(frame, label, time_column, drop_columns)
    y = frame[label].to_numpy(dtype=int)
    manifest = pd.DataFrame({"row_id": np.arange(len(frame)), "label": y})
    manifest["split"] = ""
    for name, idx in groups.items():
        manifest.loc[idx, "split"] = name
    if time_column in frame:
        manifest[time_column] = frame[time_column]
    manifest.to_csv(out / "splits.csv", index=False)
    frame_hash = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
            timeout=3, text=True, env={**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null",
                                      "GIT_CONFIG_NOSYSTEM": "1"}).strip()
    except (subprocess.SubprocessError, OSError):
        git_sha = None
    info = dict(label=label, time_column=time_column, split=split, seed=seed, balance=balance,
                threshold_objective=threshold, max_fpr=max_fpr, quick=quick, methods=list(methods),
                n_jobs=n_jobs, drop_columns=list(drop_columns), features=list(features.columns),
                raw_rows=raw_rows, deduplicated_rows=len(frame), frame_hash=frame_hash,
                git_commit=git_sha, python=platform.python_version(), platform=platform.platform(),
                versions={p: importlib.metadata.version(p) for p in
                          ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "joblib", "threadpoolctl")},
                splits={name: {"rows": len(idx), "positives": int(y[idx].sum())} for name, idx in groups.items()},
                provenance=provenance or {})
    write_json(out / "config.json", info)
    train, val, test = (groups[n] for n in ("train", "validation", "test"))
    results = []
    started = time.perf_counter()
    with threadpool_limits(limits=n_jobs):
        for i, name in enumerate(methods):
            model = build_model(name, seed=seed, n_jobs=n_jobs, quick=quick)
            pipeline = Pipeline([("preprocessing", preprocessing(features)), ("model", model)])
            fit_args = {}
            # sklearn kNN has no sample_weight API; its distance weighting is unrelated
            # to class balancing. Leave it unchanged and record this exception.
            weighted = balance == "balanced" and name != "knn"
            if weighted:
                fit_args["model__sample_weight"] = compute_sample_weight("balanced", y[train])
            start = time.perf_counter()
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                pipeline.fit(features.iloc[train], y[train], **fit_args)
            fit_seconds = time.perf_counter()-start
            start = time.perf_counter()
            val_scores = score_model(pipeline, features.iloc[val])
            chosen = select_threshold(y[val], val_scores, objective=threshold, max_fpr=max_fpr)
            test_scores = score_model(pipeline, features.iloc[test])
            if not np.isfinite(test_scores).all():
                raise ValueError(f"{name} emitted non-finite scores")
            row = {"method": name, **evaluate(y[test], test_scores, chosen),
                   "validation_f1": evaluate(y[val], val_scores, chosen)["f1"],
                   "validation_fpr": evaluate(y[val], val_scores, chosen)["fpr"],
                   "fit_seconds": fit_seconds, "score_seconds": time.perf_counter()-start,
                   "class_weighted": weighted,
                   "warnings": sorted({str(w.message) for w in captured})}
            results.append(row)
            write_json(out / f"{name}.json", {**row, "parameters": pipeline.get_params(deep=True)})
            predictions = pd.DataFrame({"row_id": np.r_[val, test], "split":
                ["validation"]*len(val) + ["test"]*len(test), "label": y[np.r_[val, test]],
                "score": np.r_[val_scores, test_scores]})
            predictions["prediction"] = (predictions["score"] >= chosen).astype(int)
            predictions.to_csv(out / f"{name}_predictions.csv.gz", index=False, compression="gzip")
            if save_models:
                joblib.dump({"pipeline": pipeline, "threshold": chosen,
                             "features": list(features.columns), "method": name}, out / f"{name}.joblib")
            pd.DataFrame(results).to_csv(out / "metrics.csv", index=False)
            (out / "progress.txt").write_text(f"{i+1}/{len(methods)} {name}\n", encoding="utf-8")
            print(f"{i+1}/{len(methods)} {name}: AP={row['average_precision']:.4f} F1={row['f1']:.4f}", flush=True)
    write_json(out / "summary.json", {"completed": len(results), "elapsed_seconds": time.perf_counter()-started,
        "best_test_ap_descriptive_only": max(results, key=lambda r: r["average_precision"])["method"],
        "note": "Test ranking is descriptive, not a model-selection procedure."})
    return pd.DataFrame(results)
