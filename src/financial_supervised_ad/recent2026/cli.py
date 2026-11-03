"""Separate entry point keeps optional research dependencies out of classic runs."""
import argparse
import hashlib
import importlib.metadata
import os
import platform
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ..data import prepare_features, split_indices, synthetic_data, validate_frame
from ..experiment import sha256_file, write_json
from ..metrics import evaluate, select_threshold
from . import METHODS_2026


def run(args):
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {out}")
    if args.n_jobs < 1:
        raise ValueError("n_jobs must be positive")
    if args.csv:
        frame = pd.read_csv(args.csv)
        source = dict(kind="user_csv", file=Path(args.csv).name, sha256=sha256_file(args.csv))
        split = args.split or "temporal"
    else:
        frame = synthetic_data(args.n_samples, args.seed)
        frame["Time"] = np.arange(len(frame))
        source = dict(kind="synthetic_smoke_only")
        split = args.split or "temporal"
    if split != "temporal" and args.method == "sttn_cp_2026":
        raise ValueError("STTN-CP requires a temporal split to preserve sequence order")
    validate_frame(frame, "Class")
    frame = frame.drop_duplicates().reset_index(drop=True)
    groups = split_indices(frame, "Class", split, "Time", args.seed)
    features = prepare_features(frame, "Class", "Time")
    if len(features.select_dtypes(exclude="number").columns):
        raise ValueError("The 2026 ULB reproductions accept numeric features only")
    X, y = features.to_numpy(dtype=float), frame.Class.to_numpy(dtype=np.int64)
    out.mkdir(parents=True, exist_ok=True)
    source_commit = os.environ.get("FINANCIAL_AD_COMMIT")
    if source_commit is None:
        try:
            source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                stderr=subprocess.DEVNULL, timeout=3,
                env={**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1"}).strip()
        except (OSError, subprocess.SubprocessError):
            source_commit = None
    cfg = dict(methods=[args.method], seed=args.seed, quick=args.quick, n_jobs=args.n_jobs,
        split=split, time_column="Time", label="Class", threshold_objective="f1",
        source_commit=source_commit, provenance=source, features=list(features),
        frame_hash=hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest(),
        python=platform.python_version(), versions={key: importlib.metadata.version(key) for key in
            ("numpy", "pandas", "scikit-learn", "torch", "interpret-core")})
    start = time.perf_counter()
    with threadpool_limits(limits=args.n_jobs):
        if args.method == "optimized_ebm_2026":
            from .ebm import OptimizedEBM2026
            model = OptimizedEBM2026(seed=args.seed, n_jobs=args.n_jobs, quick=args.quick,
                cv_folds=2 if args.quick else 3, progress_path=str(out / "progress.txt"))
            model.fit(X[groups["train"]], y[groups["train"]])
            scores = {name: model.predict_proba(X[ids])[:, 1] for name, ids in groups.items() if name != "train"}
            eligible = {name: (X[ids], y[ids], ids) for name, ids in groups.items()}
            info = dict(parameters=model.get_params(), best_parameters=model.best_parameters_,
                scaler_order=model.scaler_order_, search_levels=model.levels_,
                selected_features=[features.columns[i] for i in model.selected_features_],
                feature_importances=model.feature_importances_.tolist())
            write_json(out / "search_history.json", model.history_)
            joblib.dump(model, out / "model.joblib")
        else:
            from .sttn import train_sttn
            scores, eligible, info = train_sttn(X, y, groups, out, seed=args.seed,
                n_jobs=args.n_jobs, quick=args.quick, device=args.device, max_epochs=args.max_epochs)
    cfg["splits"] = {name: dict(rows=len(data[1]), positives=int(data[1].sum()),
        source_partition_rows=len(groups[name])) for name, data in eligible.items()}
    write_json(out / "config.json", cfg)
    manifest = pd.DataFrame({"row_id": np.arange(len(frame)), "label": y, "Time": frame.Time})
    manifest["split"] = ""
    manifest["eligible_endpoint"] = False
    for name, ids in groups.items():
        manifest.loc[ids, "split"] = name
        manifest.loc[eligible[name][2], "eligible_endpoint"] = True
    manifest.to_csv(out / "splits.csv", index=False)
    chosen = select_threshold(eligible["validation"][1], scores["validation"])
    metrics = {"method": args.method, **evaluate(eligible["test"][1], scores["test"], chosen),
               "elapsed_seconds": time.perf_counter()-start}
    pd.DataFrame([metrics]).to_csv(out / "metrics.csv", index=False)
    predictions = pd.concat([pd.DataFrame({"row_id": eligible[name][2], "split": name,
        "label": eligible[name][1], "score": scores[name]}) for name in ("validation", "test")], ignore_index=True)
    predictions["prediction"] = (predictions.score >= chosen).astype(int)
    predictions.to_csv(out / f"{args.method}_predictions.csv.gz", index=False)
    write_json(out / f"{args.method}.json", {**metrics, "training": info})
    write_json(out / "summary.json", dict(completed=1, method=args.method, metrics=metrics))
    # Threshold and schema are explicit deployment artifacts, separate from weights.
    write_json(out / "inference.json", dict(method=args.method, features=list(features),
        threshold=chosen, sequence_length=info.get("model_config", {}).get("seq_len", 1)))
    print(f"Completed {args.method}: AP={metrics['average_precision']:.4f}, F1={metrics['f1']:.4f}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS_2026, required=True)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--split", choices=["temporal", "stratified"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-epochs", type=int, help="Override STTN epoch limit; default 50 (quick: 2)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
