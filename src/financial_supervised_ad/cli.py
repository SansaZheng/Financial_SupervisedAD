import argparse
from pathlib import Path

import pandas as pd

from .data import synthetic_data
from .experiment import run_experiment, sha256_file
from .models import METHODS


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ten supervised transaction fraud baselines")
    parser.add_argument("--csv", type=Path, help="Labeled CSV; otherwise use explicitly synthetic smoke data")
    parser.add_argument("--label", default="Class")
    parser.add_argument("--time-column", default="Time")
    parser.add_argument("--split", choices=["temporal", "stratified"], default=None)
    parser.add_argument("--drop-columns", nargs="*", default=[])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--balance", choices=["none", "balanced"], default="none")
    parser.add_argument("--threshold", choices=["f1", "fpr"], default="f1")
    parser.add_argument("--max-fpr", type=float, default=0.01)
    parser.add_argument("--quick", action="store_true", help="Reduced estimators/iterations for smoke tests")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.n_jobs < 1 or not 0 <= args.max_fpr <= 1:
        parser.error("n-jobs must be positive and max-fpr must lie in [0,1]")
    if args.csv:
        frame = pd.read_csv(args.csv)
        provenance = {"kind": "user_csv", "file": args.csv.name, "sha256": sha256_file(args.csv)}
        meta_path = Path(str(args.csv)+".json")
        if meta_path.exists():
            import json
            provenance["download"] = json.loads(meta_path.read_text())
        split = args.split or "temporal"
    else:
        if args.label != "Class":
            parser.error("Synthetic data uses label Class")
        frame = synthetic_data(args.n_samples, args.seed)
        provenance = {"kind": "synthetic_smoke_only", "generator": "sklearn.make_classification"}
        split = args.split or "stratified"
    run_experiment(frame, args.output, label=args.label, time_column=args.time_column, split=split,
        methods=args.methods, seed=args.seed, balance=args.balance, threshold=args.threshold,
        max_fpr=args.max_fpr, n_jobs=args.n_jobs, quick=args.quick, save_models=args.save_models,
        drop_columns=args.drop_columns, provenance=provenance)


if __name__ == "__main__":
    main()
