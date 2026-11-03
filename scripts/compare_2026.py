"""Compare classic and 2026 methods on identical held-out transaction endpoints.

Each method retains its own frozen validation threshold; there is no test tuning.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from financial_supervised_ad.experiment import write_json
from financial_supervised_ad.metrics import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classics", type=Path, required=True)
    parser.add_argument("--recent", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    records, configs = {}, []
    for directory in [args.classics, *args.recent]:
        config = json.loads((directory / "config.json").read_text())
        configs.append(config)
        for row in pd.read_csv(directory / "metrics.csv", float_precision="round_trip").to_dict("records"):
            method = row["method"]
            if method in records:
                raise ValueError(f"Duplicate method {method}")
            predictions = pd.read_csv(directory / f"{method}_predictions.csv.gz", float_precision="round_trip")
            test = predictions[predictions.split == "test"].set_index("row_id")
            if not test.index.is_unique:
                raise ValueError("Duplicate prediction row IDs")
            records[method] = (row, test)
    for config in configs[1:]:
        for key in ("frame_hash", "seed", "split", "features"):
            if config[key] != configs[0][key]:
                raise ValueError(f"Different source/protocol for {key}; cannot align by row_id")
    common = sorted(set.intersection(*(set(test.index) for _, test in records.values())))
    if not common:
        raise ValueError("No common test endpoints")
    reference = next(iter(records.values()))[1].loc[common, "label"]
    output = []
    for method, (row, test) in records.items():
        subset = test.loc[common]
        pd.testing.assert_series_equal(subset.label, reference)
        metrics = evaluate(subset.label, subset.score, row["threshold"])
        output.append({"method": method, **metrics, "test_rows": len(common),
                       "test_positives": int(subset.label.sum())})
    pd.DataFrame(output).to_csv(args.output / "metrics.csv", index=False)
    reference.reset_index().to_csv(args.output / "common_test_endpoints.csv", index=False)
    write_json(args.output / "alignment.json", dict(frame_hash=configs[0]["frame_hash"],
        rows=len(common), positives=int(reference.sum()), methods=list(records),
        thresholds="Frozen in each source run; no test tuning", passed=True))
    print(pd.DataFrame(output)[["method", "average_precision", "f1", "test_rows"]].to_string(index=False))


if __name__ == "__main__":
    main()
