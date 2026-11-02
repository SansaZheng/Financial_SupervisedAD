"""Independently recompute reported test metrics from exported prediction files."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, auc, confusion_matrix, f1_score,
    matthews_corrcoef, precision_recall_curve, precision_score, recall_score, roc_auc_score)


def audit(directory):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    results = pd.read_csv(directory / "metrics.csv", float_precision="round_trip")
    assert len(results) == len(config["methods"])
    for row in results.to_dict("records"):
        saved = pd.read_csv(directory / f"{row['method']}_predictions.csv.gz", float_precision="round_trip")
        val, test = (saved[saved.split == name] for name in ("validation", "test"))
        assert set(val.row_id).isdisjoint(test.row_id)
        assert len(test) == config["splits"]["test"]["rows"]
        pred = (test.score.to_numpy() >= row["threshold"]).astype(int)
        np.testing.assert_array_equal(test.prediction, pred)
        y, scores = test.label, test.score
        p, r, _ = precision_recall_curve(y, scores)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        expected = dict(average_precision=average_precision_score(y, scores),
            pr_auc_trapezoid=auc(r, p), roc_auc=roc_auc_score(y, scores),
            precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred),
            f1=f1_score(y, pred), mcc=matthews_corrcoef(y, pred), fpr=fp/(tn+fp),
            tn=tn, fp=fp, fn=fn, tp=tp)
        for key, value in expected.items():
            np.testing.assert_allclose(row[key], value, atol=1e-12, rtol=1e-10,
                                       err_msg=f"{row['method']}: {key}")
        if config["threshold_objective"] == "f1":
            # Independent sklearn PR-curve implementation of the optimum.
            vp, vr, thresholds = precision_recall_curve(val.label, val.score)
            utility = np.divide(2*vp[:-1]*vr[:-1], vp[:-1]+vr[:-1],
                out=np.zeros(len(thresholds)), where=(vp[:-1]+vr[:-1]) > 0)
            optimum = thresholds[np.flatnonzero(utility == utility.max())[-1]]
            np.testing.assert_allclose(row["threshold"], optimum, atol=1e-12)
        else:
            vp = val.score.to_numpy() >= row["threshold"]
            actual_fpr = vp[val.label.to_numpy() == 0].mean()
            assert actual_fpr <= config["max_fpr"] + 1e-12
    splits = pd.read_csv(directory / "splits.csv")
    assert splits.row_id.is_unique
    if config["split"] == "temporal":
        times = splits[config["time_column"]]
        if not pd.api.types.is_numeric_dtype(times):
            times = pd.to_datetime(times, utc=True)
        assert times[splits.split == "train"].max() < times[splits.split == "validation"].min()
        assert times[splits.split == "validation"].max() < times[splits.split == "test"].min()
    report = {"passed": True, "methods": results.method.tolist(),
              "checks": ["metrics from held-out scores", "validation-only threshold optimum/constraint",
                         "prediction threshold", "disjoint validation/test", "split manifest"]}
    (directory / "audit.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    audit(parser.parse_args().directory)
