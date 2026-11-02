"""Apply a trusted, locally saved pipeline and its frozen validation threshold."""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from financial_supervised_ad.experiment import score_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    # joblib/pickle files execute code on load; only load your own trusted artifacts.
    bundle = joblib.load(args.model)
    frame = pd.read_csv(args.csv)
    missing = set(bundle["features"])-set(frame.columns)
    if missing:
        raise ValueError(f"Missing training features: {sorted(missing)}")
    from financial_supervised_ad.data import prepare_features
    features = prepare_features(frame[bundle["features"]].copy(), "__unused_label", "__unused_time")
    scores = score_model(bundle["pipeline"], features)
    pd.DataFrame({"score": scores, "prediction": (scores >= bundle["threshold"]).astype(int)}).to_csv(
        args.output, index=False)


if __name__ == "__main__":
    main()
