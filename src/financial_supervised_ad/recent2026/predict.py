"""Reload a trusted 2026 model directory and its frozen validation threshold."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def predict(directory, frame, device="cpu"):
    directory = Path(directory)
    metadata = json.loads((directory / "inference.json").read_text())
    missing = set(metadata["features"]) - set(frame)
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")
    features = frame[metadata["features"]].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    result = pd.DataFrame({"row_id": np.arange(len(frame)), "eligible_endpoint": False,
                           "score": np.nan, "prediction": pd.array([pd.NA]*len(frame), dtype="Int64")})
    if metadata["method"] == "optimized_ebm_2026":
        model = joblib.load(directory / "model.joblib")
        scores = model.predict_proba(features)[:, 1]
        endpoints = np.arange(len(frame))
    else:
        import torch
        from .sttn import make_windows, score
        from .sttn_architecture import STTNCP
        if "Time" not in frame or frame.Time.isna().any():
            raise ValueError("STTN-CP inference needs a nonmissing Time column")
        times = frame.Time
        if not pd.api.types.is_numeric_dtype(times):
            times = pd.to_datetime(times, utc=True, errors="raise")
        order = np.argsort(times.to_numpy(), kind="stable")
        scaler = joblib.load(directory / "preprocessing.joblib")
        x = scaler.transform(features[order])
        windows, _, endpoints = make_windows(x, np.zeros(len(frame), dtype=int), order,
                                            metadata["sequence_length"])
        checkpoint = torch.load(directory / "best_model.pt", map_location=device, weights_only=True)
        model = STTNCP(**checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        torch.set_num_threads(2)
        scores = score(model, windows, 32, torch.device(device))
    result.loc[endpoints, "score"] = scores
    result.loc[endpoints, "prediction"] = (scores >= metadata["threshold"]).astype(int)
    result.loc[endpoints, "eligible_endpoint"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    predict(args.model_dir, pd.read_csv(args.csv), args.device).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
