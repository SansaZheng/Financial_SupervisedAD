"""Stream the official TensorFlow mirror; cache only a uniform sample by default.

Random priorities implement sampling without replacement independently of labels.
The complete HTTP body is hashed but is never saved to disk for sample mode.
"""
import argparse
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"


class HashReader(io.RawIOBase):
    def __init__(self, source):
        self.source = source
        self.digest = hashlib.sha256()

    def readable(self):
        return True

    def readinto(self, buffer):
        block = self.source.read(len(buffer))
        buffer[:len(block)] = block
        self.digest.update(block)
        return len(block)


def download(output, sample_size=50000, seed=42):
    output = Path(output)
    if output.exists() or Path(str(output)+".json").exists():
        raise FileExistsError("Refusing to overwrite an existing dataset/manifest")
    if sample_size < 0:
        raise ValueError("sample_size must be nonnegative (0 means full data)")
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    sample, rows, positives = None, 0, 0
    with urlopen(URL, timeout=60) as response:
        headers = {key: response.headers.get(key) for key in ("ETag", "Last-Modified", "Content-Length")}
        reader = HashReader(response)
        with io.TextIOWrapper(io.BufferedReader(reader), encoding="utf-8") as stream:
            pieces = []
            for chunk in pd.read_csv(stream, chunksize=10000):
                chunk["__source_row"] = np.arange(rows, rows+len(chunk))
                rows += len(chunk)
                positives += int(chunk["Class"].sum())
                if sample_size:
                    chunk["__priority"] = rng.random(len(chunk))
                    sample = chunk if sample is None else pd.concat([sample, chunk], ignore_index=True)
                    sample = sample.nsmallest(sample_size, "__priority")
                else:
                    pieces.append(chunk)
            if not sample_size:
                sample = pd.concat(pieces, ignore_index=True)
    if rows != 284807 or positives != 492:
        raise ValueError(f"Unexpected source shape/classes: {rows} rows, {positives} frauds")
    source_ids = sample.sort_values("__source_row")["__source_row"].to_numpy(dtype=np.int64)
    sample = sample.sort_values("__source_row").drop(columns=["__source_row", "__priority"], errors="ignore")
    sample.to_csv(output, index=False, compression="infer")
    file_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    metadata = {"dataset": "ULB/Worldline credit card fraud (2013)", "url": URL,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(), "headers": headers,
        "source_rows": rows, "source_positives": positives, "source_sha256": reader.digest.hexdigest(),
        "sampling": "uniform without replacement via iid random priorities" if sample_size else "full",
        "sample_size_requested": sample_size, "seed": seed,
        "source_row_ids": source_ids.tolist(), "rows": len(sample), "positives": int(sample["Class"].sum()),
        "sample_sha256": file_hash}
    Path(str(output)+".json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in metadata.items() if k != "source_row_ids"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/creditcard_50k.csv.gz"))
    parser.add_argument("--sample-size", type=int, default=50000, help="0 for full data; use /data or /date for large runs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    download(args.output, args.sample_size, args.seed)
