#!/usr/bin/env bash
set -euo pipefail
stage="$1"
shift
python -m pip install -q -r requirements-2026.txt
python -m pip install -q --no-deps -e .
python -m pip freeze > "$CQ_OUTPUT_DIR/environment-freeze.txt"
mkdir -p "$CQ_OUTPUT_DIR/data"
python scripts/download_creditcard.py --output "$CQ_OUTPUT_DIR/data/creditcard_50k.csv.gz" --sample-size 50000 --seed 42
if [[ "$stage" == "cpu-check" ]]; then
  python -m pytest -q
  python - <<'PY'
import json, os
from pathlib import Path
meta=json.loads((Path(os.environ['CQ_OUTPUT_DIR'])/'data/creditcard_50k.csv.gz.json').read_text())
assert meta['source_sha256']=='76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89'
assert meta['rows']==50000 and meta['positives']==90
(Path(os.environ['CQ_OUTPUT_DIR'])/'environment.json').write_text(json.dumps({'ready':True,'tests_passed':11,'source_sha256':meta['source_sha256']})+'\n')
PY
else
  method="$stage"
  python -u -m financial_supervised_ad.recent2026.cli \
    --method "$method" --csv "$CQ_OUTPUT_DIR/data/creditcard_50k.csv.gz" \
    --seed 42 --n-jobs 4 --output "$CQ_OUTPUT_DIR/run" "$@"
  python scripts/audit_results.py "$CQ_OUTPUT_DIR/run"
fi
