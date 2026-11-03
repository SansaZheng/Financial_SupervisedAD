#!/usr/bin/env bash
set -euo pipefail
method="$1"
shift
python -m financial_supervised_ad.recent2026.cli \
  --method "$method" \
  --csv /home/sudingli/cq-home/financial-supervised-ad-2026/data/creditcard_50k.csv.gz \
  --seed 42 --n-jobs 4 --output "$CQ_OUTPUT_DIR/run" "$@"
python scripts/audit_results.py "$CQ_OUTPUT_DIR/run"
python -m pip freeze > "$CQ_OUTPUT_DIR/environment-freeze.txt"
