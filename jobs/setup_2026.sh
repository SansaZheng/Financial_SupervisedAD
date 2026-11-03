#!/usr/bin/env bash
set -euo pipefail
task_home=/home/sudingli/cq-home/financial-supervised-ad-2026
task_env="$task_home/env"
mkdir -p "$task_home/data" "$task_home/cache/pip" "$task_home/cache/conda"
export PIP_CACHE_DIR="$task_home/cache/pip"
export CONDA_PKGS_DIRS="$task_home/cache/conda"
printf '1/4 discover conda\n' >> "$CQ_OUTPUT_DIR/progress.txt"
conda_bin=$(command -v conda || true)
if [[ -z "$conda_bin" ]]; then
  for candidate in /home/sudingli/miniconda3/bin/conda /home/sudingli/anaconda3/bin/conda /opt/conda/bin/conda /home/sudingli/miniforge3/bin/conda; do
    if [[ -x "$candidate" ]]; then conda_bin="$candidate"; break; fi
  done
fi
if [[ -z "$conda_bin" ]]; then echo 'No conda found; need a self-contained environment' >&2; exit 1; fi
if [[ ! -x "$task_env/bin/python" ]]; then
  "$conda_bin" create -y -p "$task_env" python=3.12 pip --override-channels -c conda-forge
fi
printf '2/4 install pinned dependencies\n' >> "$CQ_OUTPUT_DIR/progress.txt"
"$task_env/bin/python" -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
"$task_env/bin/python" -m pip install -r requirements-2026.txt
"$task_env/bin/python" -m pip install --no-deps -e .
printf '3/4 stream public dataset\n' >> "$CQ_OUTPUT_DIR/progress.txt"
if [[ ! -f "$task_home/data/creditcard_50k.csv.gz" ]]; then
  "$task_env/bin/python" scripts/download_creditcard.py --output "$task_home/data/creditcard_50k.csv.gz" --sample-size 50000 --seed 42
fi
"$task_env/bin/python" -m pip freeze > "$CQ_OUTPUT_DIR/environment-freeze.txt"
"$task_env/bin/python" - <<'PY'
import importlib.metadata, json, os
from pathlib import Path
meta=json.loads(Path('/home/sudingli/cq-home/financial-supervised-ad-2026/data/creditcard_50k.csv.gz.json').read_text())
assert meta['source_sha256']=='76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89'
assert meta['rows']==50000 and meta['positives']==90
out=Path(os.environ['CQ_OUTPUT_DIR'])
(out/'environment.json').write_text(json.dumps({'ready':True,'source_sha256':meta['source_sha256'],
 'versions':{p:importlib.metadata.version(p) for p in ('torch','interpret-core','numpy','scikit-learn')}},indent=2)+'\n')
PY
printf '4/4 ready\n' >> "$CQ_OUTPUT_DIR/progress.txt"
