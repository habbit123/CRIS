#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-cris}"
PYTHON_VERSION="${PYTHON_VERSION:-}"
TORCH_INSTALL_CMD="${TORCH_INSTALL_CMD:-}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but was not found in PATH." >&2
  exit 1
fi

if [[ -z "$PYTHON_VERSION" ]]; then
  echo "Set PYTHON_VERSION before running this script, for example: PYTHON_VERSION=3.10 bash scripts/setup_linux.sh" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Using existing conda env: $ENV_NAME"
else
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi

conda activate "$ENV_NAME"
python -m pip install --upgrade pip

if [[ -n "$TORCH_INSTALL_CMD" ]]; then
  eval "$TORCH_INSTALL_CMD"
else
  echo "TORCH_INSTALL_CMD is empty. Install the correct torch/torchvision build for your CUDA setup before training." >&2
fi

python -m pip install -r requirement.txt

python - <<'PY'
missing = []
for name in ['torch', 'cv2', 'lmdb', 'pyarrow', 'wandb', 'loguru', 'numpy', 'yaml']:
    try:
        __import__(name)
    except Exception:
        missing.append(name)
if missing:
    raise SystemExit('Missing imports after setup: ' + ', '.join(missing))
print('Environment import check passed.')
PY
