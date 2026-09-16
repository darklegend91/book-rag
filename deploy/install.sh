#!/usr/bin/env bash
# Install Book RAG on a Linux server. Run from the project root: deploy/install.sh
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3.11}"
# CPU wheels by default; for a CUDA box set e.g.
#   TORCH_INDEX=https://download.pytorch.org/whl/cu124 deploy/install.sh
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"

cd "$APP_DIR"
command -v "$PYTHON" >/dev/null || { echo "need $PYTHON on PATH (set PYTHON=...)"; exit 1; }

[ -d .venv ] || "$PYTHON" -m venv .venv
.venv/bin/pip install --disable-pip-version-check --upgrade pip
# requirements.lock is pinned from macOS; on Linux install the ranges from
# requirements.txt so pip picks platform wheels, then pin your own lock with
#   .venv/bin/pip freeze > requirements.server.lock
.venv/bin/pip install --extra-index-url "$TORCH_INDEX" -r requirements.txt

[ -f .env ] || { cp deploy/env.server.example .env; echo "created .env from deploy/env.server.example"; }
mkdir -p data/books data/pyqs data/index data/exports data/eval

echo
echo "Next:"
echo "  1. Edit .env: set BOOKRAG_APP_PASSWORD and check BOOKRAG_LLM_BASE_URL."
echo "  2. .venv/bin/python -m bookrag.cli doctor"
echo "  3. Put books in data/books/ and run: .venv/bin/python -m bookrag.cli ingest"
echo "  4. .venv/bin/python -m bookrag.cli warmup     # downloads ~4.6 GB of encoder weights once"
echo "  5. sudo cp deploy/bookrag.service /etc/systemd/system/ && sudo systemctl enable --now bookrag"
