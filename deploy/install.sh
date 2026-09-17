#!/usr/bin/env bash
# Install Book RAG on a Linux server. Run from the project root: deploy/install.sh
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Python 3.10 or newer. 3.11 is what this was developed against; 3.10 is tested
# to compile and run the same code. Override with PYTHON=/path/to/python.
pick_python() {
  if [ -n "${PYTHON:-}" ]; then echo "$PYTHON"; return; fi
  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && \
       "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
      echo "$candidate"; return
    fi
  done
  echo ""
}
PYTHON="$(pick_python)"
# CPU wheels by default; for a CUDA box set e.g.
#   TORCH_INDEX=https://download.pytorch.org/whl/cu124 deploy/install.sh
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"

cd "$APP_DIR"
[ -n "$PYTHON" ] || { echo "need Python 3.10+ on PATH (set PYTHON=/path/to/python)"; exit 1; }
echo "using $($PYTHON --version) from $(command -v "$PYTHON")"

# Debian/Ubuntu ship venv separately; say so before the confusing ensurepip error.
"$PYTHON" -c "import ensurepip" 2>/dev/null || {
  echo "this Python has no venv support; install it, e.g.:"
  echo "  sudo apt install -y $(basename "$PYTHON")-venv"
  exit 1
}

[ -d .venv ] || "$PYTHON" -m venv .venv
.venv/bin/pip install --disable-pip-version-check --upgrade pip
# requirements.lock is pinned from macOS; on Linux install the ranges from
# requirements.txt so pip picks platform wheels, then pin your own lock with
#   .venv/bin/pip freeze > requirements.server.lock
.venv/bin/pip install --extra-index-url "$TORCH_INDEX" -r requirements.txt

[ -f .env ] || { cp deploy/env.server.example .env; echo "created .env from deploy/env.server.example"; }
# It holds the app password and any LLM API key.
chmod 600 .env
mkdir -p data/books data/pyqs data/index data/exports data/eval

echo
echo "Next:"
echo "  1. Edit .env: set BOOKRAG_APP_PASSWORD and check BOOKRAG_LLM_BASE_URL."
echo "  2. .venv/bin/python -m bookrag.cli doctor"
echo "  3. Put books in data/books/ and run: .venv/bin/python -m bookrag.cli ingest"
echo "  4. .venv/bin/python -m bookrag.cli warmup     # downloads ~4.6 GB of encoder weights once"
echo "  5. sudo cp deploy/bookrag.service /etc/systemd/system/ && sudo systemctl enable --now bookrag"
