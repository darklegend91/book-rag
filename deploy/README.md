# Running Book RAG on a server

Everything here assumes the app lives in `/opt/bookrag` and is served behind a
TLS reverse proxy. Adjust paths to taste.

## 1. Install

```bash
sudo useradd --system --create-home --home-dir /opt/bookrag bookrag
sudo -u bookrag git clone <your-repo> /opt/bookrag      # or rsync the project
cd /opt/bookrag
PYTHON=python3.11 deploy/install.sh                     # TORCH_INDEX=... for CUDA
```

`requirements.lock` is pinned from macOS. On Linux the installer uses the
ranges in `requirements.txt`; pin your own afterwards:

```bash
.venv/bin/pip freeze > requirements.server.lock
```

## 2. Configure

Edit `.env` (created from `deploy/env.server.example`):

- **`BOOKRAG_APP_PASSWORD`** — required. The app refuses to serve a network
  address without it, because anyone who reaches it can upload books, read
  generated papers and spend your GPU time.
- **`BOOKRAG_LLM_BASE_URL`** — use `http://127.0.0.1:8000/v1` if vLLM runs on
  the same machine.
- **`HF_HOME`** — a persistent path; the encoder weights are ~4.6 GB.

`.streamlit/config.toml` already binds `0.0.0.0:8501`, disables file watching
and allows 500 MB uploads.

## 3. First run

```bash
.venv/bin/python -m bookrag.cli doctor      # backend, index, weights, memory
# put books in data/books/
.venv/bin/python -m bookrag.cli ingest      # downloads the encoders on first use
.venv/bin/python -m bookrag.cli warmup      # loads them; ~14 s, so users don't pay it
```

## 4. Service and proxy

```bash
sudo cp deploy/bookrag.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now bookrag
sudo journalctl -u bookrag -f               # BOOKRAG_LOG_LEVEL controls detail
```

`deploy/nginx-bookrag.conf` is a TLS + websocket proxy with long timeouts (index
builds and paper generation take minutes) and a matching upload limit.

## 5. Capacity

Measured on a 16 GB Apple Silicon box, 780-chunk index:

| | |
|---|---|
| Retrieval, warm | ~1.3 s (reranking is ~95% of it) |
| Answer generation | 1–10 s, on the LLM server |
| Cold start | ~14 s, paid once by `warmup` |
| Memory | ~1.0–1.4 GB for both encoders, shared by all sessions |

Retrieval runs **one query at a time**: four simultaneous users measured ~4x the
single-query latency. For more concurrency, either move the encoders onto the
GPU server (vLLM can serve BGE embeddings and reranking) or run several app
replicas behind the proxy with sticky sessions — each replica loads its own copy
of the models.

## 6. Upgrades and rebuilds

An index rebuild is atomic and never disturbs readers: the new index is written
beside the old one and switched with a rename, and running sessions pick it up
on their next interaction. A cancelled or crashed build leaves the current index
untouched and resumes from its per-book cache.

## What is not tested here

The systemd unit, nginx config and installer were written on macOS, which has
neither systemd nor nginx. Read them before running, and expect to adjust paths,
the user name and the TLS certificate locations.
