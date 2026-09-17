# Running Book RAG on a server

Everything here assumes the app lives in `/opt/bookrag` and is served behind a
TLS reverse proxy. Adjust paths to taste.

## 1. Install

```bash
sudo useradd --system --create-home --home-dir /opt/bookrag bookrag
sudo -u bookrag git clone <your-repo> /opt/bookrag      # or rsync the project
cd /opt/bookrag
deploy/install.sh                      # picks Python 3.10+; TORCH_INDEX=... for CUDA
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
and allows 200 MB uploads.

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
| Retrieval, warm | ~1.75 s on CPU/Apple GPU (reranking is ~95% of it) |
| Answer generation | ~15 s for a 400-token answer (Qwen3-32B-AWQ, 33 tok/s) |
| Cold start | ~14 s, paid once by `warmup` |
| Memory | ~1.0–1.4 GB for both encoders, shared by all sessions |

Retrieval runs **one query at a time**: four simultaneous users measured ~4x the
single-query latency. For more concurrency, either move the encoders onto the
GPU server (vLLM can serve BGE embeddings and reranking) or run several app
replicas behind the proxy with sticky sessions — each replica loads its own copy
of the models.

### Sharing one GPU with vLLM

vLLM claims `--gpu-memory-utilization` of the whole card when it starts, and
keeps it. Measured on the 48 GB server (47.65 GiB usable) with vLLM 0.28 and
Qwen/Qwen3-32B-AWQ:

| `--gpu-memory-utilization` | vLLM holds | Left for the encoders | KV cache |
|---|---|---|---|
| 0.92 (what it ran with) | 44.7 GiB | ~2.3 GiB: not enough, chat hit CUDA out of memory | 97,664 tokens |
| **0.82** (estimated) | ~40 GiB | ~7 GiB: both encoders fit (~3 GB) | ~78,000 tokens |

At 0.82 the KV cache still holds ~10 full chat requests at once (each is at
most ~8,000 tokens: 4,500 of passages, the prompt, the answer), so nothing is
lost. Restart vLLM with it, then restart the app:

```bash
vllm serve Qwen/Qwen3-32B-AWQ --gpu-memory-utilization 0.82 --max-model-len 16384
```

`--max-model-len 16384` is optional; the app never sends more than ~8,000
tokens, and the default 40,960 only lets one runaway request hog the cache.
With `embedding.device: auto` the app checks free GPU memory as each encoder
loads and uses the CPU instead when there is not enough (and moves to the CPU if
a query runs out of GPU memory), so a vLLM that grows back to 0.92 slows
retrieval down instead of breaking it. `doctor` prints the device it picked.

**Where the time goes.** Retrieval takes about 0.5–2 s. The answer takes about
15 s: Qwen3-32B-AWQ prefills a 4,000-token prompt in 2.8 s and decodes at
33 tok/s alone, or 23 tok/s each with four users at once. Every answer also
makes a second, shorter claim-check call (`grounding.verify_answer_claims`).
To answer faster, change the LLM, not the encoders:

| Model | Weights | Why |
|---|---|---|
| Qwen/Qwen3-32B-AWQ (current) | 18 GiB | Dense 32B; the gold set was measured with it |
| Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 | 29 GiB | Mixture of experts: only ~3B parameters run per token, so it should decode several times faster (not measured here). Its KV cache is smaller per token (~96 KiB vs ~256), so ~80,000 tokens still fit at 0.82. FP8 runs natively on Ada/Hopper GPUs (L40S, RTX 6000 Ada); older GPUs use a slower fallback kernel. |
| cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit | 17 GiB | The same model in 4-bit, for Ampere GPUs like this server's RTX A6000, which cannot run FP8 natively. A community build, not one from Qwen. |
| Qwen/Qwen3-14B-AWQ | 9 GiB | Dense 14B: faster and leaves the most room, but the smallest and least capable of the three (not measured here) |

The MoE model is not a thinking model, so `BOOKRAG_LLM_ENABLE_THINKING=false`
is harmless. Before switching for users, run the gold set against both models
(`eval draft-gold` for your own books, review it, then `eval gold`) and keep the
new one only if it answers and refuses the same questions. Keep BGE-M3 and the
BGE reranker: they are ~3 GB together, and replacing either means rebuilding
the index (embedder) or recalibrating `grounding.answer_threshold` (reranker).

## Security checklist

The app code handles its own part (password gate that also applies behind a
proxy or tunnel, sign-in lockout, disarmed links and images in answers, upload
checks, no unpickling, zip-bomb limits). These are the server's part:

- **vLLM has no authentication by default.** Anyone who can reach port 8000 can
  use the GPU. Start it with `--api-key <secret>` and put the same value in
  `BOOKRAG_LLM_API_KEY`. The key only guards `/v1/*`, not `/metrics`, so also
  firewall the port (or `--host 127.0.0.1` if only this app uses it).
- **Serve the app on 127.0.0.1 only**, behind nginx or `cloudflared`, so the
  proxy (and Cloudflare Access, if used) cannot be bypassed from the LAN.
- **Put Cloudflare Access (or another SSO) in front** when it faces the
  internet. Streamlit accepts uploads from any open session before sign-in, and
  one shared password is the only other gate.
- **Run it as its own user** with `deploy/bookrag.service`, not your login
  account: a parser bug exploited by an uploaded file then reaches only
  `/opt/bookrag`.
- **`chmod 600 .env`** and use a long random password (`openssl rand -base64 24`).
- **Turn OCR off (`ingest.ocr: off`) if strangers can upload.** `ocrmypdf` runs
  Ghostscript on the file, which has a history of remote-code-execution bugs.
- **Audit what is actually installed.** The server installs from the ranges in
  `requirements.txt`, so its versions differ from `requirements.lock`, which CI
  checks: `.venv/bin/pip install pip-audit && .venv/bin/pip-audit`.
- **Close what else is listening on the box**: remote-desktop and game-streaming
  services (gnome-remote-desktop, Sunshine) sit beside the app and the GPU.

## 6. Upgrades and rebuilds

An index rebuild is atomic and never disturbs readers: the new index is written
beside the old one and switched with a rename, and running sessions pick it up
on their next interaction. A cancelled or crashed build leaves the current index
untouched and resumes from its per-book cache.

## What is not tested here

The systemd unit, nginx config and installer were written on macOS, which has
neither systemd nor nginx. Read them before running, and expect to adjust paths,
the user name and the TLS certificate locations.
