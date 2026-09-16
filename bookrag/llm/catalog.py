"""What each installed model is good for, in this pipeline.

Ollama tells you a model's size and parameter count; it does not tell you
whether to point grounded answering at it. This turns `/api/tags` into advice.

Guidance is matched by pattern so a model pulled later still gets sensible
copy, and falls back to parameter-count heuristics for anything unrecognised.
Throughput figures marked `measured: True` were benchmarked on this machine
(see the notes in config.yaml); the rest are extrapolated from parameter count
and should be treated as rough.
"""
from __future__ import annotations

import re

# Measured on this project's 16 GB Apple Silicon box, Q4_K_M, decode only.
_MEASURED_TOK_S = {
    "qwen3:8b": 9.8,
    "qwen3:4b-instruct": 15.3,
}

_GUIDANCE: list[tuple[re.Pattern, dict]] = [
    (re.compile(r"^qwen3:8b"), {
        "role": "accuracy",
        "use_for": "Derivations, formulas, multi-step explanations, exam-question "
                   "writing and verification. The default for physics and "
                   "engineering material.",
        "avoid_for": "Nothing on quality grounds — it is simply the slowest option "
                     "here and the heaviest on RAM.",
        "examples": [
            "Derive the radiation resistance of a Hertzian dipole",
            "Explain the Carnot cycle step by step and show why efficiency "
            "depends only on the two temperatures",
            "Generating an exam paper (paper.generate) or verifying questions",
        ],
    }),
    (re.compile(r"^qwen3:4b-instruct"), {
        "role": "speed",
        "use_for": "Prose books, summaries, definition lookups, and any question "
                   "whose answer is quoted rather than reasoned. Frees ~2.7 GB "
                   "versus the 8B, which matters when memory is tight.",
        "avoid_for": "Derivations and multi-part questions — it drops steps, and "
                     "weaker citation discipline turns into refusals under "
                     "grounding.require_citations.",
        "examples": [
            "What do the Okinawans eat?",
            "Define ikigai in the book's own words",
            "Summarise what a chapter says about handling stress",
        ],
    }),
    (re.compile(r"^qwen3:\d+b$"), {
        "role": "reasoning (thinking)",
        "use_for": "Nothing here by default: this is the thinking variant, and "
                   "this pipeline runs with llm.think false.",
        "avoid_for": "Chat — you pay for reasoning tokens that get stripped. "
                     "Prefer the -instruct build at the same size.",
        "examples": [],
    }),
    (re.compile(r"^qwen2\.5:"), {
        "role": "fallback",
        "use_for": "A second opinion when a qwen3 answer looks wrong. Solid "
                   "general instruction-following.",
        "avoid_for": "Being the default — qwen3 at the same size is stronger on "
                     "technical material.",
        "examples": [
            "Re-asking a question qwen3:8b refused, to see if it was a retrieval "
            "problem or a model problem",
        ],
    }),
    (re.compile(r"embed|bge|nomic|minilm", re.I), {
        "role": "embedding — not for chat",
        "use_for": "Nothing. This is an embedding model; it cannot answer "
                   "questions. Embeddings here come from sentence-transformers, "
                   "not Ollama.",
        "avoid_for": "Selecting it as the answering model at all.",
        "examples": [],
    }),
]


def _fallback(params_b: float) -> dict:
    """Guidance for a model this catalog has no specific entry for."""
    if params_b and params_b <= 2:
        return {"role": "very fast, low quality",
                "use_for": "Quick lookups where a wrong answer is cheap.",
                "avoid_for": "Grounded answering — too weak to hold citations."}
    if params_b and params_b <= 5:
        return {"role": "speed",
                "use_for": "Prose, summaries, definition lookups.",
                "avoid_for": "Derivations and multi-step reasoning."}
    if params_b and params_b <= 9:
        return {"role": "balanced",
                "use_for": "General grounded answering.",
                "avoid_for": "Nothing specific; watch RAM alongside the encoders."}
    return {"role": "accuracy, heavy",
            "use_for": "Hard reasoning, if you have the memory for it.",
            "avoid_for": "16 GB machines running encoders at the same time."}


def _params_b(details: dict) -> float:
    raw = str(details.get("parameter_size", "") or "")
    m = re.match(r"([\d.]+)\s*([BbMm])", raw)
    if not m:
        return 0.0
    val = float(m.group(1))
    return val / 1000.0 if m.group(2).lower() == "m" else val


def describe_model(name: str, size_bytes: int = 0, details: dict | None = None) -> dict:
    details = details or {}
    params = _params_b(details)
    guide = next((g for pat, g in _GUIDANCE if pat.search(name)), None) or _fallback(params)

    measured = name in _MEASURED_TOK_S
    if measured:
        tok_s = _MEASURED_TOK_S[name]
    elif params:
        # Decode is memory-bandwidth bound, so throughput falls roughly with
        # parameter count. Anchored on the measured 8.2B point.
        tok_s = round(9.8 * (8.2 / params), 1)
    else:
        tok_s = 0.0

    return {
        "name": name,
        "size_gb": round(size_bytes / 1e9, 1) if size_bytes else 0.0,
        "params": details.get("parameter_size", ""),
        "quant": details.get("quantization_level", ""),
        "tok_s": tok_s,
        "tok_s_measured": measured,
        "chat_capable": "not for chat" not in guide["role"],
        "examples": [],
        **guide,
    }


def summary_line(d: dict) -> str:
    """One-line label for a dropdown or list."""
    bits = [d["role"]]
    if d["size_gb"]:
        bits.append(f"{d['size_gb']} GB")
    if d["tok_s"]:
        bits.append(f"~{d['tok_s']} tok/s" + ("" if d["tok_s_measured"] else " est."))
    return f"{d['name']} — " + " · ".join(bits)
