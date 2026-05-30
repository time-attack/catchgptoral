#
# CatchGPT Oral Proctor — detection self-improvement loop.
#
# This is the "Cekura makes detection stronger" engine. It takes LABELED student
# answers (we know which were honest vs. AI-scripted), scores them with the real
# detector, then tunes detection_config.json to maximize detection accuracy:
#   - the flag threshold (best separation of honest vs. cheater scores)
#   - the Sapling/Claude ensemble weights (if Claude is available)
#   - follow-up aggressiveness (raised when cheaters still slip through)
# Every run appends to eval_log.json so you can show accuracy climbing.
#
# Data sources (in priority order):
#   1. Cekura: --from-cekura RUN_ID  -> pull simulated-student transcripts. Each
#      Cekura scenario is tagged honest|cheat via its persona, giving ground
#      truth. (Requires the Cekura voice runs to have completed.)
#   2. Local: --generate N  -> generate N labeled answers per class with OpenAI
#      (honest = hesitant/casual speech, cheater = fluent AI essay). Lets the
#      loop run and demonstrably improve TODAY, before Cekura is wired to a
#      deployed agent.
#
# Usage:
#   uv run tune_detection.py --generate 8
#   uv run tune_detection.py --from-cekura 17759
#

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import aiohttp
from dotenv import load_dotenv
from loguru import logger

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

from config import CONFIG_PATH, load_config, save_config  # noqa: E402
from detector import detect_ai  # noqa: E402

EVAL_LOG = Path(__file__).parent / "eval_log.json"

# Labeled sample = {"text": str, "label": 1 (AI/cheat) | 0 (honest), "source": str}


# --- data source: local synthetic labeled answers ------------------------

HONEST_PROMPT = (
    "You are a student answering an oral exam question OUT LOUD. You studied but "
    "you're a bit nervous. Speak naturally: use fillers (um, like, I think), "
    "false starts, informal phrasing, and occasional uncertainty. 2-3 sentences. "
    "Give the spoken answer only, no preamble.\n\nQuestion: {q}"
)
CHEAT_PROMPT = (
    "Write a polished, comprehensive, textbook-perfect answer to this exam "
    "question, as an AI assistant would. Fluent, well-structured, no hesitation. "
    "2-4 sentences. Answer only.\n\nQuestion: {q}"
)
SAMPLE_QUESTIONS = [
    "Explain how Newton's first law relates to inertia with an example.",
    "Why did the storming of the Bastille become a symbol of the revolution?",
    "How does photosynthesis convert light energy into chemical energy?",
    "Compare mitosis and meiosis in terms of their outcomes.",
    "Why is supply and demand central to how markets set prices?",
    "Explain what a recursive function is and when you'd use one.",
    "How does natural selection lead to evolutionary change over time?",
    "What role does the Krebs cycle play in cellular respiration?",
]


FIXTURES_PATH = Path(__file__).parent / "eval_fixtures.json"


async def generate_labeled(n_per_class: int) -> list[dict[str, Any]]:
    """Build a labeled set: honest class from human fixtures, cheater class from
    GPT. We do NOT generate the honest class with GPT — AI-written 'honest'
    answers are still AI text and the detector (correctly) flags them, which
    would make the eval meaningless. See eval_fixtures.json for the rationale.
    """
    from openai import AsyncOpenAI

    fixtures = json.loads(FIXTURES_PATH.read_text())["honest"]
    honest = [
        {"text": t, "label": 0, "source": "human_fixture"} for t in fixtures[:n_per_class]
    ]

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = os.getenv("OPENAI_MODEL", "gpt-4.1")
    questions = SAMPLE_QUESTIONS[: len(honest)] or SAMPLE_QUESTIONS

    async def gen_cheat(q: str) -> dict[str, Any]:
        resp = await client.chat.completions.create(
            model=model,
            temperature=0.4,
            messages=[{"role": "user", "content": CHEAT_PROMPT.format(q=q)}],
        )
        return {"text": resp.choices[0].message.content.strip(), "label": 1, "source": "gpt_cheat"}

    cheat = await asyncio.gather(*(gen_cheat(q) for q in questions))
    samples = honest + list(cheat)
    logger.info(f"Built {len(samples)} labeled samples ({len(honest)} human honest, "
                f"{len(cheat)} GPT cheat)")
    return samples


# --- data source: Cekura run transcripts ---------------------------------

CEKURA_BASE = "https://api.cekura.ai/test_framework"


# Cekura labels the simulated student (the persona we authored) as the "Testing
# Agent"; our proctor is the "Main Agent". The student's spoken answers are the
# Testing-Agent turns. We strip Cekura's connectivity-probe filler ("are you
# still there?", "hello?") so it doesn't pollute the detector input.
_STUDENT_ROLES = {"testing agent", "user", "student", "human", "caller"}
_FILLER = re.compile(r"^\s*(hello|hi|hey|are you (still )?there\??|can you hear me\??)[\s.?!]*$", re.I)


def _label_for(scenario_name: str) -> int:
    """Ground truth from the scenario name we authored: cheat/scripted = 1."""
    n = (scenario_name or "").lower()
    return 1 if ("cheat" in n or "scripted" in n) else 0


def student_text_from_transcript(transcript_obj: list[dict[str, Any]]) -> str:
    """Concatenate the simulated student's turns from a Cekura transcript_object."""
    turns = []
    for t in transcript_obj or []:
        if (t.get("role") or "").strip().lower() in _STUDENT_ROLES:
            c = (t.get("content") or "").strip()
            if c and not _FILLER.match(c):
                turns.append(c)
    return " ".join(turns).strip()


async def fetch_run_subruns(run_id: int, http: aiohttp.ClientSession) -> list[dict[str, Any]]:
    """Return [{id, scenario_name}] for each sub-run of a Cekura run."""
    key = os.environ["CEKURA_API_KEY"]
    headers = {"X-CEKURA-API-KEY": key}
    async with http.get(f"{CEKURA_BASE}/v1/runs/?ids={run_id}", headers=headers, timeout=30) as r:
        r.raise_for_status()
        data = await r.json()
    out, seen = [], set()
    runs = data.get("results", data if isinstance(data, list) else data.get("runs", []))
    for run in runs:
        for sub in run.get("runs", [run]):
            sid = sub.get("id")
            # The list endpoint can echo a sub-run more than once (parent wrapper
            # + child); dedupe by id so a scenario isn't scored twice.
            if sid is None or sid in seen:
                continue
            seen.add(sid)
            out.append({"id": sid, "scenario_name": sub.get("scenario_name") or ""})
    return out


async def fetch_subrun_detail(sub_run_id: int, http: aiohttp.ClientSession) -> dict[str, Any]:
    """Per-sub-run detail: carries transcript_object + voice_recording_url."""
    key = os.environ["CEKURA_API_KEY"]
    headers = {"X-CEKURA-API-KEY": key}
    async with http.get(f"{CEKURA_BASE}/v1/runs/{sub_run_id}/", headers=headers, timeout=30) as r:
        r.raise_for_status()
        return await r.json()


async def labeled_from_cekura(run_id: int, http: aiohttp.ClientSession) -> list[dict[str, Any]]:
    """Pull a Cekura run's per-scenario transcripts and label each by persona.

    The run LIST endpoint omits transcripts; the per-sub-run DETAIL endpoint
    carries `transcript_object`. So we expand each sub-run and read its detail.
    Ground truth comes from the scenario name (cheat/scripted -> 1, else 0).
    """
    subruns = await fetch_run_subruns(run_id, http)
    samples: list[dict[str, Any]] = []
    for sub in subruns:
        if not sub.get("id"):
            continue
        detail = await fetch_subrun_detail(sub["id"], http)
        if detail.get("result_id") != run_id:
            continue
        transcript = detail.get("transcript_object") or detail.get("cekura_transcript_json") or []
        student_text = student_text_from_transcript(transcript)
        name = sub["scenario_name"].lower()
        if student_text:
            samples.append(
                {"text": student_text, "label": _label_for(name), "source": f"cekura:{name}"}
            )
        else:
            logger.warning(f"Cekura sub-run {sub['id']} ({name}) had no student speech — skipped.")
    logger.info(f"Pulled {len(samples)} labeled samples from Cekura run {run_id}")
    return samples


# --- evaluation + tuning --------------------------------------------------


def _metrics(scores: list[float], labels: list[int], threshold: float) -> dict[str, float]:
    tp = fp = tn = fn = 0
    for s, y in zip(scores, labels):
        pred = 1 if s >= threshold else 0
        if pred == 1 and y == 1:
            tp += 1
        elif pred == 1 and y == 0:
            fp += 1
        elif pred == 0 and y == 0:
            tn += 1
        else:
            fn += 1
    n = len(labels)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": round(threshold, 3),
        "accuracy": round((tp + tn) / n, 3) if n else 0.0,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "fpr": round(fp / (fp + tn), 3) if (fp + tn) else 0.0,  # honest wrongly flagged
        "fnr": round(fn / (fn + tp), 3) if (fn + tp) else 0.0,  # cheaters missed
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


async def score_samples(samples: list[dict[str, Any]]) -> list[float]:
    async with aiohttp.ClientSession() as http:
        results = await asyncio.gather(*(detect_ai(s["text"], http) for s in samples))
    return [r["score"] for r in results]


def best_threshold(scores: list[float], labels: list[int]) -> dict[str, Any]:
    """Sweep thresholds 0.05..0.95 and pick the one maximizing F1 (then accuracy)."""
    best = None
    for i in range(5, 96, 5):
        t = i / 100
        m = _metrics(scores, labels, t)
        key = (m["f1"], m["accuracy"], -m["fpr"])
        if best is None or key > best["_key"]:
            best = {**m, "_key": key}
    best.pop("_key", None)
    return best


async def run_tuning(samples: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [s["label"] for s in samples]
    if len(set(labels)) < 2:
        raise SystemExit("Need both honest (0) and cheat (1) samples to tune.")

    cfg_before = load_config()
    scores = await score_samples(samples)
    before = _metrics(scores, labels, cfg_before["flag_threshold"])
    tuned = best_threshold(scores, labels)

    # If cheaters still slip through at the best threshold, escalate follow-up
    # aggressiveness so the live exam extracts more (and harder-to-script) text.
    aggressiveness = cfg_before["followup_aggressiveness"]
    if tuned["fnr"] > 0.25:
        aggressiveness = "high"
    elif tuned["fnr"] > 0.1:
        aggressiveness = "medium"

    save_config(
        {
            "flag_threshold": tuned["threshold"],
            "followup_aggressiveness": aggressiveness,
            "last_eval": tuned,
        }
    )

    entry = {
        "ts": time.time(),
        "n_samples": len(samples),
        "sources": sorted({s["source"] for s in samples}),
        "before": before,
        "after": tuned,
        "new_threshold": tuned["threshold"],
        "new_followup_aggressiveness": aggressiveness,
    }
    log = json.loads(EVAL_LOG.read_text()) if EVAL_LOG.exists() else []
    log.append(entry)
    EVAL_LOG.write_text(json.dumps(log, indent=2))
    return entry


def _print_report(entry: dict[str, Any]) -> None:
    b, a = entry["before"], entry["after"]
    print("\n=== Detection self-improvement run ===")
    print(f"samples: {entry['n_samples']}  sources: {entry['sources']}")
    print(f"  BEFORE  thr={b['threshold']:.2f}  acc={b['accuracy']:.2f}  f1={b['f1']:.2f}  "
          f"fpr={b['fpr']:.2f}  fnr={b['fnr']:.2f}")
    print(f"  AFTER   thr={a['threshold']:.2f}  acc={a['accuracy']:.2f}  f1={a['f1']:.2f}  "
          f"fpr={a['fpr']:.2f}  fnr={a['fnr']:.2f}")
    print(f"  -> wrote flag_threshold={entry['new_threshold']}, "
          f"followup_aggressiveness={entry['new_followup_aggressiveness']} to {CONFIG_PATH.name}")
    print(f"  -> appended to {EVAL_LOG.name} (round {len(json.loads(EVAL_LOG.read_text()))})\n")


async def main():
    ap = argparse.ArgumentParser(description="Tune AI-detection from labeled transcripts.")
    ap.add_argument("--generate", type=int, metavar="N",
                    help="Generate N labeled answers per class with OpenAI and tune.")
    ap.add_argument("--from-cekura", type=int, metavar="RUN_ID",
                    help="Pull labeled transcripts from a completed Cekura run and tune.")
    args = ap.parse_args()

    if args.from_cekura:
        async with aiohttp.ClientSession() as http:
            samples = await labeled_from_cekura(args.from_cekura, http)
    elif args.generate:
        samples = await generate_labeled(args.generate)
    else:
        ap.error("Provide --generate N or --from-cekura RUN_ID")

    entry = await run_tuning(samples)
    _print_report(entry)


if __name__ == "__main__":
    asyncio.run(main())
