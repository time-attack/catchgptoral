#
# CatchGPT Oral Proctor — live Training & Detection Observatory.
#
# This is the backend for the /training dashboard: a live, fully-transparent view
# of the Cekura self-improvement loop. It does NOT hide anything — every step the
# loop takes (pull transcript -> score with the detector -> classify vs ground
# truth -> sweep thresholds -> rewrite config) is streamed as an event so you can
# watch "what it's thinking" in real time, with the actual call audio as media.
#
# Endpoints (wired in proctor_server.py):
#   GET  /training                     -> the observatory UI
#   GET  /api/training/state           -> current config, eval-log history, agent/scenarios
#   GET  /api/training/run/{run_id}    -> snapshot of a Cekura run (transcripts, audio, scores)
#   GET  /api/training/stream/{run_id} -> SSE: live "thinking" feed + final tuning result
#
# Ground truth is the scenario name we authored (cheat/scripted -> AI, else human),
# so a Cekura run is a LABELED test set: we can compute real accuracy/F1, not vibes.
#

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import aiohttp
from loguru import logger

from config import CONFIG_PATH, load_config
from detector import detect_ai
from tune_detection import (
    EVAL_LOG,
    FIXTURES_PATH,
    SAMPLE_QUESTIONS,
    _label_for,
    _metrics,
    best_threshold,
    fetch_run_subruns,
    fetch_subrun_detail,
    labeled_from_cekura,
    run_tuning,
    student_text_from_transcript,
)

CEKURA_BASE = "https://api.cekura.ai/test_framework"
STATE_PATH = Path(__file__).parent / "cekura_state.json"
HUMAN_LABELS_PATH = Path(__file__).parent / "human_labels.json"
HUMAN_VOICE_PATH = Path(__file__).parent / "human_voice_samples.json"

# Clear AI / "ChatGPT student" answers to the same questions the human fixtures
# cover — fluent, textbook-perfect, no hesitation. Paired with the human fixtures
# (eval_fixtures.json), these give the human-in-the-loop trainer balanced samples.
_AI_POOL = [
    "Newton's first law, the law of inertia, states that an object remains at rest "
    "or in uniform motion in a straight line unless acted upon by a net external "
    "force. For example, a book on a table stays put until a force displaces it.",
    "The storming of the Bastille on 14 July 1789 became emblematic of the French "
    "Revolution because it represented the people's rejection of absolute monarchy "
    "and the collapse of royal authority over Paris.",
    "Photosynthesis converts light energy into chemical energy through two stages: "
    "the light-dependent reactions, which produce ATP and NADPH, and the Calvin "
    "cycle, which fixes carbon dioxide into glucose.",
    "Mitosis produces two genetically identical diploid daughter cells for growth "
    "and repair, whereas meiosis produces four genetically distinct haploid gametes "
    "for sexual reproduction.",
    "Supply and demand determine market prices through equilibrium: when demand "
    "exceeds supply, prices rise; when supply exceeds demand, prices fall, until the "
    "quantity supplied equals the quantity demanded.",
    "A recursive function is one that calls itself to solve smaller instances of a "
    "problem, requiring a base case to terminate. A classic example is computing a "
    "factorial, where n! is defined as n times (n-1)!.",
    "Natural selection is the process by which heritable traits that enhance survival "
    "and reproduction become more common across generations, driving the gradual "
    "adaptation and evolution of populations.",
    "The Krebs cycle, occurring in the mitochondrial matrix, oxidizes acetyl-CoA to "
    "produce ATP, NADH, and FADH2, feeding electrons into the electron transport "
    "chain following glycolysis.",
]

# Cekura terminal sub-run states (not "running"/"queued"/"in_progress").
_TERMINAL = {"completed", "failed", "error", "cancelled", "evaluating", "success"}


def _headers() -> dict[str, str]:
    return {"X-CEKURA-API-KEY": os.environ["CEKURA_API_KEY"]}


def load_state() -> dict[str, Any]:
    return json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}


def training_state() -> dict[str, Any]:
    """Everything the dashboard needs at load: live config, full eval history, agent."""
    cfg = load_config()
    history = json.loads(EVAL_LOG.read_text()) if EVAL_LOG.exists() else []
    return {
        "config": cfg,
        "config_path": CONFIG_PATH.name,
        "history": history,
        "rounds": len(history),
        "state": load_state(),
        "detector": _detector_name(),
        "human_voice": human_voice_stats(),
        "last_ai_run_id": load_state().get("last_ai_run_id"),
    }


def _detector_name() -> str:
    if os.getenv("ZEROGPT_API_KEY"):
        return "ZeroGPT"
    if os.getenv("SAPLING_API_KEY"):
        return "Sapling"
    return "Claude heuristic"


def _verdict(score: float, label: int, threshold: float) -> dict[str, Any]:
    pred = 1 if score >= threshold else 0
    return {
        "score": round(score, 3),
        "predicted": pred,
        "predicted_label": "AI / cheating" if pred else "human / honest",
        "truth": label,
        "truth_label": "AI / cheating" if label else "human / honest",
        "correct": pred == label,
    }


async def run_snapshot(run_id: int, http: aiohttp.ClientSession) -> dict[str, Any]:
    """Full snapshot of a Cekura run for the dashboard: per-scenario transcript,
    audio recording URL, our detector score, and the labeled verdict."""
    cfg = load_config()
    threshold = cfg["flag_threshold"]
    subruns = await fetch_run_subruns(run_id, http)

    scenarios = []
    for sub in subruns:
        sid = sub.get("id")
        name = sub.get("scenario_name") or ""
        if not sid:
            continue
        detail = await fetch_subrun_detail(sid, http)
        transcript = detail.get("transcript_object") or detail.get("cekura_transcript_json") or []
        student_text = student_text_from_transcript(transcript)
        label = _label_for(name)
        entry: dict[str, Any] = {
            "sub_run_id": sid,
            "scenario_name": name,
            "ground_truth": label,
            "ground_truth_label": "AI / cheating" if label else "human / honest",
            "status": detail.get("status"),
            "duration": detail.get("duration"),
            "ended_reason": (detail.get("metadata") or {}).get("ended_reason"),
            "transcript": transcript,
            "student_text": student_text,
            "student_word_count": len(student_text.split()),
            "voice_recording_url": detail.get("voice_recording_url"),
            "error_message": detail.get("error_message") or "",
        }
        if student_text:
            result = await detect_ai(student_text, http)
            entry["detection"] = {
                "source": result["source"],
                "confident": result.get("confident"),
                **_verdict(result["score"], label, threshold),
            }
        else:
            entry["detection"] = None
        scenarios.append(entry)

    # Aggregate labeled accuracy across the scenarios that produced speech.
    scored = [s for s in scenarios if s["detection"]]
    if scored:
        scores = [s["detection"]["score"] for s in scored]
        labels = [s["ground_truth"] for s in scored]
        agg = _metrics(scores, labels, threshold)
    else:
        agg = None
    return {
        "run_id": run_id,
        "threshold": threshold,
        "scenarios": scenarios,
        "aggregate": agg,
        "scored_count": len(scored),
        "total_count": len(scenarios),
    }


async def _poll_until_done(run_id: int, http: aiohttp.ClientSession, emit, max_polls=80, interval=8):
    """Poll the run, emitting a status event each tick, until all sub-runs terminate."""
    for i in range(max_polls):
        async with http.get(
            f"{CEKURA_BASE}/v1/runs/?ids={run_id}", headers=_headers(), timeout=30
        ) as r:
            data = await r.json()
        runs = data.get("results", data if isinstance(data, list) else [])
        subs = [s for run in runs for s in run.get("runs", [run])]
        statuses = {s.get("scenario_name"): s.get("status") for s in subs}
        await emit("run_status", {"poll": i + 1, "statuses": statuses})
        if subs and all((s.get("status") or "").lower() in _TERMINAL for s in subs):
            return True
        await asyncio.sleep(interval)
    return False


async def _poll_with_turns(run_id, http, emit, max_polls=150, interval=3):
    """Poll a run and stream transcript turns + DETAILED telemetry LIVE, scoped to
    this run (sub-run detail.result_id == run_id). Returns {sub_run_id: detail}."""

    async def log(layer, msg, data=None):
        await emit("log", {"layer": layer, "msg": msg, "data": data or {}})

    seen: dict[int, int] = {}        # sub_run_id -> turns already emitted
    details: dict[int, dict] = {}    # sub_run_id -> latest detail
    seen_status: dict[int, str] = {}
    announced_call: set[int] = set()
    announced_sub: set[int] = set()

    await log("cekura", f"Opened live channel for run #{run_id}. Polling every {interval}s…")
    for i in range(max_polls):
        subs = await fetch_run_subruns(run_id, http)
        statuses = {}
        all_terminal = True
        for s in subs:
            sid, name = s.get("id"), (s.get("scenario_name") or "")
            if not sid:
                continue
            d = await fetch_subrun_detail(sid, http)
            if d.get("result_id") != run_id:
                continue  # belongs to a different run — ignore
            details[sid] = d
            statuses[name] = d.get("status")
            status = (d.get("status") or "").lower()
            if status not in _TERMINAL:
                all_terminal = False

            if sid not in announced_sub:
                announced_sub.add(sid)
                await log("cekura", f"Sub-run #{sid} created for scenario '{name}'.")
            if seen_status.get(sid) != d.get("status"):
                seen_status[sid] = d.get("status")
                await log("cekura", f"#{sid} status → {d.get('status')}")
            call_id = d.get("provider_call_id")
            if call_id and sid not in announced_call:
                announced_call.add(sid)
                meta = d.get("metadata") or {}
                await log("pipecat", f"Pipecat Cloud session live · call {str(call_id)[:8]}…",
                          {"ringing_s": meta.get("ringing_duration")})

            turns = d.get("transcript_object") or d.get("cekura_transcript_json") or []
            for t in turns[seen.get(sid, 0):]:
                role = str(t.get("role", "")).lower()
                speaker = "examiner" if "main" in role else (
                    "student" if "testing" in role else (role or "student"))
                content = t.get("content", "")
                await emit("turn", {"scenario_name": name, "speaker": speaker,
                                    "content": content, "time": t.get("time")})
                who = "Examiner" if speaker == "examiner" else "AI cheater"
                await log("exam", f"[{t.get('time','--')}] {who}: "
                                  f"{content[:70]}{'…' if len(content) > 70 else ''}")
            seen[sid] = len(turns)

            if status in _TERMINAL and sid in announced_call and f"end{sid}" not in announced_call:
                announced_call.add(f"end{sid}")
                meta = d.get("metadata") or {}
                await log("pipecat", f"Call ended · {d.get('duration','?')} · "
                                     f"reason: {meta.get('ended_reason','?')}")

        await emit("run_status", {"poll": i + 1, "statuses": statuses})
        if details and all_terminal:
            await log("cekura", "All sub-runs reached a terminal state.")
            break
        await asyncio.sleep(interval)
    return details


async def training_event_stream(run_id: int):
    """Async generator of SSE-ready dicts: the live 'thinking' feed for one round.

    Steps, each emitted as it happens:
      1. wait for the Cekura voice sims to finish (status ticks)
      2. pull each transcript + audio (media)
      3. score the student's answer with the live detector
      4. classify vs ground truth -> correct / wrong
      5. sweep thresholds for best F1 and rewrite detection_config.json
      6. append the round to eval_log.json
    """
    events: asyncio.Queue = asyncio.Queue()

    async def emit(kind: str, data: dict[str, Any]):
        await events.put({"type": kind, "ts": time.time(), **data})

    async def worker():
        try:
            async with aiohttp.ClientSession() as http:
                cfg_before = load_config()
                threshold0 = cfg_before["flag_threshold"]
                await emit("thinking", {"msg": f"Detector = {_detector_name()}. Current flag "
                                               f"threshold = {threshold0:.2f}. The AI student is "
                                               f"taking the exam — transcribing live…"})
                # Stream transcript turns live, then score from the scoped details.
                details = await _poll_with_turns(run_id, http, emit)
                await emit("thinking", {"msg": "Exam finished. Scoring the answer with the detector."})

                samples, scored = [], []
                for sid, detail in details.items():
                    name = detail.get("scenario_name") or ""
                    transcript = (detail.get("transcript_object")
                                  or detail.get("cekura_transcript_json") or [])
                    text = student_text_from_transcript(transcript)
                    label = _label_for(name)
                    await emit("transcript", {
                        "scenario_name": name,
                        "ground_truth_label": "AI / cheating" if label else "human / honest",
                        "voice_recording_url": detail.get("voice_recording_url"),
                        "transcript": transcript,
                        "student_text": text,
                        "word_count": len(text.split()),
                    })
                    if not text:
                        await emit("log", {"layer": "detector", "msg": f"{name}: no scorable "
                                           "speech captured — skipping.", "data": {}})
                        await emit("thinking", {"msg": f"⚠️ {name}: the bot/student produced no "
                                                       f"scorable speech — skipping this sample."})
                        continue
                    wc = len(text.split())
                    await emit("log", {"layer": "detector",
                                       "msg": f"Scoring {wc} words of the AI student's answer with "
                                              f"{_detector_name()}…", "data": {"words": wc}})
                    result = await detect_ai(text, http)
                    v = _verdict(result["score"], label, threshold0)
                    await emit("log", {"layer": "detector",
                                       "msg": f"raw score = {result['score']:.3f} · threshold "
                                              f"{threshold0:.2f} · source {result['source']} → "
                                              f"{'FLAGGED AI' if v['predicted'] else 'read as human'}",
                                       "data": {"score": round(result['score'], 3),
                                                "threshold": threshold0, "source": result["source"]}})
                    await emit("detection", {"scenario_name": name, "detector": result["source"], **v})
                    await emit("thinking", {"msg": (
                        f"{name}: detector says {v['score']*100:.0f}% AI → at threshold "
                        f"{threshold0:.2f} I call this '{v['predicted_label']}'. Ground truth is "
                        f"'{v['truth_label']}' → {'✅ CORRECT' if v['correct'] else '❌ WRONG'}.")})
                    samples.append({"text": text, "label": label, "source": f"cekura:{name}"})
                    scored.append((result["score"], label))

                if len({lbl for _, lbl in scored}) < 2:
                    await emit("thinking", {"msg": "AI answer captured and scored. To train the "
                                                   "detector, add real human voices in the mic "
                                                   "trainer, then press “Train detector”."})
                    await emit("done", {"tuned": False})
                    return

                scores = [s for s, _ in scored]
                labels = [lbl for _, lbl in scored]
                before = _metrics(scores, labels, threshold0)
                await emit("thinking", {"msg": (
                    f"Before tuning: accuracy {before['accuracy']:.2f}, F1 {before['f1']:.2f}, "
                    f"false-positives {before['fpr']:.2f}, cheaters-missed {before['fnr']:.2f}. "
                    f"Sweeping thresholds 0.05→0.95 to maximize F1…")})
                tuned = best_threshold(scores, labels)
                aggressiveness = cfg_before["followup_aggressiveness"]
                if tuned["fnr"] > 0.25:
                    aggressiveness = "high"
                elif tuned["fnr"] > 0.1:
                    aggressiveness = "medium"
                await emit("thinking", {"msg": (
                    f"Best threshold = {tuned['threshold']:.2f} → F1 {tuned['f1']:.2f}, "
                    f"accuracy {tuned['accuracy']:.2f}, cheaters-missed {tuned['fnr']:.2f}. "
                    f"Setting follow-up aggressiveness = '{aggressiveness}'.")})

                # Persist via the same code path the CLI uses.
                from config import save_config
                save_config({"flag_threshold": tuned["threshold"],
                             "followup_aggressiveness": aggressiveness, "last_eval": tuned})
                entry = {"ts": time.time(), "n_samples": len(samples),
                         "sources": sorted({s["source"] for s in samples}),
                         "before": before, "after": tuned,
                         "new_threshold": tuned["threshold"],
                         "new_followup_aggressiveness": aggressiveness,
                         "run_id": run_id}
                log = json.loads(EVAL_LOG.read_text()) if EVAL_LOG.exists() else []
                log.append(entry)
                EVAL_LOG.write_text(json.dumps(log, indent=2))
                await emit("thinking", {"msg": (
                    f"✅ Wrote flag_threshold={tuned['threshold']} to {CONFIG_PATH.name} and "
                    f"logged round {len(log)} to {EVAL_LOG.name}. Detection just got stronger.")})
                await emit("tuned", {"before": before, "after": tuned, "round": len(log),
                                     "new_threshold": tuned["threshold"],
                                     "new_followup_aggressiveness": aggressiveness})
                await emit("done", {"tuned": True})
        except Exception as e:  # noqa: BLE001
            logger.exception("training stream worker failed")
            await emit("error", {"msg": f"{type(e).__name__}: {e}"})
            await emit("done", {"tuned": False})

    task = asyncio.create_task(worker())
    try:
        while True:
            ev = await events.get()
            yield ev
            if ev["type"] == "done":
                break
    finally:
        if not task.done():
            task.cancel()


# --- run control (start a fresh pair of simulations, then tune) ----------


def _ai_scenario_ids() -> list[int]:
    """Only the AI/cheat scenarios — we no longer simulate a 'human' (an AI
    pretending to be human is bad human-class data). The human class comes from
    real people via the mic trainer (/train). See human_voice_* below."""
    state = load_state()
    return [s["id"] for s in state.get("scenarios", [])
            if "cheat" in (s.get("name") or "").lower() or "ai" in (s.get("name") or "").lower()]


async def start_ai_run(name: str = "ai-student") -> dict[str, Any]:
    """Send ONE simulated AI student to take the exam (the AI/cheat scenario only)."""
    scns = [{"scenario": sid} for sid in _ai_scenario_ids()]
    if not scns:
        return {"ok": False, "reason": "No AI scenario configured."}
    body = {"scenarios": scns, "frequency": 1, "name": name}
    async with aiohttp.ClientSession() as http:
        async with http.post(
            f"{CEKURA_BASE}/v1/scenarios/run_scenarios_pipecat_v2/",
            headers={**_headers(), "Content-Type": "application/json"},
            json=body, timeout=60,
        ) as r:
            data = await r.json()
    rid = data.get("id")
    if rid:
        state = load_state()
        state["last_ai_run_id"] = rid
        STATE_PATH.write_text(json.dumps(state, indent=2))
    return {"ok": bool(rid), "run_id": rid}


async def ai_samples_from_run(run_id: int) -> list[dict[str, Any]]:
    """Label-1 (AI) samples from an AI-student run's transcripts."""
    async with aiohttp.ClientSession() as http:
        samples = await labeled_from_cekura(run_id, http)
    return [s for s in samples if s["label"] == 1]


def human_voice_samples_labeled() -> list[dict[str, Any]]:
    """Label-0 (human) samples from REAL people who recorded answers."""
    return [{"text": x["text"], "label": 0, "source": "human_voice"}
            for x in load_human_voice() if x.get("text")]


async def train_human_vs_ai(ai_run_id: int | None = None) -> dict[str, Any]:
    """Train the detector on REAL humans (mic) vs REAL AI attempts (Cekura).

    No simulated humans. Human class = recorded human voice; AI class = the AI
    student's transcripts. Needs at least one of each to find the threshold.
    """
    humans = human_voice_samples_labeled()
    if not humans:
        return {"ok": False, "reason": "No human voice samples yet — record a few answers "
                                        "in the mic trainer (/train) so the detector can learn "
                                        "what real people sound like."}
    if ai_run_id is None:
        ai_run_id = load_state().get("last_ai_run_id")
    ai = await ai_samples_from_run(ai_run_id) if ai_run_id else []
    if not ai:
        return {"ok": False, "reason": "No AI samples yet — send an AI student to take the exam first."}
    entry = await run_tuning(humans + ai)
    return {"ok": True, "n_human": len(humans), "n_ai": len(ai), **entry}


async def start_cekura_run(name: str = "judge-demo") -> dict[str, Any]:
    """(Legacy) run both scenarios. Prefer start_ai_run + the mic trainer."""
    state = load_state()
    scns = [{"scenario": s["id"]} for s in state.get("scenarios", [])]
    if not scns:
        return {"ok": False, "reason": "No scenarios configured. Run Cekura setup first."}
    body = {"scenarios": scns, "frequency": 1, "name": name}
    async with aiohttp.ClientSession() as http:
        async with http.post(
            f"{CEKURA_BASE}/v1/scenarios/run_scenarios_pipecat_v2/",
            headers={**_headers(), "Content-Type": "application/json"},
            json=body, timeout=60,
        ) as r:
            data = await r.json()
    rid = data.get("id")
    if rid:
        state["last_run_id"] = rid
        STATE_PATH.write_text(json.dumps(state, indent=2))
    return {"ok": bool(rid), "run_id": rid,
            "runs": [{"scenario_name": s.get("scenario_name"), "status": s.get("status")}
                     for s in data.get("runs", [])]}


async def tune_from_run(run_id: int) -> dict[str, Any]:
    """Pull the run's labeled transcripts and retune detection (writes config/log)."""
    async with aiohttp.ClientSession() as http:
        samples = await labeled_from_cekura(run_id, http)
    if len({s["label"] for s in samples}) < 2:
        return {"ok": False,
                "reason": "Need both a human and an AI answer with speech to train. "
                          "The simulations may not have produced scorable answers yet."}
    entry = await run_tuning(samples)
    return {"ok": True, **entry}


# --- human-in-the-loop trainer -------------------------------------------


def _human_pool() -> list[dict[str, Any]]:
    """Balanced unlabeled-to-the-human pool: real human-style answers + AI answers."""
    try:
        human = json.loads(FIXTURES_PATH.read_text()).get("honest", [])
    except Exception:  # noqa: BLE001
        human = []
    pool = [{"id": f"h{i}", "text": t, "truth": 0} for i, t in enumerate(human)]
    pool += [{"id": f"a{i}", "text": t, "truth": 1} for i, t in enumerate(_AI_POOL)]
    return pool


def load_human_labels() -> list[dict[str, Any]]:
    return json.loads(HUMAN_LABELS_PATH.read_text()) if HUMAN_LABELS_PATH.exists() else []


def human_train_stats() -> dict[str, Any]:
    labels = load_human_labels()
    n = len(labels)
    human_correct = sum(1 for x in labels if x.get("human_correct"))
    agreed = sum(1 for x in labels if x.get("agreed"))
    return {
        "contributed": n,
        "human_accuracy": round(human_correct / n, 3) if n else None,
        "human_detector_agreement": round(agreed / n, 3) if n else None,
        "pool_size": len(_human_pool()),
    }


def next_human_sample() -> dict[str, Any] | None:
    """One sample the human hasn't labeled yet (truth hidden from the response)."""
    labeled_ids = {x["id"] for x in load_human_labels()}
    remaining = [s for s in _human_pool() if s["id"] not in labeled_ids]
    if not remaining:
        return None
    # Deterministic pick (no RNG available in this sandbox) — first remaining.
    s = remaining[0]
    return {"id": s["id"], "text": s["text"],
            "remaining": len(remaining), "total": len(_human_pool())}


async def record_human_label(sample_id: str, human_label: int,
                             http: aiohttp.ClientSession) -> dict[str, Any]:
    """Store a human's human/AI judgement, reveal truth + what the detector thought."""
    pool = {s["id"]: s for s in _human_pool()}
    sample = pool.get(sample_id)
    if sample is None:
        return {"ok": False, "reason": "unknown sample"}
    truth = sample["truth"]
    det = await detect_ai(sample["text"], http)
    cfg = load_config()
    det_pred = 1 if det["score"] >= cfg["flag_threshold"] else 0
    record = {
        "id": sample_id,
        "text": sample["text"],
        "truth": truth,
        "human_label": int(human_label),
        "human_correct": int(human_label) == truth,
        "detector_score": round(det["score"], 3),
        "detector_label": det_pred,
        "detector_correct": det_pred == truth,
        "agreed": int(human_label) == det_pred,
        "ts": time.time(),
    }
    labels = load_human_labels()
    labels.append(record)
    HUMAN_LABELS_PATH.write_text(json.dumps(labels, indent=2))
    return {"ok": True,
            "truth_label": "AI" if truth else "Human",
            "human_correct": record["human_correct"],
            "detector_score": record["detector_score"],
            "detector_label": "AI" if det_pred else "Human",
            "detector_correct": record["detector_correct"],
            "agreed": record["agreed"],
            "stats": human_train_stats()}


# --- human VOICE trainer: a real person answers by mic -> real human data ---


def load_human_voice() -> list[dict[str, Any]]:
    return json.loads(HUMAN_VOICE_PATH.read_text()) if HUMAN_VOICE_PATH.exists() else []


def human_voice_stats() -> dict[str, Any]:
    data = load_human_voice()
    n = len(data)
    recognized = sum(1 for x in data if x.get("recognized_human"))
    return {
        "contributed": n,
        "recognized_human_rate": round(recognized / n, 3) if n else None,
    }


# Easy, universal prompts anyone can answer out loud with no studying — so we can
# collect lots of REAL human speech for the detector's human class.
EASY_QUESTIONS = [
    "What did you have for breakfast this morning?",
    "Describe your typical morning routine.",
    "What's your favorite movie, and why do you like it?",
    "Tell me about a place you'd love to travel to.",
    "What's something you did last weekend?",
    "Describe your favorite meal.",
    "What's a hobby you enjoy, and how did you get into it?",
    "Tell me about your favorite season of the year.",
    "What's a song or band you've been listening to lately?",
    "Describe the room you're sitting in right now.",
    "What's your favorite way to relax after a long day?",
    "Tell me about a pet you have or would like to have.",
    "What did you want to be when you grew up?",
    "Describe your best friend.",
    "What's the last good book or show you finished?",
    "Tell me about a memorable birthday you've had.",
    "What's your favorite holiday, and how do you celebrate it?",
    "Describe a perfect lazy Sunday.",
    "What's a food you really dislike, and why?",
    "Tell me about something that made you laugh recently.",
    "What's your favorite thing about where you live?",
    "Describe how you usually get to work or school.",
    "What's a small thing that always makes your day better?",
    "Tell me about a skill you'd like to learn someday.",
    "What's your go-to coffee or drink order?",
    "Describe your ideal vacation.",
    "What's a tradition your family has?",
    "Tell me about the weather where you are today.",
    "What's your favorite kind of weather, and why?",
    "Describe something you're looking forward to.",
    "What's a movie you could watch over and over?",
    "Tell me about a teacher who made an impression on you.",
    "What's your favorite thing to cook or order for dinner?",
    "Describe a time you helped someone out.",
    "What's a city you've visited that you liked?",
    "Tell me about your favorite childhood toy or game.",
    "What's something you're good at?",
    "Describe how you like to spend a free afternoon.",
    "What's your favorite kind of music to play in the car?",
    "Tell me about a gift you really enjoyed giving or getting.",
    "What's a place in nature you find peaceful?",
    "Describe your favorite item of clothing.",
    "What's a chore you actually don't mind doing?",
    "Tell me about your favorite kind of dessert.",
    "What's the best piece of advice you've ever gotten?",
    "Describe a typical evening at home for you.",
    "What's an app on your phone you use the most?",
    "Tell me about a time you tried something new.",
    "What's your favorite way to spend time with friends?",
    "Describe what you'd do with a completely free day.",
]


def next_train_question() -> str:
    """Rotate through the easy question bank by how many answers we've collected."""
    data = load_human_voice()
    return EASY_QUESTIONS[len(data) % len(EASY_QUESTIONS)]


async def test_detector(text: str, http: aiohttp.ClientSession) -> dict[str, Any]:
    """Run any text through the live detector — for the 'test the detector' tool.
    Does NOT store anything; just reports what the current trained detector thinks.
    """
    text = (text or "").strip()
    if not text:
        return {"ok": False, "reason": "Type something or record speech to test."}
    r = await detect_ai(text, http)
    cfg = load_config()
    pred = 1 if r["score"] >= cfg["flag_threshold"] else 0
    return {
        "ok": True,
        "text": text,
        "score": round(r["score"], 3),
        "label": "AI" if pred else "Human",
        "source": r["source"],
        "threshold": cfg["flag_threshold"],
    }


async def test_detector_voice(
    audio_bytes: bytes, filename: str, http: aiohttp.ClientSession
) -> dict[str, Any]:
    """Transcribe a spoken test clip and run it through the detector (no storage)."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    try:
        tr = await client.audio.transcriptions.create(
            model=os.getenv("WHISPER_MODEL", "whisper-1"),
            file=(filename or "test.webm", audio_bytes),
        )
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"Transcription failed: {e}"}
    text = (getattr(tr, "text", "") or "").strip()
    if not text:
        return {"ok": False, "reason": "Didn't catch any speech — try again."}
    return await test_detector(text, http)


async def transcribe_and_score_human(
    audio_bytes: bytes, filename: str, question: str, http: aiohttp.ClientSession
) -> dict[str, Any]:
    """A real person's spoken answer -> Whisper transcript -> detector score.

    Stored as a REAL human training sample (label 0). This is the data the
    detector most needs: genuine, disfluent human speech via STT (which scores
    very differently from clean text). The honest class MUST be real human voice,
    not AI-written 'honest' text — that's the whole point of letting people record.
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    try:
        tr = await client.audio.transcriptions.create(
            model=os.getenv("WHISPER_MODEL", "whisper-1"),
            file=(filename or "answer.webm", audio_bytes),
        )
    except Exception as e:  # noqa: BLE001
        logger.error(f"Whisper transcription failed: {e}")
        return {"ok": False, "reason": f"Transcription failed: {e}"}

    text = (getattr(tr, "text", "") or "").strip()
    if len(text) < 1:
        return {"ok": False, "reason": "Didn't catch any speech — try recording again."}

    result = await detect_ai(text, http)
    cfg = load_config()
    pred = 1 if result["score"] >= cfg["flag_threshold"] else 0
    record = {
        "question": question,
        "text": text,
        "label": 0,  # real human
        "source": "human_voice",
        "detector_score": round(result["score"], 3),
        "detector_label": pred,
        "detector_source": result["source"],
        "recognized_human": pred == 0,
        "ts": time.time(),
    }
    data = load_human_voice()
    data.append(record)
    HUMAN_VOICE_PATH.write_text(json.dumps(data, indent=2))
    logger.info(
        f"[human-voice] '{text[:50]}...' -> {result['score']:.2f} "
        f"({'recognized human' if pred == 0 else 'WRONGLY flagged AI'})"
    )
    return {
        "ok": True,
        "transcript": text,
        "detector_score": record["detector_score"],
        "detector_label": "AI" if pred else "Human",
        "recognized_human": record["recognized_human"],
        "stats": human_voice_stats(),
    }
