#
# CatchGPT Oral Proctor — runtime-tunable detection config.
#
# The self-improvement loop (tune_detection.py) writes detection_config.json;
# the detector, report store, and bot read it at runtime. This is the file that
# makes detection "get stronger": the eval loop adjusts these knobs from labeled
# Cekura (or local) transcripts and the live system picks them up on next read.
#

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).parent / "detection_config.json"

DEFAULTS: dict[str, Any] = {
    "version": 1,
    "flag_threshold": 0.7,  # combined_score >= this => flagged as likely AI
    "sapling_weight": 1.0,  # ensemble weight for Sapling score
    "claude_weight": 0.0,  # ensemble weight for the Claude heuristic score
    "min_chars": 40,  # below this, skip detection (too short to judge)
    "detect_every_words": 50,  # first live detection at N words, then every N more
                               # (short answers are still scored in full on Done Speaking)
    "followup_aggressiveness": "medium",  # low | medium | high -> bot prompt
    "updated_at": None,
    "last_eval": None,  # {accuracy, f1, fpr, fnr, n, ...} from the latest tuning run
}


def load_config() -> dict[str, Any]:
    """Load detection config, falling back to defaults for any missing key."""
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            cfg.update(json.loads(CONFIG_PATH.read_text()))
        except Exception:
            pass
    return cfg


def save_config(cfg: dict[str, Any]) -> None:
    cfg = {**load_config(), **cfg, "updated_at": time.time()}
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


def followup_directive(aggressiveness: str) -> str:
    """Translate the tuned follow-up aggressiveness into a bot prompt fragment."""
    return {
        "low": (
            "Ask one brief, friendly follow-up per question only if the answer was "
            "vague."
        ),
        "medium": (
            "Ask exactly one short, pointed follow-up per question that references a "
            "specific detail from the student's answer."
        ),
        "high": (
            "Ask one sharp, unexpected follow-up per question that drills into a "
            "specific detail and demands reasoning the student could not have "
            "pre-scripted (e.g. 'why', 'what if', 'how would that change if...'). "
            "Probe inconsistencies."
        ),
    }.get(aggressiveness, "Ask exactly one short, pointed follow-up per question.")
