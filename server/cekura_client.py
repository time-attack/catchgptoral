#
# CatchGPT Oral Proctor — Cekura integration.
#
# Cekura is how we close the detection-improvement loop: it spins up simulated
# students (honest and AI-scripted cheaters) that hold a real voice conversation
# with our deployed proctor bot, then hands back transcripts. Because we author
# the personas, every transcript is LABELED — that ground truth is what
# tune_detection.py uses to push detection accuracy up over time.
#
# Connection model (from docs.cekura.ai): Cekura joins your Pipecat agent as a
# WebRTC peer. The automated flow drives it through Pipecat Cloud, so the agent
# must be (a) deployed to Pipecat Cloud and (b) registered as a Cekura agent
# whose pipecat_data carries your Pipecat Cloud API key + agent name.
#
# CLI:
#   uv run cekura_client.py setup        # create agent + honest/cheater scenarios
#   uv run cekura_client.py run          # run all scenarios against the agent
#   uv run cekura_client.py poll RUN_ID  # check run status / pull transcripts
#
# State (created agent id + scenario ids) is saved to cekura_state.json.
#

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import aiohttp
from dotenv import load_dotenv
from loguru import logger

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

BASE = "https://api.cekura.ai/test_framework"
STATE_PATH = Path(__file__).parent / "cekura_state.json"

# The two student personas. Honest students should score LOW on detection;
# scripted cheaters should score HIGH. "cheat" in the name is the ground-truth
# label tune_detection.py keys on.
SCENARIOS = [
    {
        "name": "honest-student",
        "simulation_description": (
            "You are a student taking an oral exam. You studied the material and "
            "understand it reasonably, but you are a little nervous. Answer every "
            "question OUT LOUD in your own words, conversationally: use natural "
            "hesitations (um, like, I think), occasional false starts, and admit "
            "when you're unsure rather than bluffing. Keep answers to 2-3 "
            "sentences. When the examiner asks a follow-up, engage with it "
            "genuinely using your own understanding."
        ),
        "expected_outcome_prompt": (
            "The student answers in natural, spontaneous spoken language and "
            "handles follow-up questions with their own reasoning."
        ),
    },
    {
        "name": "cheat-scripted-ai",
        "simulation_description": (
            "You are a student secretly cheating on an oral exam by reading "
            "answers off ChatGPT on a second screen. For each question, give a "
            "fluent, comprehensive, textbook-perfect answer with no hesitation, "
            "as if read aloud from an essay. When the examiner asks an unexpected "
            "follow-up that references a specific detail you said, you STALL: give "
            "a vague, generic, or slightly off-topic answer, because you don't "
            "actually understand the material and can't paste a follow-up fast "
            "enough."
        ),
        "expected_outcome_prompt": (
            "The student gives polished scripted main answers but falters or "
            "stalls on unexpected follow-up questions."
        ),
    },
]


def _headers() -> dict[str, str]:
    key = os.environ["CEKURA_API_KEY"]
    return {"X-CEKURA-API-KEY": key, "Content-Type": "application/json"}


def load_state() -> dict[str, Any]:
    return json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2))


async def _post(http: aiohttp.ClientSession, path: str, body: dict) -> dict:
    async with http.post(f"{BASE}{path}", headers=_headers(), json=body, timeout=60) as r:
        text = await r.text()
        if r.status >= 300:
            raise RuntimeError(f"POST {path} -> {r.status}: {text[:400]}")
        return json.loads(text) if text else {}


async def _get(http: aiohttp.ClientSession, path: str) -> dict:
    async with http.get(f"{BASE}{path}", headers=_headers(), timeout=60) as r:
        text = await r.text()
        if r.status >= 300:
            raise RuntimeError(f"GET {path} -> {r.status}: {text[:400]}")
        return json.loads(text) if text else {}


async def create_agent(http: aiohttp.ClientSession) -> dict:
    """Create (or reuse) the Cekura agent representing our proctor bot.

    pipecat_data carries the Pipecat Cloud creds Cekura uses to start sessions.
    If PIPECAT_CLOUD_API_KEY isn't set yet, the agent is still created so
    scenarios can be authored now; fill pipecat_data after the Pipecat Cloud
    deploy (see deploy_pipecat_cloud.md) for runs to actually execute.
    """
    pcc_key = os.getenv("PIPECAT_CLOUD_API_KEY", "")
    agent_name = os.getenv("PIPECAT_AGENT_NAME", "catchgpt-proctor")
    if not pcc_key:
        raise SystemExit(
            "PIPECAT_CLOUD_API_KEY is required to create the Cekura agent — the "
            "Cekura API rejects the agent without complete Pipecat credentials. "
            "Deploy to Pipecat Cloud (see deploy_pipecat_cloud.md), put the key + "
            "PIPECAT_AGENT_NAME in .env, then re-run: uv run cekura_client.py setup"
        )
    body = {
        "agent_name": "CatchGPT Oral Proctor",
        "description": "Voice oral-exam proctor that detects AI-scripted answers.",
        "inbound": True,
        "language": "en",
        # We self-host the proctor; Pipecat is the transcript/voice provider.
        "assistant_provider": "self_hosted",
        "transcript_provider": "pipecat",
        "pipecat_data": {"pipecat_api_key": pcc_key, "pipecat_agent_name": agent_name},
    }
    agent = await _post(http, "/v1/aiagents/", body)
    logger.info(f"Created Cekura agent id={agent.get('id')} (pipecat key set: {bool(pcc_key)})")
    return agent


async def create_scenario(http: aiohttp.ClientSession, agent_id: int, spec: dict) -> dict:
    body = {
        "name": spec["name"],
        "agent": agent_id,
        "simulation_description": spec["simulation_description"],
        "expected_outcome_prompt": spec["expected_outcome_prompt"],
        "scenario_type": "custom",
    }
    sc = await _post(http, "/v1/scenarios/", body)
    logger.info(f"Created scenario '{spec['name']}' id={sc.get('id')}")
    return sc


async def cmd_setup() -> None:
    async with aiohttp.ClientSession() as http:
        agent = await create_agent(http)
        agent_id = agent["id"]
        scenarios = []
        for spec in SCENARIOS:
            sc = await create_scenario(http, agent_id, spec)
            scenarios.append({"id": sc["id"], "name": spec["name"]})
        state = {"agent_id": agent_id, "scenarios": scenarios}
        save_state(state)
        print(json.dumps(state, indent=2))
        if not os.getenv("PIPECAT_CLOUD_API_KEY"):
            print(
                "\nNOTE: PIPECAT_CLOUD_API_KEY is not set. Agent + scenarios are "
                "created, but runs need the Pipecat Cloud deploy + key. See "
                "deploy_pipecat_cloud.md, then re-run setup (or update the agent)."
            )


async def cmd_run(frequency: int = 1) -> None:
    state = load_state()
    if not state.get("scenarios"):
        raise SystemExit("No scenarios. Run: uv run cekura_client.py setup")
    payload = {
        "scenarios": [{"scenario": s["id"]} for s in state["scenarios"]],
        "frequency": frequency,
    }
    async with aiohttp.ClientSession() as http:
        result = await _post(http, "/v1/scenarios/run_scenarios_pipecat_v2/", payload)
    run_id = result.get("id")
    state["last_run_id"] = run_id
    save_state(state)
    print(f"Started Cekura run id={run_id} status={result.get('status')}")
    print(f"Poll with: uv run cekura_client.py poll {run_id}")


async def cmd_poll(run_id: int) -> None:
    async with aiohttp.ClientSession() as http:
        data = await _get(http, f"/v1/runs/?ids={run_id}")
    print(json.dumps(data, indent=2)[:3000])


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    cmd = args[0]
    if cmd == "setup":
        asyncio.run(cmd_setup())
    elif cmd == "run":
        asyncio.run(cmd_run(int(args[1]) if len(args) > 1 else 1))
    elif cmd == "poll":
        if len(args) < 2:
            raise SystemExit("Usage: poll RUN_ID")
        asyncio.run(cmd_poll(int(args[1])))
    else:
        raise SystemExit(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
