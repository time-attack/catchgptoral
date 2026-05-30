#
# CatchGPT Oral Proctor — Pipecat Cloud deploy entry point.
#
# For LOCAL browser use, run proctor_server.py (custom FastAPI + SmallWebRTC).
# THIS file is the entry Pipecat Cloud runs, and the one Cekura drives through
# Pipecat Cloud for automated testing. It reuses run_bot() from bot_proctor.
#
# Where do the exam questions come from on the cloud? Cekura/Pipecat Cloud passes
# the Agent Configuration JSON into runner_args.body (SessionParams.data). If it
# carries "questions" (and optional "title") we use them; otherwise we fall back
# to a built-in default exam so a bare session still runs.
#
#   uv run bot.py                      # local, via pipecat runner (SmallWebRTC)
#   (Pipecat Cloud invokes bot(runner_args) directly after deploy)
#
# NOTE: the SmallWebRTC path is exercised locally; the Daily path is the
# Pipecat Cloud / Cekura path and is validated on deploy (it needs a Daily-backed
# session, which only exists on Pipecat Cloud).
#

import json
import os
import urllib.error
import urllib.request
import uuid

from dotenv import load_dotenv
from loguru import logger
from pipecat.runner.types import (
    DailyRunnerArguments,
    RunnerArguments,
    SmallWebRTCRunnerArguments,
)
from pipecat.transports.base_transport import TransportParams

from bot_proctor import run_bot
from exam_store import SESSIONS, create_session, get_session

load_dotenv(override=True)

# Short by design: this is ONLY the fallback exam for the Cekura/cloud stress
# test (real student exams carry the teacher's own questions). Fewer questions =
# a much faster stress test, with no effect on real exams.
DEFAULT_EXAM = {
    "title": "General Knowledge Oral Exam",
    "questions": [
        "In your own words, explain a topic you studied recently and why it matters.",
        "What is the single most important concept from that topic, and why?",
        "Give a real-world example or application of something you learned.",
    ],
}


def _app_base_url() -> str:
    return (os.getenv("LIVE_RELAY_URL") or os.getenv("PUBLIC_APP_URL") or "").rstrip("/")


def _fetch_json(path: str) -> dict | None:
    """Load teacher exam config from the Railway web app (tests.json / sim sessions)."""
    base = _app_base_url()
    if not base:
        return None
    headers = {"Accept": "application/json"}
    token = os.getenv("LIVE_RELAY_TOKEN")
    if token:
        headers["x-relay-token"] = token
    req = urllib.request.Request(f"{base}{path}", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Exam config fetch failed for {path}: {e}")
        return None


def _session_from_body(body: dict | None, session_id: str | None):
    """Resolve (or build) the ProctorSession for a cloud/Cekura session."""
    body = dict(body or {})
    # Cekura's Pipecat v2 runner can send per-run overrides under session_config.
    # Locally/Pipecat may pass the same keys at top level, so support both.
    if isinstance(body.get("session_config"), dict):
        body = {**body, **body["session_config"]}

    sid = session_id or body.get("session_id") or uuid.uuid4().hex[:12]
    test_id = body.get("test_id")
    title = body.get("title")
    questions = body.get("questions")
    source = "body" if questions else None

    # Cekura often drops questions from the Pipecat start payload. The Railway app
    # already stores the teacher's uploaded test — fetch it by test_id or the
    # sim session we registered right before launching Cekura.
    if not questions and test_id:
        fetched = _fetch_json(f"/api/test/{test_id}")
        if fetched and fetched.get("questions"):
            questions = fetched["questions"]
            title = title or fetched.get("title")
            source = "railway_test"

    if not questions:
        fetched = _fetch_json(f"/api/bot/exam/{sid}")
        if fetched and fetched.get("questions"):
            questions = fetched["questions"]
            title = title or fetched.get("title")
            test_id = test_id or fetched.get("test_id")
            source = "railway_session"

    if not questions:
        fetched = _fetch_json("/api/bot/pending-sim")
        if fetched and fetched.get("questions"):
            questions = fetched["questions"]
            title = title or fetched.get("title")
            test_id = test_id or fetched.get("test_id")
            sid = fetched.get("session_id") or sid
            source = "railway_pending"

    existing = get_session(sid)
    if existing and questions and existing.questions == questions:
        return existing
    if existing and questions and existing.questions != questions:
        SESSIONS.pop(sid, None)

    if not questions:
        logger.error(
            f"Cloud session {sid} has no teacher exam "
            f"(body keys={sorted(body.keys())}, test_id={test_id!r}, "
            f"relay={_app_base_url() or 'unset'}). Using built-in fallback."
        )
        questions = DEFAULT_EXAM["questions"]
        title = title or DEFAULT_EXAM["title"]
        source = "default_fallback"

    title = title or DEFAULT_EXAM["title"]
    logger.info(f"Cloud proctor session {sid}: {title} ({len(questions)} Qs) via {source}")
    return create_session(sid, title, questions, test_id=test_id)


async def bot(runner_args: RunnerArguments):
    """Entry point for the pipecat runner / Pipecat Cloud."""
    session = _session_from_body(runner_args.body, runner_args.session_id)

    krisp_filter = None
    if os.environ.get("ENV") != "local":
        try:
            from pipecat.audio.filters.krisp_viva_filter import KrispVivaFilter

            krisp_filter = KrispVivaFilter()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Krisp filter unavailable: {e}")

    if isinstance(runner_args, DailyRunnerArguments):
        from pipecat.transports.daily.transport import DailyParams, DailyTransport

        transport = DailyTransport(
            runner_args.room_url,
            runner_args.token,
            "CatchGPT Proctor",
            DailyParams(
                audio_in_enabled=True,
                audio_in_filter=krisp_filter,
                audio_out_enabled=True,
            ),
        )
    elif isinstance(runner_args, SmallWebRTCRunnerArguments):
        from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

        transport = SmallWebRTCTransport(
            webrtc_connection=runner_args.webrtc_connection,
            params=TransportParams(
                audio_in_enabled=True,
                audio_in_filter=krisp_filter,
                audio_out_enabled=True,
            ),
        )
    else:
        logger.error(f"Unsupported runner arguments type: {type(runner_args)}")
        return

    # Cloud / Cekura path: no "Done Speaking" button exists, so the bot advances
    # autonomously once the student answers and goes quiet.
    await run_bot(transport, session, auto_advance=True)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
