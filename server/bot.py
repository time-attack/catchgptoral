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

import os
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
from exam_store import create_session, get_session

load_dotenv(override=True)

DEFAULT_EXAM = {
    "title": "General Knowledge Oral Exam",
    "questions": [
        "In your own words, explain a topic you studied recently and why it matters.",
        "What is the single most important concept from that topic, and why?",
        "Explain a key term from the material as if teaching it to a classmate.",
        "How does one idea you learned connect to or depend on another?",
        "Give a real-world example or application of something you learned.",
        "What part of the material did you find most challenging, and why?",
    ],
}


def _session_from_body(body: dict | None, session_id: str | None):
    """Resolve (or build) the ProctorSession for a cloud/Cekura session."""
    body = body or {}
    sid = session_id or body.get("session_id") or uuid.uuid4().hex[:12]
    existing = get_session(sid)
    if existing:
        return existing
    questions = body.get("questions") or DEFAULT_EXAM["questions"]
    title = body.get("title") or DEFAULT_EXAM["title"]
    logger.info(f"Creating cloud proctor session {sid}: {title} ({len(questions)} Qs)")
    return create_session(sid, title, questions)


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

    await run_bot(transport, session)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
