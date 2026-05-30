#
# CatchGPT Oral Proctor — FastAPI server.
#
# Single entry point for the demo:
#   GET  /                      -> the teacher/student web UI (index.html)
#   POST /upload-exam           -> PDF in, generated questions + session_id out
#   POST /api/offer             -> SmallWebRTC signaling; starts the proctor bot
#   PATCH /api/offer            -> trickle ICE candidates
#   GET  /events/{session_id}   -> Server-Sent Events: live transcript + scores
#   GET  /report/{session_id}   -> final / in-progress report JSON
#
# Run locally:  uv run proctor_server.py   (serves on http://localhost:7860)
#

from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiohttp
import fitz  # PyMuPDF
import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from loguru import logger
from pipecat.runner.types import SmallWebRTCRunnerArguments
from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)

from bot_proctor import CONTROLLERS, bot, commit_answer
from detector import generate_questions, grade_session
from exam_store import (
    create_session,
    create_test,
    get_session,
    get_test,
    record_test_result,
)
from training_observer import (
    human_train_stats,
    human_voice_stats,
    next_human_sample,
    next_train_question,
    pending_sim,
    record_human_label,
    run_snapshot,
    start_ai_run,
    start_cekura_run,
    test_detector,
    test_detector_voice,
    train_human_vs_ai,
    training_event_stream,
    training_state,
    transcribe_and_score_human,
    tune_from_run,
)
from training_observer import (
    load_state as load_cekura_state,
)

load_dotenv(override=True)

STATIC_DIR = Path(__file__).parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"
TEACHER_HTML = STATIC_DIR / "teacher.html"
TRAINING_HTML = STATIC_DIR / "training.html"
TRAIN_HTML = STATIC_DIR / "train.html"
TEST_HTML = STATIC_DIR / "test.html"


def _env_csv(name: str, default: str = "") -> list[str]:
    return [x.strip() for x in os.getenv(name, default).split(",") if x.strip()]


def _ice_server_payloads() -> list[dict[str, Any]]:
    """ICE config used by both browser and server-side aiortc.

    SmallWebRTC works locally with host/STUN candidates, but a deployed server
    behind Railway/NAT needs TURN for reliable media relay.
    """
    payloads: list[dict[str, Any]] = []
    stun_urls = _env_csv("WEBRTC_STUN_URLS", "stun:stun.l.google.com:19302")
    if stun_urls:
        payloads.append({"urls": stun_urls})

    turn_urls = _env_csv("WEBRTC_TURN_URLS")
    if turn_urls:
        turn_server: dict[str, Any] = {"urls": turn_urls}
        username = os.getenv("WEBRTC_TURN_USERNAME")
        credential = os.getenv("WEBRTC_TURN_CREDENTIAL")
        if username:
            turn_server["username"] = username
        if credential:
            turn_server["credential"] = credential
        payloads.append(turn_server)
    elif os.getenv("ENV") != "local":
        logger.warning("WEBRTC_TURN_URLS is unset; deployed SmallWebRTC may time out on ICE.")
    return payloads


def _server_ice_servers() -> list[IceServer]:
    return [
        IceServer(
            urls=payload["urls"],
            username=payload.get("username"),
            credential=payload.get("credential"),
        )
        for payload in _ice_server_payloads()
    ]


_webrtc_handler = SmallWebRTCRequestHandler(ice_servers=_server_ice_servers())


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await _webrtc_handler.close()


app = FastAPI(title="CatchGPT Oral Proctor", lifespan=lifespan)


@app.get("/")
async def teacher_home():
    """Teacher portal: upload a PDF, stress-test, share a student link, see results."""
    if not TEACHER_HTML.exists():
        raise HTTPException(500, "teacher.html missing")
    return FileResponse(TEACHER_HTML)


@app.get("/take/{test_id}")
async def take_test(test_id: str):
    """Student link: opens the voice oral exam for a teacher's test."""
    if get_test(test_id) is None:
        raise HTTPException(404, "This exam link is invalid or expired.")
    return FileResponse(INDEX_HTML)


# --- Teacher: create a test, share it, read results ----------------------


def _question_count(value: int) -> int:
    return max(1, min(10, value))


@app.post("/api/teacher/upload")
async def teacher_upload(file: UploadFile = File(...), num_questions: int = Form(3)):
    """Teacher uploads a PDF -> generate questions -> create a shareable Test."""
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file")
    try:
        text = _extract_pdf_text(raw)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Could not read PDF: {e}") from e
    if len(text.strip()) < 50:
        raise HTTPException(400, "Could not extract enough text from this PDF.")
    num = _question_count(num_questions or int(os.getenv("NUM_QUESTIONS", "3")))
    result = await generate_questions(text, num=num)
    test = create_test(result["title"], result["questions"])
    logger.info(f"Created test {test['test_id']}: {test['title']} ({len(test['questions'])} Qs)")
    return {
        "test_id": test["test_id"],
        "title": test["title"],
        "questions": test["questions"],
        "num_questions": len(test["questions"]),
        "student_link": f"/take/{test['test_id']}",
    }


@app.get("/api/test/{test_id}")
async def api_get_test(test_id: str):
    test = get_test(test_id)
    if test is None:
        raise HTTPException(404, "Unknown test")
    return {"test_id": test_id, "title": test["title"],
            "questions": test["questions"], "num_questions": len(test["questions"])}


@app.get("/api/bot/exam/{session_id}")
async def api_bot_exam(session_id: str, request: Request):
    """Pipecat Cloud pulls the teacher's exam for a Cekura sim session."""
    if _LIVE_TOKEN and request.headers.get("x-relay-token") != _LIVE_TOKEN:
        raise HTTPException(403, "bad relay token")
    session = get_session(session_id)
    if session is None:
        raise HTTPException(404, "Unknown session")
    return {
        "session_id": session_id,
        "title": session.title,
        "questions": session.questions,
        "test_id": session.test_id,
        "num_questions": len(session.questions),
    }


@app.get("/api/bot/pending-sim")
async def api_bot_pending_sim():
    """Fallback when Cekura drops session_config — returns the latest registered sim."""
    sim = pending_sim()
    if sim is None:
        raise HTTPException(404, "No pending sim")
    return sim


@app.post("/api/test/{test_id}/start")
async def api_start_test(test_id: str):
    """A student begins the test: mint a fresh proctor session seeded from it."""
    test = get_test(test_id)
    if test is None:
        raise HTTPException(404, "Unknown test")
    session_id = uuid.uuid4().hex[:12]
    create_session(session_id, test["title"], test["questions"], test_id=test_id)
    logger.info(f"Student session {session_id} started for test {test_id}")
    return {"session_id": session_id, "title": test["title"],
            "questions": test["questions"], "num_questions": len(test["questions"])}


@app.get("/api/teacher/{test_id}/results")
async def api_test_results(test_id: str):
    test = get_test(test_id)
    if test is None:
        raise HTTPException(404, "Unknown test")
    return {"test_id": test_id, "title": test["title"],
            "num_questions": len(test["questions"]),
            "results": sorted(test.get("results", []),
                              key=lambda r: r.get("taken_at", 0), reverse=True)}


@app.post("/upload-exam")
async def upload_exam(file: UploadFile = File(...), num_questions: int = Form(3)):
    """Accept an exam PDF, extract its text, generate oral exam questions, and
    register a proctoring session."""
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file")

    try:
        text = _extract_pdf_text(raw)
    except Exception as e:  # noqa: BLE001
        logger.error(f"PDF parse failed: {e}")
        raise HTTPException(400, f"Could not read PDF: {e}") from e

    if len(text.strip()) < 50:
        raise HTTPException(
            400, "Could not extract enough text from this PDF (is it a scanned image?)."
        )

    num = _question_count(num_questions or int(os.getenv("NUM_QUESTIONS", "3")))
    logger.info(f"Generating {num} questions from {len(text)} chars of exam text")
    result = await generate_questions(text, num=num)

    session_id = uuid.uuid4().hex[:12]
    create_session(session_id, result["title"], result["questions"])
    logger.info(f"Created session {session_id}: {result['title']} ({len(result['questions'])} Qs)")

    return {
        "session_id": session_id,
        "title": result["title"],
        "questions": result["questions"],
        "num_questions": len(result["questions"]),
    }


def _extract_pdf_text(raw: bytes) -> str:
    parts = []
    with fitz.open(stream=raw, filetype="pdf") as doc:
        for page in doc:
            parts.append(page.get_text())
    return "\n".join(parts)


@app.post("/api/offer")
async def offer(request: SmallWebRTCRequest, background_tasks: BackgroundTasks):
    """WebRTC signaling. The session_id arrives in request_data and is threaded
    to the bot via runner_args.body so it loads the right exam."""
    request_data = request.request_data or {}
    session_id = request_data.get("session_id")
    if not session_id or get_session(session_id) is None:
        raise HTTPException(404, f"Unknown exam session_id: {session_id!r}")

    async def webrtc_connection_callback(connection: SmallWebRTCConnection):
        runner_args = SmallWebRTCRunnerArguments(
            webrtc_connection=connection,
            body={"session_id": session_id},
            session_id=session_id,
        )
        background_tasks.add_task(bot, runner_args)

    answer = await _webrtc_handler.handle_web_request(
        request=request,
        webrtc_connection_callback=webrtc_connection_callback,
    )
    return answer


@app.get("/api/ice-servers")
async def ice_servers():
    """Browser-side ICE config for SmallWebRTC."""
    return {"ice_servers": _ice_server_payloads()}


@app.patch("/api/offer")
async def ice_candidate(request: SmallWebRTCPatchRequest):
    await _webrtc_handler.handle_patch_request(request)
    return {"status": "success"}


@app.get("/events/{session_id}")
async def events(session_id: str):
    """Server-Sent Events stream for the live dashboard."""
    session = get_session(session_id)
    if session is None:
        raise HTTPException(404, "Unknown session")

    async def event_gen():
        # Send the current state immediately so a late-joining dashboard syncs.
        snapshot = {"type": "snapshot", "report": session.to_report()}
        yield f"data: {json.dumps(snapshot)}\n\n"

        queue = session.subscribe()
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                    yield f"data: {json.dumps(event)}\n\n"
                except TimeoutError:
                    # Heartbeat keeps proxies and the browser connection alive.
                    yield ": keep-alive\n\n"
        finally:
            session.unsubscribe(queue)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- Live transcript relay -----------------------------------------------
#
# During a Cekura stress test the proctor bot runs on Pipecat Cloud and Cekura
# only hands us the transcript when the call ENDS. But the bot itself knows every
# word live. When this server is reachable from the cloud (i.e. deployed), the
# bot POSTs each live event here and the teacher dashboard streams it (SSE) in
# real time. In-memory fan-out, no storage. No-op locally (the cloud bot can't
# reach localhost) — the dashboard then falls back to Cekura's end-of-call data.

_LIVE_CHANNELS: dict[str, set[asyncio.Queue]] = {}
_LIVE_HISTORY: dict[str, list[dict]] = {}
_LIVE_BACKLOG = int(os.getenv("LIVE_RELAY_BACKLOG", "300"))
_LIVE_TOKEN = os.getenv("LIVE_RELAY_TOKEN")


@app.post("/api/live/push/{channel}")
async def live_push(channel: str, request: Request):
    """The cloud bot posts one live event ({type, ...}); fan it out to dashboards."""
    if _LIVE_TOKEN and request.headers.get("x-relay-token") != _LIVE_TOKEN:
        raise HTTPException(403, "bad relay token")
    event = await request.json()
    hist = _LIVE_HISTORY.setdefault(channel, [])
    if event.get("type") == "reset":
        _LIVE_HISTORY[channel] = [event]
    else:
        hist.append(event)
        if len(hist) > _LIVE_BACKLOG:
            del hist[: len(hist) - _LIVE_BACKLOG]
    for q in list(_LIVE_CHANNELS.get(channel, ())):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:  # pragma: no cover
            pass
    return {"ok": True, "subscribers": len(_LIVE_CHANNELS.get(channel, ()))}


@app.get("/api/live/stream/{channel}")
async def live_stream(channel: str):
    """The teacher dashboard subscribes here (SSE) to watch the call word-by-word."""
    q: asyncio.Queue = asyncio.Queue()
    _LIVE_CHANNELS.setdefault(channel, set()).add(q)

    async def gen():
        for ev in _LIVE_HISTORY.get(channel, []):
            yield f"data: {json.dumps(ev)}\n\n"
        try:
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"data: {json.dumps(ev)}\n\n"
                except (TimeoutError, asyncio.TimeoutError):
                    yield ": keep-alive\n\n"
        finally:
            _LIVE_CHANNELS.get(channel, set()).discard(q)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.post("/done-speaking/{session_id}")
async def done_speaking(session_id: str):
    """Student clicked 'Done Speaking' — COMMIT the current question's score.

    Live detection runs every N words while speaking (provisional `live_score`);
    this is the only thing that commits a `combined_score` into the report.
    Re-scores the full finalized answer for accuracy; if the transcript hasn't
    finalized yet, falls back to the latest provisional score.
    """
    session = get_session(session_id)
    if session is None:
        raise HTTPException(404, "Unknown session")
    if session.current_record is None:
        return {"ok": False, "reason": "No active question to submit yet."}

    async with aiohttp.ClientSession() as http:
        committed = await commit_answer(session, http)
    if committed is None:
        return {"ok": False, "reason": "No answer captured yet for this question."}

    # Advance the exam: the bot speaks the next question (or the closing line).
    controller = CONTROLLERS.get(session_id)
    if controller is not None:
        await controller.next_question()

    return {"ok": True, **committed}


# --- Live Training & Detection Observatory -------------------------------


@app.get("/training")
async def training_page():
    if not TRAINING_HTML.exists():
        raise HTTPException(500, "training.html missing")
    return FileResponse(TRAINING_HTML)


@app.get("/api/training/state")
async def api_training_state():
    """Live config, full eval-log history, agent + scenarios, and last run id."""
    return JSONResponse(training_state())


@app.get("/api/training/run/{run_id}")
async def api_training_run(run_id: int):
    """Snapshot of one Cekura run: transcripts, audio, detector scores, verdicts."""
    async with aiohttp.ClientSession() as http:
        try:
            return JSONResponse(await run_snapshot(run_id, http))
        except Exception as e:  # noqa: BLE001
            logger.error(f"run_snapshot({run_id}) failed: {e}")
            raise HTTPException(502, f"Cekura fetch failed: {e}") from e


@app.get("/api/training/stream/{run_id}")
async def api_training_stream(run_id: int):
    """SSE: the live 'what it's thinking' feed for a tuning round on this run."""

    async def gen():
        async for event in training_event_stream(run_id):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/training/latest-run")
async def api_training_latest_run():
    state = load_cekura_state()
    return {"run_id": state.get("last_run_id"), "state": state}


@app.post("/api/training/run-ai")
async def api_training_run_ai(payload: dict | None = None):
    """Send one simulated AI student to take the exam (no fake-human sim)."""
    payload = payload or {}
    test_id = payload.get("test_id")
    if not test_id:
        return JSONResponse(
            {
                "ok": False,
                "reason": "No uploaded test selected. Upload a packet first, then run the AI simulator for that exact test.",
            },
            status_code=400,
        )
    test = get_test(test_id)
    if test is None:
        raise HTTPException(404, "Unknown test")
    return JSONResponse(await start_ai_run(test=test))


@app.post("/api/training/train")
async def api_training_train(payload: dict | None = None):
    """Train detector on REAL humans (mic) vs REAL AI attempts. Returns before/after."""
    ai_run_id = (payload or {}).get("ai_run_id")
    return JSONResponse(await train_human_vs_ai(ai_run_id))


@app.post("/api/training/start-run")
async def api_training_start_run():
    """(Legacy) launch both scenarios. Prefer /run-ai + the mic trainer."""
    return JSONResponse(await start_cekura_run())


@app.post("/api/training/tune/{run_id}")
async def api_training_tune(run_id: int):
    """(Legacy) train on a finished run's labeled answers; returns before/after."""
    return JSONResponse(await tune_from_run(run_id))


# --- Human-in-the-loop trainer -------------------------------------------


@app.get("/train")
async def train_page():
    if not TRAIN_HTML.exists():
        raise HTTPException(500, "train.html missing")
    return FileResponse(TRAIN_HTML)


@app.get("/api/train/next")
async def api_train_next():
    sample = next_human_sample()
    return {"sample": sample, "stats": human_train_stats()}


@app.post("/api/train/label")
async def api_train_label(payload: dict):
    sample_id = payload.get("id")
    human_label = payload.get("label")
    if sample_id is None or human_label not in (0, 1):
        raise HTTPException(400, "Need {id, label: 0|1}")
    async with aiohttp.ClientSession() as http:
        return JSONResponse(await record_human_label(sample_id, int(human_label), http))


# --- Test the detector (text or voice; no storage) -----------------------


@app.get("/test")
async def test_detector_page():
    if not TEST_HTML.exists():
        raise HTTPException(500, "test.html missing")
    return FileResponse(TEST_HTML)


@app.post("/api/detector/test")
async def api_detector_test(payload: dict):
    async with aiohttp.ClientSession() as http:
        return JSONResponse(await test_detector(payload.get("text", ""), http))


@app.post("/api/detector/test-voice")
async def api_detector_test_voice(audio: UploadFile = File(...)):
    raw = await audio.read()
    if not raw:
        raise HTTPException(400, "Empty audio")
    async with aiohttp.ClientSession() as http:
        return JSONResponse(await test_detector_voice(raw, audio.filename or "test.webm", http))


@app.get("/api/train/question")
async def api_train_question():
    """A question for the human to answer out loud (rotates through the bank)."""
    return {"question": next_train_question(), "stats": human_voice_stats()}


@app.post("/api/train/voice")
async def api_train_voice(question: str = Form(...), audio: UploadFile = File(...)):
    """A real person's spoken answer -> transcribe -> score -> store as human data."""
    raw = await audio.read()
    if not raw:
        raise HTTPException(400, "Empty audio")
    async with aiohttp.ClientSession() as http:
        return JSONResponse(
            await transcribe_and_score_human(raw, audio.filename or "answer.webm", question, http)
        )


@app.get("/report/{session_id}")
async def report(session_id: str):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(404, "Unknown session")
    # Grade any answered-but-ungraded questions (covers the early "End exam" path
    # where the bot didn't reach the completion grading step). Idempotent.
    try:
        await grade_session(session)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Grading on report failed: {e}")
    # If this attempt belongs to a teacher's shared Test, record the result so the
    # teacher's dashboard shows who has taken it and how they scored.
    if session.test_id:
        try:
            record_test_result(session.test_id, session)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Recording test result failed: {e}")
    return JSONResponse(session.to_report())


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    logger.info(f"CatchGPT Oral Proctor listening on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
