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

import aiohttp
import fitz  # PyMuPDF
import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from loguru import logger
from pipecat.runner.types import SmallWebRTCRunnerArguments
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)

from bot_proctor import CONTROLLERS, bot
from detector import detect_ai, generate_questions, grade_session
from exam_store import create_session, get_session

load_dotenv(override=True)

STATIC_DIR = Path(__file__).parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"

_webrtc_handler = SmallWebRTCRequestHandler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await _webrtc_handler.close()


app = FastAPI(title="CatchGPT Oral Proctor", lifespan=lifespan)


@app.get("/")
async def index():
    if not INDEX_HTML.exists():
        raise HTTPException(500, "index.html missing")
    return FileResponse(INDEX_HTML)


@app.post("/upload-exam")
async def upload_exam(file: UploadFile = File(...)):
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

    num = int(os.getenv("NUM_QUESTIONS", "6"))
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
    record = session.current_record
    if record is None:
        return {"ok": False, "reason": "No active question to submit yet."}

    text = record.combined_text
    source = "committed"
    if len(text.strip()) >= 1:
        async with aiohttp.ClientSession() as http:
            result = await detect_ai(text, http)
        if result["source"] != "too_short":
            record.combined_score = result["score"]
            source = result["source"]
        elif record.live_score is not None:
            record.combined_score = record.live_score
        else:
            record.combined_score = result["score"]
            source = result["source"]
    elif record.live_score is not None:
        record.combined_score = record.live_score
    else:
        return {"ok": False, "reason": "No answer captured yet for this question."}

    record.detector_source = source
    record.turn_scores.append(record.combined_score)
    committed_index = record.index
    committed_score = record.combined_score
    logger.info(
        f"[q{committed_index}] COMMITTED score={committed_score:.2f} via {source} (Done Speaking)"
    )
    session.emit(
        "suspicion",
        {
            "index": committed_index,
            "combined_score": round(committed_score, 3),
            "detector_source": source,
            "overall_score": round(session.overall_score, 3)
            if session.overall_score is not None
            else None,
            "question": record.to_dict(),
        },
    )

    # Advance the exam: the bot speaks the next question (or the closing line).
    controller = CONTROLLERS.get(session_id)
    if controller is not None:
        await controller.next_question()

    return {"ok": True, "index": committed_index, "combined_score": committed_score}


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
    return JSONResponse(session.to_report())


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    logger.info(f"CatchGPT Oral Proctor listening on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
