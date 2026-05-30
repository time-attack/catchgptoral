#
# CatchGPT Oral Proctor — Pipecat voice-agent bot.
#
# The bot is an oral examiner. It reads questions (generated from an uploaded
# exam PDF) aloud, listens to the student's spoken answers, asks a pointed
# follow-up per question, and runs AI-content detection on every answer. Live
# suspicion scores and transcript stream to the teacher dashboard over SSE via
# the per-session event bus in exam_store.
#
# Pipeline: Gradium STT -> [user probe] -> aggregator -> OpenAI LLM ->
#           [bot probe] -> Gradium TTS -> transport out.
#
# This module is driven by proctor_server.py, which mints the WebRTC
# connection and passes the exam session_id through runner_args.body.
#

import asyncio
import os

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments, SmallWebRTCRunnerArguments
from pipecat.services.gradium.stt import GradiumSTTService
from pipecat.services.gradium.tts import GradiumTTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.workers.runner import WorkerRunner

from config import load_config
from detector import detect_ai, grade_session
from exam_store import ProctorSession, get_session

load_dotenv(override=True)


class UserAnswerProbe(FrameProcessor):
    """Captures the student's transcribed speech and scores their answers.

    Placed right after STT. Detection runs incrementally: every time the
    student's cumulative answer to the current question grows by N words
    (detect_every_words in detection_config.json, default 10), we re-score the
    FULL cumulative answer so far (10 words, then 20, then 30, ...). Scoring runs
    off the pipeline's critical path via background tasks so detection latency
    never stalls the conversation. The displayed suspicion is the score of the
    full running answer.
    """

    def __init__(self, session: ProctorSession, http_session: aiohttp.ClientSession):
        super().__init__()
        self._session = session
        self._http = http_session
        self._every = max(1, int(load_config().get("detect_every_words", 10)))
        # Strong refs to in-flight scoring tasks. Without this, asyncio only
        # holds weak refs and a fire-and-forget task can be garbage-collected
        # before it runs — which silently kills detection.
        self._tasks: set[asyncio.Task] = set()
        # Per-question accumulation state (reset when the question advances).
        self._q_index = -1
        self._committed = ""  # finalized answer text for the current question
        self._interim = ""  # latest in-progress (not yet finalized) transcript
        self._fired_buckets = 0  # how many N-word buckets we've already scored
        self._best_wc = 0  # largest word count scored (drops stale results)

    def _reset_for(self, idx: int) -> None:
        self._q_index = idx
        self._committed = ""
        self._interim = ""
        self._fired_buckets = 0
        self._best_wc = 0

    def _cumulative(self) -> str:
        return f"{self._committed} {self._interim}".strip()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        record = self._session.current_record
        if record is not None and record.index != self._q_index:
            self._reset_for(record.index)

        if isinstance(frame, InterimTranscriptionFrame) and frame.text:
            self._interim = frame.text.strip()
            self._session.emit("interim", {"role": "student", "text": self._interim})
            self._maybe_score(record)
        elif isinstance(frame, TranscriptionFrame) and frame.text and frame.text.strip():
            text = frame.text.strip()
            self._session.add_transcript("student", text)
            self._session.emit("transcript", {"role": "student", "text": text})
            if record is not None:
                record.answers.append(text)
                self._committed = self._cumulative_with_final(text)
                self._interim = ""
                # Only score once the answer is long enough (every N words). We do
                # NOT force a detection on every pause — short answers are scored
                # in full when the student clicks Done Speaking.
                self._maybe_score(record)

        await self.push_frame(frame, direction)

    def _cumulative_with_final(self, final_text: str) -> str:
        return f"{self._committed} {final_text}".strip()

    def _maybe_score(self, record, force: bool = False) -> None:
        """Fire detection when the cumulative answer crosses a new N-word bucket."""
        if record is None:
            return
        text = self._cumulative()
        wc = len(text.split())
        bucket = wc // self._every
        if force or bucket > self._fired_buckets:
            self._fired_buckets = max(self._fired_buckets, bucket)
            task = asyncio.create_task(self._score(record.index, text, wc))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _score(self, record_index: int, text: str, wc: int):
        session = self._session
        record = session.records[record_index]
        logger.debug(f"[q{record_index}] scoring {wc} words...")
        try:
            result = await detect_ai(text, self._http)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Detection error for q{record_index}: {e}")
            return

        # Too short to judge yet — wait for more words.
        if result["source"] == "too_short":
            return
        # Drop stale results: a shorter-window score that lands after a longer one.
        if wc < self._best_wc:
            return
        self._best_wc = wc

        # PROVISIONAL only — do NOT commit. The score is committed to the report
        # when the student clicks "Done Speaking" (see /done-speaking endpoint).
        record.live_score = result["score"]
        record.detector_source = result["source"]

        logger.info(
            f"[q{record_index}] live {wc} words -> score={result['score']:.2f} via {result['source']}"
        )
        session.emit(
            "live_score",
            {
                "index": record_index,
                "words": wc,
                "live_score": round(result["score"], 3),
                "detector_source": result["source"],
            },
        )


async def commit_answer(session: ProctorSession, http_session: aiohttp.ClientSession) -> dict | None:
    """Score + COMMIT the current question's answer (shared by the 'Done Speaking'
    button and the cloud auto-advance watcher). Mirrors the live scoring: re-score
    the full finalized answer; fall back to the provisional live_score if the text
    is too short. Emits the committed 'suspicion' event. Returns the commit or None.
    """
    record = session.current_record
    if record is None:
        return None
    text = record.combined_text
    source = "committed"
    if len(text.strip()) >= 1:
        result = await detect_ai(text, http_session)
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
        return None

    record.detector_source = source
    record.turn_scores.append(record.combined_score)
    logger.info(f"[q{record.index}] COMMITTED score={record.combined_score:.2f} via {source}")
    session.emit(
        "suspicion",
        {
            "index": record.index,
            "combined_score": round(record.combined_score, 3),
            "detector_source": source,
            "overall_score": round(session.overall_score, 3)
            if session.overall_score is not None
            else None,
            "question": record.to_dict(),
        },
    )
    return {"index": record.index, "combined_score": record.combined_score}


class AutoAdvanceWatcher(FrameProcessor):
    """Cloud / automated-testing path ONLY: advances the exam with no button.

    The local browser exam is button-driven ("Done Speaking"). But an automated
    caller (e.g. a Cekura simulated student) can't click — so without this the bot
    reads Q1, goes silent, and the call dies on a silence timeout. This watcher
    listens after the examiner finishes a question: once the student has given a
    real answer and then gone quiet for `silence_secs`, it commits the score and
    speaks the next question. Gated off (`auto_advance=False`) for the local UI.
    """

    def __init__(self, session, http_session, get_controller, silence_secs, min_words):
        super().__init__()
        self._session = session
        self._http = http_session
        self._get_controller = get_controller
        self._silence = silence_secs
        self._min_words = min_words
        self._bot_speaking = False
        self._spoke = False  # student gave real speech since the question was read
        self._timer: asyncio.Task | None = None
        self._advancing = False

    def _cancel(self) -> None:
        if self._timer and not self._timer.done():
            self._timer.cancel()
        self._timer = None

    def _arm(self) -> None:
        self._cancel()
        self._timer = asyncio.create_task(self._fire())

    def _record_has_answer(self) -> bool:
        record = self._session.current_record
        return bool(record and len(record.combined_text.split()) >= self._min_words)

    async def _fire(self) -> None:
        try:
            await asyncio.sleep(self._silence)
        except asyncio.CancelledError:
            return
        if self._bot_speaking or self._advancing:
            return
        record = self._session.current_record
        if record is None or not self._spoke:
            return
        if len(record.combined_text.split()) < self._min_words:
            return  # too little to be a real answer (ignore "hello?" / probes)
        self._advancing = True
        try:
            await commit_answer(self._session, self._http)
            controller = self._get_controller()
            if controller is not None:
                logger.info(f"[auto-advance] q{record.index} answered + silent — next question")
                await controller.next_question()
        except Exception as e:  # noqa: BLE001
            logger.error(f"auto-advance failed: {e}")
        finally:
            self._spoke = False
            self._advancing = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            self._cancel()  # don't advance while the examiner is talking
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            # If the stop event arrives late, don't erase an already-captured
            # answer and leave the simulated student hanging.
            self._spoke = self._record_has_answer()
            if self._spoke:
                self._arm()
        elif isinstance(frame, (InterimTranscriptionFrame, TranscriptionFrame)):
            if getattr(frame, "text", "").strip() and not self._bot_speaking:
                self._spoke = True
                self._arm()  # (re)start the silence countdown on every utterance
        await self.push_frame(frame, direction)


class BotStateProbe(FrameProcessor):
    """Emits turn-state events so the UI can show 'Examiner speaking' vs 'Your
    turn'. Placed just before the output transport, which pushes
    BotStarted/StoppedSpeakingFrame upstream when TTS audio starts/stops.
    """

    def __init__(self, session: ProctorSession):
        super().__init__()
        self._session = session

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStartedSpeakingFrame):
            self._session.emit("bot_speaking", {"speaking": True})
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._session.emit("bot_speaking", {"speaking": False})
        await self.push_frame(frame, direction)


async def _relay_forwarder(session: ProctorSession, http: aiohttp.ClientSession) -> asyncio.Task | None:
    """Stream this session's LIVE events out to the public relay so the teacher
    dashboard can watch the cloud call in real time (Cekura only delivers the
    transcript after the call ends). No-op unless LIVE_RELAY_URL is set, so the
    local browser path is unaffected.
    """
    url = os.getenv("LIVE_RELAY_URL")
    if not url:
        return None
    base = url.rstrip("/")
    channel = os.getenv("LIVE_RELAY_CHANNEL", "stress")
    token = os.getenv("LIVE_RELAY_TOKEN")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Relay-Token"] = token

    async def post(event: dict) -> None:
        try:
            async with http.post(
                f"{base}/api/live/push/{channel}", json=event, headers=headers, timeout=8
            ):
                pass
        except Exception as e:  # noqa: BLE001
            logger.debug(f"relay push failed: {e}")

    # Clear any prior run's backlog, then announce this exam.
    await post({"type": "reset"})
    await post({"type": "meta", "title": session.title, "total": len(session.questions)})

    queue = session.subscribe()

    async def pump() -> None:
        try:
            while True:
                event = await queue.get()
                await post(event)
        except asyncio.CancelledError:
            pass
        finally:
            session.unsubscribe(queue)

    logger.info(f"[relay] forwarding session {session.session_id} -> {base}/push/{channel}")
    return asyncio.create_task(pump())


def _build_stt(backend: str):
    """STT service. STT_BACKEND=gradium (default) or nvidia (Nemotron Speech)."""
    if backend == "nvidia":
        from nvidia_stt import NVidiaWebSocketSTTService

        return NVidiaWebSocketSTTService(
            url=os.environ["NVIDIA_ASR_URL"],
            strip_interim_prefix=True,
        )
    return GradiumSTTService(
        api_key=os.environ["GRADIUM_API_KEY"],
        settings=GradiumSTTService.Settings(language=Language.EN),
    )


# Per-session controllers, so the /done-speaking HTTP endpoint can advance the
# exam (the bot itself never decides when an answer is finished — the student
# does, via the button).
CONTROLLERS: dict[str, "ExamController"] = {}


class ExamController:
    """Drives the exam by voice, on demand.

    There is NO LLM and NO automatic turn-taking in the live loop: the bot reads
    a question, then stays completely silent while the student answers (one
    uninterrupted turn). The student advances by clicking "Done Speaking", which
    commits the score (server) and calls next_question() here to speak the next
    one. This is what makes the bot un-interruptible and one-turn-per-question.
    """

    def __init__(self, session: ProctorSession, worker: PipelineWorker, auto_advance: bool = False):
        self._session = session
        self._worker = worker
        self._auto = auto_advance

    async def _speak(self, text: str) -> None:
        await self._worker.queue_frames([TTSSpeakFrame(text)])

    def _announce(self, record) -> None:
        self._session.emit(
            "question",
            {
                "index": record.index,
                "total": len(self._session.questions),
                "question": record.question,
            },
        )
        self._session.add_transcript("examiner", record.question)
        self._session.emit("examiner_speech", {"role": "examiner", "text": record.question})

    async def start(self) -> None:
        # Greeting + first question in ONE utterance, so the UI shows a single
        # clean "Examiner speaking" -> "Your turn" transition (no flicker).
        n = len(self._session.questions)
        record = self._session.advance()
        if record is None:
            return
        self._announce(record)
        if self._auto:
            # Voice-only / automated caller: no button exists. Cue them to answer
            # now, and to just pause when finished — the bot advances on silence.
            await self._speak(
                f"Welcome to your oral exam. There are {n} questions. Answer each one out "
                "loud in your own words. When you finish an answer, just pause for a moment "
                f"and I'll move on. Here is question one. {record.question}"
            )
        else:
            await self._speak(
                f"Welcome to your oral exam. There are {n} questions. Answer each one out "
                "loud in your own words; take your time, and click Done Speaking when you "
                f"finish. Question 1. {record.question}"
            )

    async def next_question(self) -> None:
        record = self._session.advance()
        if record is None:
            self._session.status = "completed"
            # Grade all answers (LLM) before publishing the final report.
            await self._speak(
                "That's the last question. Give me a moment to score your exam."
            )
            try:
                await grade_session(self._session)
            except Exception as e:  # noqa: BLE001
                logger.error(f"Grading error: {e}")
            report = self._session.to_report()
            logger.info(
                f"Exam {self._session.session_id} complete: suspicion="
                f"{report['overall_score']} grade={report['overall_grade']}"
            )
            self._session.emit("completed", {"report": report})
            await self._speak("All done. Thanks for your time. You can close the window now.")
            return
        self._announce(record)
        await self._speak(f"Question {record.index + 1}. {record.question}")


async def run_bot(transport: BaseTransport, session: ProctorSession, auto_advance: bool = False):
    """Main proctor bot logic for a single exam session.

    Button-driven by default: STT + live detection while the student speaks, no
    LLM in the loop, so the bot can't barge in. Progression is via "Done Speaking".

    When `auto_advance=True` (the cloud / automated-testing path, e.g. Cekura),
    there's no button, so an AutoAdvanceWatcher moves to the next question once the
    student has answered and gone quiet. The local browser UI keeps auto_advance
    off so its one-turn-per-question, no-interruption behaviour is unchanged.
    """
    logger.info(
        f"Starting proctor bot for session {session.session_id} ({session.title}); "
        f"auto_advance={auto_advance}"
    )

    http_session = aiohttp.ClientSession()

    # --- Services ---------------------------------------------------------
    # STT_BACKEND = gradium (default) | nvidia. No LLM runs live (the exam is
    # button-driven); the LLM is used only for question generation at upload.
    stt_backend = os.getenv("STT_BACKEND", "gradium").lower()
    stt = _build_stt(stt_backend)
    tts = GradiumTTSService(
        api_key=os.environ["GRADIUM_API_KEY"],
        settings=GradiumTTSService.Settings(
            voice=os.getenv("GRADIUM_VOICE_ID", "_6Aslh2DxfmnRLmP"),
        ),
    )

    # VAD exists ONLY to segment the student's speech so Gradium flushes partial
    # transcripts (for the live transcript + detection). Because there's no LLM
    # or assistant downstream, a VAD "stop" never makes the bot respond — it just
    # finalizes a transcript fragment. Higher min_volume/confidence so faint
    # background chatter is ignored. All env-overridable.
    vad_params = VADParams(
        confidence=float(os.getenv("VAD_CONFIDENCE", "0.8")),
        start_secs=float(os.getenv("VAD_START_SECS", "0.2")),
        stop_secs=float(os.getenv("VAD_STOP_SECS", "1.0")),
        min_volume=float(os.getenv("VAD_MIN_VOLUME", "0.75")),
    )
    logger.info(
        f"Proctor (button-driven, no live LLM). STT={stt_backend}({type(stt).__name__}), "
        f"VAD stop_secs={vad_params.stop_secs} min_volume={vad_params.min_volume}"
    )

    # The aggregator is kept solely to host the VAD analyzer; its context output
    # goes nowhere (no LLM), so it can't trigger a bot response.
    context = LLMContext()
    user_aggregator, _assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=vad_params),
        ),
    )

    user_probe = UserAnswerProbe(session, http_session)
    bot_state_probe = BotStateProbe(session)

    processors = [transport.input(), stt, user_probe]
    if auto_advance:
        # No button on the automated path — advance on answer-then-silence.
        processors.append(
            AutoAdvanceWatcher(
                session,
                http_session,
                get_controller=lambda: CONTROLLERS.get(session.session_id),
                silence_secs=float(os.getenv("AUTO_ADVANCE_SILENCE_SECS", "2.5")),
                min_words=int(os.getenv("AUTO_ADVANCE_MIN_WORDS", "6")),
            )
        )
    processors += [user_aggregator, tts, bot_state_probe, transport.output()]
    pipeline = Pipeline(processors)

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
        ),
    )

    controller = ExamController(session, worker, auto_advance=auto_advance)
    CONTROLLERS[session.session_id] = controller

    # Live relay (cloud/Cekura path): stream the bot's own transcript out to the
    # teacher dashboard as it happens. Started before connect so it captures the
    # opening "connected"/question events. No-op unless LIVE_RELAY_URL is set.
    relay_task = await _relay_forwarder(session, http_session)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Student connected to session {session.session_id}")
        session.status = "in_progress"
        session.emit("connected", {"title": session.title, "total": len(session.questions)})
        await controller.start()

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Student disconnected from session {session.session_id}")
        CONTROLLERS.pop(session.session_id, None)
        if relay_task is not None:
            relay_task.cancel()
        await http_session.close()
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Entry point invoked by the WebRTC offer handler in proctor_server.py."""
    if not isinstance(runner_args, SmallWebRTCRunnerArguments):
        logger.error(f"Proctor bot only supports SmallWebRTC; got {type(runner_args)}")
        return

    body = runner_args.body or {}
    session_id = body.get("session_id") or runner_args.session_id
    session = get_session(session_id) if session_id else None
    if session is None:
        logger.error(f"No exam session found for session_id={session_id!r}; aborting bot")
        return

    krisp_filter = None
    if os.environ.get("ENV") != "local":
        try:
            from pipecat.audio.filters.krisp_viva_filter import KrispVivaFilter

            krisp_filter = KrispVivaFilter()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Krisp filter unavailable: {e}")

    webrtc_connection: SmallWebRTCConnection = runner_args.webrtc_connection
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_in_filter=krisp_filter,
            audio_out_enabled=True,
        ),
    )

    await run_bot(transport, session)
