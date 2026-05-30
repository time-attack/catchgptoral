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

    def __init__(self, session: ProctorSession, worker: PipelineWorker):
        self._session = session
        self._worker = worker

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


async def run_bot(transport: BaseTransport, session: ProctorSession):
    """Main proctor bot logic for a single exam session.

    Button-driven: STT + live detection while the student speaks, no LLM in the
    loop, so the bot can't barge in. The VAD only segments speech for the
    transcript (Gradium flushes on pause). Progression is via "Done Speaking".
    """
    logger.info(f"Starting proctor bot for session {session.session_id} ({session.title})")

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

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_probe,
            user_aggregator,
            tts,
            bot_state_probe,
            transport.output(),
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
        ),
    )

    controller = ExamController(session, worker)
    CONTROLLERS[session.session_id] = controller

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
