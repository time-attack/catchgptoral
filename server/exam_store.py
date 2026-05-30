#
# CatchGPT Oral Proctor — in-memory session store + SSE event bus.
#
# A single process holds all proctoring sessions. Each session bundles the
# exam questions, the live per-question detection records, the running
# transcript, and a fan-out event bus that the dashboard subscribes to over
# Server-Sent Events.
#
# Nothing here is persisted — this is a hackathon demo. Sessions live for the
# lifetime of the server process.
#

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import load_config

TESTS_PATH = Path(__file__).parent / "tests.json"


def flag_threshold() -> float:
    """Current suspicion flag threshold (tuned by the self-improvement loop)."""
    return float(load_config()["flag_threshold"])


@dataclass
class QuestionRecord:
    """Detection state for a single exam question.

    A question accumulates one or more spoken answer turns. While the student
    speaks we compute a PROVISIONAL `live_score` (refreshed every N words) but do
    NOT commit it. `combined_score` is the COMMITTED score, set only when the
    student clicks "Done Speaking" — that's what drives the report and gauge.
    """

    index: int
    question: str
    answers: list[str] = field(default_factory=list)
    turn_scores: list[float] = field(default_factory=list)
    combined_score: float | None = None  # committed (Done Speaking)
    live_score: float | None = None  # provisional (updates every N words)
    detector_source: str | None = None
    follow_up: str | None = None
    # Correctness grading (filled at end of exam)
    grade: str | None = None  # letter, e.g. "B+"
    grade_score: int | None = None  # 0-100
    grade_feedback: str | None = None

    @property
    def combined_text(self) -> str:
        return " ".join(a.strip() for a in self.answers if a.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "question": self.question,
            "answers": self.answers,
            "combined_text": self.combined_text,
            "turn_scores": [round(s, 3) for s in self.turn_scores],
            "combined_score": round(self.combined_score, 3)
            if self.combined_score is not None
            else None,
            "live_score": round(self.live_score, 3) if self.live_score is not None else None,
            "detector_source": self.detector_source,
            "follow_up": self.follow_up,
            "grade": self.grade,
            "grade_score": self.grade_score,
            "grade_feedback": self.grade_feedback,
            "flagged": self.combined_score is not None
            and self.combined_score >= flag_threshold(),
        }


class ProctorSession:
    """Everything we know about one oral exam, plus its live event bus."""

    def __init__(self, session_id: str, title: str, questions: list[str], test_id: str | None = None):
        self.session_id = session_id
        self.title = title
        self.questions = questions
        self.test_id = test_id  # the teacher's Test this attempt belongs to (if any)
        self.records: list[QuestionRecord] = [
            QuestionRecord(index=i, question=q) for i, q in enumerate(questions)
        ]
        self.current_index: int = -1  # No question asked yet.
        self.status: str = "created"  # created | in_progress | completed
        self.transcript: list[dict[str, Any]] = []
        self.created_at: float = time.time()

        # Fan-out event bus. Each SSE subscriber gets its own queue so a slow
        # consumer can't block the others or the bot pipeline.
        self._subscribers: list[asyncio.Queue] = []

    # --- event bus --------------------------------------------------------

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)

    def emit(self, event_type: str, data: dict[str, Any]) -> None:
        """Broadcast an event to every subscriber. Never blocks."""
        payload = {"type": event_type, "ts": time.time(), **data}
        for q in list(self._subscribers):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:  # pragma: no cover - unbounded queues
                pass

    # --- current question helpers ----------------------------------------

    @property
    def current_record(self) -> QuestionRecord | None:
        if 0 <= self.current_index < len(self.records):
            return self.records[self.current_index]
        return None

    def advance(self) -> QuestionRecord | None:
        """Move to the next question. Returns None when the exam is done."""
        if self.current_index + 1 >= len(self.records):
            self.current_index = len(self.records)
            return None
        self.current_index += 1
        self.status = "in_progress"
        return self.current_record

    # --- transcript -------------------------------------------------------

    def add_transcript(self, role: str, text: str) -> None:
        self.transcript.append({"role": role, "text": text, "ts": time.time()})

    # --- reporting --------------------------------------------------------

    def scored_records(self) -> list[QuestionRecord]:
        return [r for r in self.records if r.combined_score is not None]

    @property
    def overall_score(self) -> float | None:
        scored = self.scored_records()
        if not scored:
            return None
        return sum(r.combined_score for r in scored) / len(scored)

    def _letter(self, score: float) -> str:
        for cut, g in [(97, "A+"), (93, "A"), (90, "A-"), (87, "B+"), (83, "B"),
                       (80, "B-"), (77, "C+"), (73, "C"), (70, "C-"), (60, "D"), (0, "F")]:
            if score >= cut:
                return g
        return "F"

    def to_report(self) -> dict[str, Any]:
        overall = self.overall_score
        threshold = flag_threshold()
        flagged = [r.index for r in self.scored_records() if r.combined_score >= threshold]
        if overall is None:
            level = "unknown"
        elif overall >= threshold:
            level = "high"
        elif overall >= 0.4:
            level = "medium"
        else:
            level = "low"

        graded = [r for r in self.records if r.grade_score is not None]
        grade_score = round(sum(r.grade_score for r in graded) / len(graded), 1) if graded else None
        overall_grade = self._letter(grade_score) if grade_score is not None else None

        return {
            "session_id": self.session_id,
            "title": self.title,
            "status": self.status,
            "num_questions": len(self.questions),
            "num_scored": len(self.scored_records()),
            "overall_score": round(overall, 3) if overall is not None else None,
            "suspicion_level": level,
            "flag_threshold": threshold,
            "flagged_questions": flagged,
            "num_graded": len(graded),
            "grade_score": grade_score,
            "overall_grade": overall_grade,
            "questions": [r.to_dict() for r in self.records],
            "transcript": self.transcript,
        }


# Global session registry, keyed by session_id.
SESSIONS: dict[str, ProctorSession] = {}


def create_session(
    session_id: str, title: str, questions: list[str], test_id: str | None = None
) -> ProctorSession:
    session = ProctorSession(session_id, title, questions, test_id=test_id)
    SESSIONS[session_id] = session
    return session


def get_session(session_id: str) -> ProctorSession | None:
    return SESSIONS.get(session_id)


# --- Tests: a teacher's reusable oral exam, shareable to many students -------
#
# A Test is what the teacher creates (title + questions). Each student who opens
# the share link gets their own ProctorSession seeded from the Test; when they
# finish, a small result snapshot is recorded back onto the Test. Tests persist
# to tests.json so share links survive a server restart.


def _load_tests() -> dict[str, Any]:
    if TESTS_PATH.exists():
        try:
            return json.loads(TESTS_PATH.read_text())
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _save_tests(tests: dict[str, Any]) -> None:
    TESTS_PATH.write_text(json.dumps(tests, indent=2))


def create_test(title: str, questions: list[str]) -> dict[str, Any]:
    tests = _load_tests()
    test_id = uuid.uuid4().hex[:8]
    test = {
        "test_id": test_id,
        "title": title,
        "questions": questions,
        "created_at": time.time(),
        "results": [],
    }
    tests[test_id] = test
    _save_tests(tests)
    return test


def get_test(test_id: str) -> dict[str, Any] | None:
    return _load_tests().get(test_id)


def record_test_result(test_id: str, session: ProctorSession) -> None:
    """Append a finished student's result snapshot to the test (idempotent)."""
    tests = _load_tests()
    test = tests.get(test_id)
    if test is None:
        return
    report = session.to_report()
    if report.get("num_scored", 0) == 0:
        return  # nothing answered yet — don't record an empty attempt
    results = [r for r in test.get("results", []) if r.get("session_id") != session.session_id]
    results.append({
        "session_id": session.session_id,
        "taken_at": time.time(),
        "overall_score": report.get("overall_score"),
        "suspicion_level": report.get("suspicion_level"),
        "flagged_questions": report.get("flagged_questions", []),
        "num_scored": report.get("num_scored"),
        "grade_score": report.get("grade_score"),
        "overall_grade": report.get("overall_grade"),
    })
    test["results"] = results
    _save_tests(tests)
