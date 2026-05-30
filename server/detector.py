#
# CatchGPT Oral Proctor — AI-detection + exam-question generation.
#
# Two responsibilities:
#   1. detect_ai(text)        -> a 0.0-1.0 "this was AI-generated" score.
#      Primary path is Sapling's detector API; if that's unavailable we fall
#      back to a Claude-based heuristic prompt.
#   2. generate_questions(pdf_text) -> 5-8 oral exam questions derived from an
#      uploaded exam, via GPT (OpenAI) with a Claude fallback.
#

from __future__ import annotations

import json
import os
import re
from typing import Any

import aiohttp
from loguru import logger

from config import load_config

ZEROGPT_URL = "https://api.zerogpt.com/api/detect/detectText"
SAPLING_URL = "https://api.sapling.ai/api/v1/aidetect"


async def detect_ai(text: str, http_session: aiohttp.ClientSession) -> dict[str, Any]:
    """Score how likely `text` was AI-generated.

    Returns {"score": float 0-1, "source": str, "confident": bool}. Higher score
    == more likely AI-generated / read from a script.

    Primary detector is ZeroGPT (DeepAnalyse). Sapling and a Claude heuristic
    remain as fallbacks if ZeroGPT is unavailable, so detection never silently
    no-ops.
    """
    cfg = load_config()
    text = (text or "").strip()
    if len(text) < cfg["min_chars"]:
        return {"score": 0.0, "source": "too_short", "confident": False}

    if os.getenv("ZEROGPT_API_KEY"):
        try:
            return await _detect_zerogpt(text, os.environ["ZEROGPT_API_KEY"], http_session)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"ZeroGPT detection failed ({e}); falling back")

    if os.getenv("SAPLING_API_KEY"):
        try:
            return await _detect_sapling(text, os.environ["SAPLING_API_KEY"], http_session)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Sapling detection failed ({e}); falling back")

    return await _detect_claude(text)


async def _detect_zerogpt(
    text: str, key: str, http_session: aiohttp.ClientSession
) -> dict[str, Any]:
    headers = {"ApiKey": key, "Content-Type": "application/json"}
    async with http_session.post(
        ZEROGPT_URL, headers=headers, json={"input_text": text}, timeout=20
    ) as resp:
        body = await resp.text()
        if resp.status != 200:
            raise RuntimeError(f"ZeroGPT HTTP {resp.status}: {body[:200]}")
        payload = json.loads(body)
    if not payload.get("success"):
        raise RuntimeError(f"ZeroGPT error: {payload.get('message')}")
    d = payload.get("data") or {}
    # `fakePercentage` (0-100) is the headline AI score. For short text it can be
    # 0; in that case fall back to `isHuman` (0-100, 100 = fully human).
    score = float(d.get("fakePercentage", 0) or 0) / 100.0
    is_human = d.get("isHuman")
    if score == 0.0 and isinstance(is_human, (int, float)) and is_human < 100:
        score = max(0.0, (100.0 - float(is_human)) / 100.0)
    return {
        "score": max(0.0, min(1.0, score)),
        "source": "zerogpt",
        "confident": True,
    }


async def _detect_sapling(
    text: str, key: str, http_session: aiohttp.ClientSession
) -> dict[str, Any]:
    payload = {"key": key, "text": text}
    async with http_session.post(SAPLING_URL, json=payload, timeout=15) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"Sapling HTTP {resp.status}: {body[:200]}")
        data = await resp.json()
    score = float(data.get("score", 0.0))
    return {
        "score": max(0.0, min(1.0, score)),
        "source": "sapling",
        "confident": True,
        "sentence_scores": data.get("sentence_scores"),
    }


async def _detect_claude(text: str) -> dict[str, Any]:
    """Heuristic fallback: ask Claude how AI-generated the answer sounds."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("No ANTHROPIC_API_KEY for fallback detection; returning neutral score")
        return {"score": 0.5, "source": "unavailable", "confident": False}

    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=api_key)
    prompt = (
        "You are an AI-content detector for a spoken oral exam. The text below is a "
        "transcript of a student's spoken answer. Spoken, honest answers are usually "
        "informal, contain hesitations, self-corrections, fillers, and uneven structure. "
        "Answers that were read aloud from an AI-generated script tend to be unusually "
        "fluent, well-structured, comprehensive, and 'essay-like' for speech.\n\n"
        "Rate the probability that this answer was read from an AI-generated script, "
        "from 0.0 (clearly spontaneous human speech) to 1.0 (clearly read from an AI "
        "script). Respond with ONLY a JSON object: {\"score\": <float>}.\n\n"
        f"Transcript:\n{text}"
    )
    try:
        msg = await client.messages.create(
            model=os.getenv("DETECTOR_CLAUDE_MODEL", "claude-haiku-4-5-20251001"),
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text if msg.content else "{}"
        score = _extract_score(raw)
        return {"score": score, "source": "claude_heuristic", "confident": True}
    except Exception as e:  # noqa: BLE001
        logger.error(f"Claude fallback detection failed: {e}")
        return {"score": 0.5, "source": "error", "confident": False}


def _extract_score(raw: str) -> float:
    """Pull a 0-1 float out of an LLM response (JSON or bare number)."""
    try:
        obj = json.loads(raw)
        return max(0.0, min(1.0, float(obj["score"])))
    except Exception:
        m = re.search(r"[-+]?\d*\.?\d+", raw)
        if m:
            return max(0.0, min(1.0, float(m.group())))
    return 0.5


# --- exam question generation --------------------------------------------

QUESTION_SYSTEM = (
    "You are an experienced oral examiner. Given the text of an exam or study "
    "document, write {num} oral exam questions that probe genuine understanding "
    "of the material. The questions will be read aloud to a student who must "
    "answer out loud, so:\n"
    "- Each question must be answerable in 1-3 spoken sentences.\n"
    "- Favor 'why', 'how', 'explain', and 'compare' questions over fact recall.\n"
    "- Reference specific concepts from the document so a student who actually "
    "studied can shine and one reading from a script will stumble.\n"
    "- Keep each question to a single sentence, no sub-parts.\n"
    'Respond with ONLY a JSON object: {{"title": "<short exam title>", '
    '"questions": ["...", "..."]}}.'
)


async def generate_questions(pdf_text: str, num: int = 6) -> dict[str, Any]:
    """Generate oral exam questions from extracted PDF text.

    Tries OpenAI (GPT) first since that key is always configured for the bot,
    then falls back to Claude. As a last resort returns a generic question set
    so the demo never hard-fails on an upload.
    """
    excerpt = pdf_text.strip()[:12000]  # Keep prompt within budget.
    system = QUESTION_SYSTEM.format(num=num)

    if os.getenv("OPENAI_API_KEY"):
        try:
            return await _questions_openai(system, excerpt, num)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"OpenAI question generation failed ({e}); trying Claude")

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            return await _questions_claude(system, excerpt, num)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Claude question generation failed: {e}")

    logger.error("All question generators failed; returning generic fallback questions")
    return {
        "title": "Oral Exam",
        "questions": [
            "In your own words, summarize the main idea of the material you studied.",
            "What was the most important concept, and why does it matter?",
            "Explain a key term from the material as if teaching it to a classmate.",
            "How does one idea in the material connect to or depend on another?",
            "What is a real-world example or application of something you learned?",
            "What part of the material did you find most challenging, and why?",
        ][:num],
    }


async def _questions_openai(system: str, excerpt: str, num: int) -> dict[str, Any]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = await client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1"),
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Exam document text:\n\n{excerpt}"},
        ],
    )
    return _parse_questions(resp.choices[0].message.content, num)


async def _questions_claude(system: str, excerpt: str, num: int) -> dict[str, Any]:
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = await client.messages.create(
        model=os.getenv("QUESTION_CLAUDE_MODEL", "claude-sonnet-4-6"),
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": f"Exam document text:\n\n{excerpt}"}],
    )
    raw = msg.content[0].text if msg.content else "{}"
    return _parse_questions(raw, num)


def _parse_questions(raw: str | None, num: int) -> dict[str, Any]:
    raw = raw or "{}"
    # Tolerate ```json fences.
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    obj = json.loads(raw)
    questions = [q.strip() for q in obj.get("questions", []) if q and q.strip()]
    if not questions:
        raise ValueError("No questions in model response")
    return {
        "title": (obj.get("title") or "Oral Exam").strip(),
        "questions": questions[: max(num, len(questions))],
    }


# --- answer grading (end of exam) ----------------------------------------

import asyncio  # noqa: E402

GRADE_SYSTEM = (
    "You are grading a student's SPOKEN answer to an oral exam question. The "
    "answer is a speech transcript, so ignore disfluencies, filler words, and "
    "informal phrasing — grade only the correctness, completeness, and "
    "understanding shown. Be a fair but rigorous examiner.\n"
    "Return ONLY JSON: {\"score\": <0-100 int>, \"grade\": \"<letter A+..F>\", "
    '"feedback": "<one concise sentence of feedback>"}.'
)


async def grade_answer(question: str, answer: str) -> dict[str, Any] | None:
    """Grade one spoken answer with the LLM. Returns None if there's no answer."""
    answer = (answer or "").strip()
    if not answer:
        return None
    if not os.getenv("OPENAI_API_KEY"):
        return None
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    try:
        resp = await client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1"),
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": GRADE_SYSTEM},
                {"role": "user", "content": f"Question: {question}\n\nStudent's answer: {answer}"},
            ],
        )
        obj = json.loads(resp.choices[0].message.content)
        score = max(0, min(100, int(round(float(obj.get("score", 0))))))
        return {
            "score": score,
            "grade": str(obj.get("grade", _letter(score))).strip(),
            "feedback": str(obj.get("feedback", "")).strip(),
        }
    except Exception as e:  # noqa: BLE001
        logger.error(f"Grading failed: {e}")
        return None


def _letter(score: float) -> str:
    for cut, g in [(97, "A+"), (93, "A"), (90, "A-"), (87, "B+"), (83, "B"),
                   (80, "B-"), (77, "C+"), (73, "C"), (70, "C-"), (60, "D"), (0, "F")]:
        if score >= cut:
            return g
    return "F"


async def grade_session(session: Any) -> None:
    """Grade every answered-but-ungraded question on the session, concurrently.

    Idempotent: questions already graded are skipped. Mutates the records in
    place (grade, grade_score, grade_feedback).
    """
    pending = [
        r for r in session.records if r.combined_text.strip() and r.grade_score is None
    ]
    if not pending:
        return
    results = await asyncio.gather(*(grade_answer(r.question, r.combined_text) for r in pending))
    for record, res in zip(pending, results):
        if res:
            record.grade_score = res["score"]
            record.grade = res["grade"]
            record.grade_feedback = res["feedback"]
    logger.info(f"Graded {sum(1 for r in results if r)} answers for session {session.session_id}")
