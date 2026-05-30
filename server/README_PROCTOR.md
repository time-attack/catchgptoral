# CatchGPT Oral Proctor

A voice agent that turns an uploaded exam PDF into a spoken oral exam, listens to
the student's answers, asks AI-tripping follow-ups, and produces a live
"suspicion score" dashboard for teachers — with a Cekura-driven loop that makes
the AI-detection stronger over time.

Built on the Pipecat YC hackathon starter.

## Run locally (browser demo)

```bash
cd server
cp .env.example .env      # fill GRADIUM_API_KEY, OPENAI_API_KEY, SAPLING_API_KEY
uv sync
uv run proctor_server.py  # http://localhost:7860
```

Upload a PDF → review the generated questions → **Start Oral Exam** → talk to the
proctor. The dashboard shows the current question, a live transcript, per-question
suspicion bars, and an overall gauge; at the end you get a report card.

## Architecture

| File | Role |
|------|------|
| `proctor_server.py` | FastAPI: `/upload-exam`, `/`, WebRTC `/api/offer`, SSE `/events/{id}`, `/report/{id}` |
| `static/index.html` | Single-file UI: upload → questions → live dashboard → report |
| `bot_proctor.py` | Pipecat pipeline + the proctor logic. Two frame-processor probes capture student/examiner speech; `get_next_question`/`finish_exam` tools drive the exam |
| `detector.py` | AI detection (Sapling primary, Claude fallback) as a tunable ensemble + exam-question generation (GPT/Claude) |
| `exam_store.py` | In-memory session store + SSE event bus + report/scoring |
| `config.py` / `detection_config.json` | Runtime-tunable detection knobs (threshold, ensemble weights, follow-up aggressiveness) |
| `tune_detection.py` | The self-improvement loop: labeled transcripts → score → retune config → log accuracy |
| `cekura_client.py` | Cekura: create agent + honest/cheater scenarios, run, poll |
| `bot.py` | Pipecat Cloud deploy entry (reuses run_bot; Daily + SmallWebRTC) |

### Detection flow
After each spoken answer, the answer (and the running combined answer for that
question, including follow-ups) is scored 0–1 by the detector off the pipeline's
critical path. Scores stream to the dashboard via SSE. Questions at/above the
tuned `flag_threshold` are flagged; the overall score is the mean across
questions.

## Sponsors / stack
- **Pipecat** — orchestration (SmallWebRTC local, Daily/Pipecat Cloud for Cekura)
- **Gradium** — STT + TTS
- **OpenAI GPT-4.1** — examiner LLM (Responses API) + question generation
- **NVIDIA Nemotron-3-Super** (on **AWS**) — alternate examiner LLM. Toggle with
  `LLM_BACKEND=nemotron` (STT toggle: `STT_BACKEND=nvidia`, when the ASR endpoint
  is reachable)
- **Sapling** — AI-content detection
- **Cekura** — labeled student simulation + the detection-improvement loop
  (see `deploy_pipecat_cloud.md`)

## The self-improvement loop

```bash
# Local demo (no deploy needed) — proves the mechanism with real Sapling:
uv run tune_detection.py --generate 8

# With Cekura (after Pipecat Cloud deploy — see deploy_pipecat_cloud.md):
uv run cekura_client.py setup
uv run cekura_client.py run
uv run tune_detection.py --from-cekura <RUN_ID>
```

Honest class = human-style fixtures (`eval_fixtures.json`); cheater class =
GPT-generated (cheaters really use AI). The loop sweeps the threshold to maximize
F1 and raises follow-up aggressiveness when cheaters slip through, writing
`detection_config.json` and appending to `eval_log.json`. A local run improved
**F1 0.55 → 0.80** (FNR 0.62 → 0.00, FPR 0.0 → 0.5 — the precision/recall
tradeoff is real and visible in the log).

## Env vars
See `.env.example`. Required: `GRADIUM_API_KEY`, `OPENAI_API_KEY`. Recommended:
`SAPLING_API_KEY` (detection) and `CEKURA_API_KEY` (loop). For Cekura runs:
`PIPECAT_CLOUD_API_KEY` + `PIPECAT_AGENT_NAME` after deploy.
