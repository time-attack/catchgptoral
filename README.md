# CatchGPT

Voice oral exams for teachers — with a way to **stress-test** whether the exam actually catches students who sound like they're reading ChatGPT.

Built on the [Pipecat YC Voice Agents Hackathon](https://github.com/pipecat-ai/yc-voice-agents-hackathon) starter.

Students use chatgpt for all of their tests, and one of the best most obvious solutions is to just do oral exams. However teachers cant do that at scale, 
so CatchGPT will automatically do all of these tests for you, and catch the students live. It uses Cekura to make simulated cheaters to easily detect 
what cheaters sound like. Using regular detectors alone like ZeroGPT on transcribed AI text actually passes through as human (because it cant detect the grammar
and formatting), so we use Cekura to train ZeroGPT on what the AI text would sound like, so the Ai detector could be as accurate as copy and pasting 
---

## The problem

Oral exams are supposed to be hard to cheat on. In practice, a student can keep ChatGPT open and read polished answers out loud. Teachers often don't know if their questions and follow-ups would catch that until it happens in a real class.

CatchGPT lets a teacher **run the exam by voice**, score how “AI-like” each answer sounds, and **rehearse a cheater** taking the same exam before any real student does.

---

## What to try in the demo

The demo link is right here https://catchgptoral-production.up.railway.app/



| Step | What happens |
|------|----------------|
| Upload a PDF | Questions are generated for an oral exam |
| Share / take the exam | Someone speaks answers in the browser — no typing |
| View results | Each answer gets a suspicion score |
| **Launch stress test** | A simulated cheating student calls in by voice; watch the live conversation and scores |
| **Detector lab** (`/train`) | See detection accuracy improve across training rounds |

The stress test is the main loop: a simulated student (honest or cheating) goes through the **same voice proctor** a real student would.

---

## How it works

**1. Teacher setup**  
Upload course material → GPT generates oral-exam questions → teacher shares a link.

**2. Voice exam (Pipecat + Gradium)**  
A [Pipecat](https://pipecat.ai) agent runs the conversation: [Gradium](https://gradium.ai) transcribes what the student says and speaks the examiner’s questions. The pipeline listens, decides the next question or follow-up, and responds in real time — locally in the browser for the demo, or on **Pipecat Cloud** for automated runs.

**3. Cheating detection**  
After each answer, a detector scores how likely the text is AI-generated (Sapling API with fallbacks). Scores show on the teacher dashboard.

**4. Stress-test and improve (Cekura)**  
[Cekura](https://cekura.com) places simulated callers — honest student or scripted cheater — into **real voice sessions** with the deployed proctor. We pull transcripts from those runs (labeled cheater vs honest), retune thresholds in `tune_detection.py`, and write updated settings to `detection_config.json`. Metrics accumulate in `eval_log.json` (local runs improved F1 from ~0.55 to ~0.80). Each round of simulated exams can make the detector sharper for the next one.

**5. Optional open-model path (NVIDIA on AWS)**  
The same Pipecat layout can swap the examiner to **Nemotron** on AWS (`LLM_BACKEND=nemotron`, `bot-nemotron.py`) and optionally Nemotron speech for STT (`STT_BACKEND=nvidia`) — one voice agent, two model backends.

Cloud voice sessions use **Daily** (via Pipecat Cloud) so simulated callers can connect to the deployed bot. The starter’s **Twilio** path is there for phone-based exams; this demo uses in-browser WebRTC for speed.

---

## Architecture

```
PDF upload → generated questions
       ↓
Voice session (Pipecat: listen → examine → speak)
       ↑↓ Gradium STT / TTS
       ↓
Per-answer suspicion scores → teacher dashboard
       ↓
Cekura simulated calls → labeled transcripts → detector tuning → next run
```

---

## Run locally

```bash
cd server
cp .env.example .env
uv sync
uv run proctor_server.py
```

Open **http://localhost:7860**. You need API keys in `.env` (see `.env.example`).

- Teacher UI: upload, stress test, results  
- Training history: `/training`  
- Deploy + Cekura: `server/deploy_pipecat_cloud.md`  
- Deeper docs: `server/README_PROCTOR.md`

---

## Stack

Pipecat · Gradium · Cekura · OpenAI (exam generation + default examiner) · NVIDIA Nemotron on AWS (alternate examiner/STT) · Daily (Pipecat Cloud transport) · Twilio (starter telephony) · Sapling (AI-text detection)
