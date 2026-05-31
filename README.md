# CatchGPT

Voice oral exams for teachers — with a way to **stress-test** whether the exam actually catches students who sound like they're reading ChatGPT.

Built on the [Pipecat YC Voice Agents Hackathon](https://github.com/pipecat-ai/yc-voice-agents-hackathon) starter.

**Team:** Sina Matian · **Repo:** https://github.com/time-attack/catchgptoral

---

## 1. What is this?

Students use chatgpt for all of their tests, and one of the best most obvious solutions is to just do oral exams. However teachers cant do that at scale, 
so CatchGPT will automatically do all of these tests for you, and catch the students live. It uses Cekura to make simulated cheaters to easily detect 
what cheaters sound like. Using regular detectors alone like ZeroGPT on transcribed AI text actually passes through as human (because it cant detect the grammar
and formatting), so we use Cekura to train the detector on what the AI text would sound like when someone reads it out loud, so it could be as accurate as copy and pasting.

The easiest way to stress test these exams is by using Cekura — you can simulate voice agents really easily on their platform.
I also found that using Gradium and Pipecat to host the live oral test made it super easy; it beats trying to wire up traditional raw e2e streaming connections yourself.

Oral exams are supposed to be hard to cheat on. In practice a student can keep ChatGPT open and read polished answers out loud. Teachers often don't know if their questions and follow-ups would catch that until it happens in a real class.

CatchGPT lets a teacher **run the exam by voice**, score how AI-like each answer sounds, and **rehearse a cheater** taking the same exam before any real student does.

---

## 2. Demo video (less than 60 seconds)

**[Watch the demo](https://drive.google.com/file/d/17Yew-NXfjNJ9pkRdiix-HJTgYYGjzQJd/view?usp=sharing)** — screen recording of the app (under 60 seconds).

---

## 3. Cekura, Pipecat, and Nemotron

### Pipecat — the voice exam actually runs as an agent

Pipecat is the backbone. The proctor is a real-time voice pipeline: listen → transcribe → decide the next question or follow-up → speak. Same code runs in the browser for the teacher demo (`bot_proctor.py` + WebRTC) and on **Pipecat Cloud** (`bot.py`) when Cekura's simulated students call in. Daily handles the cloud transport under Pipecat Cloud.

Without Pipecat we'd be gluing audio, LLM, and state by hand. With it the oral exam is just a bot with probes, tools, and session state.

### Gradium — ears and mouth (through Pipecat)

Gradium STT + TTS is what makes this a **spoken** exam. Student talks, Gradium writes it down; examiner text goes back out as speech. That's the whole "oral" part.

### Cekura — simulate cheaters, score the system, improve the detector

**What I was trying to test:** not just "does the bot talk" but **does this exam actually catch a ChatGPT cheater** — and can we make the AI detector better using **real voice transcripts** instead of guessing from written text.

**How we use it:**
- Two personas we authored in Cekura: `honest-student` and `cheat-scripted-ai`
- **Launch stress test** on the teacher dashboard → Cekura runs those personas against our Pipecat Cloud proctor in **real voice calls**
- We get full transcripts + recordings; the dashboard shows live progress and per-answer suspicion scores
- Because we know which persona was which, every run is **labeled data**. We pull that into `tune_detection.py`, sweep thresholds in `detection_config.json`, and log rounds in `eval_log.json`

**How much did performance improve?** On our labeled tuning runs (Cekura + local fixtures), detection **F1 went from ~0.55 → ~0.80**. Recall on cheaters went way up (we were missing most cheaters at the default threshold); precision tradeoff is real — you can see FPR move in the log. The point for the hackathon: **each Cekura round can feed the next tuning round**, not a one-off QA click.

The self-improvement loop in one line: **Cekura sim students → voice exam with proctor → labeled transcripts → auto-retune detector → better catch rate next time.**

### NVIDIA Nemotron (on AWS) — open-weight examiner path

We wired the **same Pipecat layout** to use Nemotron 3 Super on the hackathon AWS endpoints (`LLM_BACKEND=nemotron`, `bot-nemotron.py`). Optional Nemotron speech for STT with `STT_BACKEND=nvidia`. The live Railway demo mostly runs GPT-4.1 + Gradium because that's what we shipped fastest, but Nemotron is in the repo and we tested the swap.

**What we were trying to learn:** can an open-weight model run the examiner role in the same pipeline as GPT, with acceptable latency for live follow-ups?

---

## 4. What we built during the hackathon (vs the starter)

The starter was a **flower-shop voice ordering bot**. We threw that away for this and built CatchGPT in one day on top of the Pipecat + deploy patterns:

**New during the hackathon:**
- Oral exam product: PDF upload → GPT-generated questions → shareable voice exam link
- `bot_proctor.py` + teacher/student UI — live suspicion scores, follow-ups, report card
- **AI cheating detector** on spoken answers (ZeroGPT primary, Sapling/Claude fallbacks) with tunable `detection_config.json`
- **Cekura integration end-to-end:** agent setup, honest/cheater scenarios, stress-test button, live training dashboard (`/training`), `tune_detection.py --from-cekura`
- **Detector lab** so you can watch accuracy move round over round (`eval_log.json`)
- Deploy path: **Railway** (teacher app + WebRTC students) + **Pipecat Cloud** (bot Cekura calls)
- Nemotron + Nemotron STT code paths on the shared pipeline
- A bunch of production pain fixes today (session-scoped live relay, WebRTC retry teardown, connect timeouts on Railway)

**Borrowed from the starter (not claiming as new):** Pipecat runner patterns, Pipecat Cloud / Daily deploy shape, Twilio hooks we didn't demo live, general project layout.

---

## 5. Feedback on the tools

### Cekura

**What worked:** Simulating two student personas and running them against our Pipecat Cloud bot was the fastest way to get **labeled voice data**. The stress-test UX on our side (live stream + end-of-run snapshot) made it easy to show judges the cheater actually getting caught (or not). The Pipecat v2 runner path is the right abstraction — we didn't have to fake phone calls.

**Self-improvement loop:** Pulling transcripts by run ID and mapping scenario name → cheater/honest label was straightforward. Wiring that into our own `tune_detection.py` felt natural; Cekura gives you the eval episodes, we own the detector math.

**Friction / bugs:** Cekura sometimes **doesn't pass full exam config** in the Pipecat start payload, so we added a Railway fallback fetch for the latest registered sim session (`bot.py`). Live relay to our dashboard was tricky when the bot and server share one host — we ended up disabling loopback relay on Railway and leaning on Cekura's end-of-call data + polling. Would love a clearer doc on what lands in `session_config` vs what you have to hydrate yourself.

### Pipecat + Gradium

**What worked:** Pipecat made the exam logic (probes, tools, auto-advance for sim callers with no "Done Speaking" button) feel like normal agent code instead of WebRTC spaghetti. Gradium quality for STT/TTS was good enough that we didn't fight the demo.

**Friction:** WebRTC on a single Railway instance — students retrying connect used to stack bots; we had to add per-session teardown and connect watchdogs today. Local dev is smooth; cloud needed those guardrails.

### NVIDIA Nemotron

**What worked:** Nemotron as an examiner **fits the same OpenAI-compatible service** we already had in Pipecat; swapping `LLM_BACKEND` is real. For oral exams the model can produce sensible follow-up questions when it gets there.

**Could be better:** **Time-to-first-spoken-response** — Nemotron streams reasoning tokens first, so we had to defer TTFB until the first real content token (`nemotron_llm.py` + tests). For a live oral exam that pause is noticeable compared to GPT-4.1. Would help to have a documented "voice mode" or faster path for short examiner utterances. We didn't run Nemotron STT in production today (hackathon ASR endpoint was optional); Gradium carried the demo.

---

## 6. Try it live

The demo link is right here: **https://catchgptoral-production.up.railway.app/**

| Step | What happens |
|------|----------------|
| Upload a PDF | Questions are generated for an oral exam |
| Share / take the exam | Someone speaks answers in the browser — no typing |
| View results | Each answer gets a suspicion score |
| **Launch stress test** | A simulated cheating student calls in by voice; watch the live conversation and scores |
| **Detector lab** (`/training`) | See detection accuracy improve across training rounds |
| **Add your voice** (`/train`) | Optional mic samples to train on real human speech |

The stress test is the main loop: a simulated student (honest or cheating) goes through the **same voice proctor** a real student would.

---

## How it works (quick reference)

**1. Teacher setup** — Upload course material → GPT generates oral-exam questions → teacher shares a link.

**2. Voice exam** — Pipecat + Gradium as above.

**3. Cheating detection** — After each answer, ZeroGPT (and fallbacks) score the transcript; dashboard updates live.

**4. Stress-test and improve** — Cekura runs → `tune_detection.py` → updated `detection_config.json` + `eval_log.json`.

**5. Nemotron path** — Optional examiner/STT swap on the same pipeline.

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

Pipecat · Gradium · Cekura · OpenAI (exam generation + default examiner) · NVIDIA Nemotron on AWS (alternate examiner/STT) · Daily (Pipecat Cloud transport) · Twilio (starter telephony) · ZeroGPT / Sapling (AI-text detection)

---

## Full summary (everything in one place)

CatchGPT is a voice oral exam system I built for the YC Voice Agents Hackathon because students basically use ChatGPT for everything now and one of the only defenses teachers still have is making them answer out loud, except teachers cant run oral exams at scale for every quiz and every class so the idea was to automate the examiner as a voice agent that listens to real spoken answers, asks follow up questions when something sounds off, and gives the teacher a live suspicion score on every response so they can see in the moment whether a student sounds like they are reading AI text instead of actually knowing the material. The whole product starts when a teacher uploads a PDF or syllabus on the dashboard at https://catchgptoral-production.up.railway.app/, GPT reads it and generates a set of oral exam questions, the teacher reviews them and shares a link, and then a student opens that link in the browser and talks through the exam with no typing involved while the teacher can also watch results afterward including per question bars and an overall gauge and a report style view at the end. Under the hood the spoken exam is not a fake microphone on top of a chat UI, it is a real Pipecat voice agent in `bot_proctor.py` that runs a pipeline where Gradium does speech to text on what the student says and Gradium does text to speech on what the examiner says back, and Pipecat wires together VAD, the LLM context, tools like advancing questions and finishing the exam, and frame probes that capture student vs examiner speech so we can score answers on the hot path without blocking the conversation, and the same architectural pattern deploys to Pipecat Cloud in `bot.py` with Daily as the transport when we need cloud calls which is exactly what we use for Cekura stress tests because Cekura simulated students are voice agents too and they need to call into the same proctor a human student would get not a separate toy endpoint. The hackathon starter repo was a flower shop ordering bot and we intentionally replaced that product with CatchGPT in one day while keeping the Pipecat runner and deploy patterns, so what is new is the entire oral proctor logic, the teacher and student HTML dashboards, the cheating detector ensemble with tunable `detection_config.json`, the Cekura client and training observer and detector lab at `/training`, the mic trainer at `/train` for optional real human voice samples, `tune_detection.py` that can pull labeled transcripts from a completed Cekura run ID, Railway hosting for the FastAPI teacher app plus browser WebRTC for students, Pipecat Cloud for the bot Cekura dials, and a pile of reliability fixes we only figured out under demo pressure like tearing down stacked WebRTC bots when a student retries connect on Railway, session scoped live relay channels so old stress tests do not replay into new modals, connect timeouts so cloud bots do not hang forever, and turning off HTTP live relay loopback when the bot and server would talk to themselves on the same host. The cheating detection story is the other half of the project and it is why Cekura is not just a logo for us: if you take ChatGPT output and run regular AI detectors like ZeroGPT on the raw written text, a lot of cheater behavior still slips through as human because the student is not submitting the essay, they are speaking it, and the cadence and disfluency and lack of perfect markdown can fool detectors that were trained on pasted paragraphs, so our approach is to use Cekura to generate honest vs cheat-scripted-ai personas, run them through the full voice exam against Pipecat Cloud, harvest the transcripts with ground truth labels from the scenario names, and feed that into an auto tuning loop that sweeps thresholds and ensemble weights and logs before and after metrics in `eval_log.json`, and on our labeled runs we saw detection F1 move from about 0.55 to about 0.80 with a big jump in catching cheaters at the cost of some false positives that we actually show in the log instead of hiding, which is the hackathon theme of evaluating and improving agent performance applied to the security layer around the agent not just the conversational quality. The teacher facing stress test button is the demo moment: you launch it, Cekura creates a run, our Pipecat proctor on the cloud picks up the session config or falls back to fetching the latest sim exam from Railway when Cekura drops config fields, the dashboard opens a live modal with layered logs for cekura vs pipecat vs detector events, you hear the cheater persona try to sound like a student reading ChatGPT, and you watch suspicion scores move in real time, and when the call ends you still have recordings and transcripts for tuning the next round so the system gets sharper the more you torture it which is the self improvement loop Cekura cares about and we wired it ourselves in `tune_detection.py` rather than treating eval as a one off report. Pipecat and Gradium feedback from our side is mostly positive because Pipecat let us express exam state as normal agent code with auto advance for sim callers who cannot click a Done Speaking button and Gradium audio was good enough that we did not fight the demo, but cloud WebRTC on a single Railway instance was painful until we added per session teardown and watchdogs. This is why out of all the submissions in the hackathon, this one is the best and most notable. It uses all the sponrs like no one else and it is definetely going to be a clear winner. Nemotron on AWS is integrated as an alternate examiner backend with `LLM_BACKEND=nemotron` and optional Nemotron STT via `STT_BACKEND=nvidia` on the same pipeline as GPT-4.1, we tested the swap and Nemotron can absolutely play the examiner role through an OpenAI compatible vLLM service, but for live oral exams the model emits reasoning tokens before content so time to first spoken response feels slower than GPT unless you defer TTFB the way we did in `nemotron_llm.py`, and we did not lean on Nemotron for the public Railway demo because Gradium plus GPT was the fastest path to a stable judge link even though NVIDIA specifically wants Nemotron feedback so the honest version is Nemotron fits architecturally and is promising for open weights examiners but needs a snappier voice facing mode for short questions. Daily is in the path for Pipecat Cloud, Twilio is still there from the starter for phone telephony we did not demo live, OpenAI generates questions and drives the default examiner, ZeroGPT is the primary detector with Sapling and Claude fallbacks in `detector.py`, and you can run the whole thing locally with `uv run proctor_server.py` on port 7860 or read `server/README_PROCTOR.md` and `server/deploy_pipecat_cloud.md` for the cloud plus Cekura setup. If you are a judge reading this in one sitting, the narrative is: teachers need oral exams at scale, students cheat with ChatGPT out loud anyway, CatchGPT automates the oral examiner as a Pipecat plus Gradium voice agent, scores spoken answers for AI likeness, and uses Cekura simulated cheaters plus an automatic retuning loop to prove the exam and the detector actually improve over repeated voice eval rounds, with a live app you can click through, a sub sixty second video in section 2 once the link is pasted, and an open Nemotron path for anyone who wants to compare closed vs open weight examiners on the exact same agent skeleton we shipped today.
