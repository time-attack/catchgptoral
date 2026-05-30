# Deploy to Pipecat Cloud + wire Cekura

This is the ONE part that needs your account — Pipecat Cloud requires a signup +
a browser login I can't do for you. ~5 minutes, then everything else is wired.

## 1. Sign up + install the CLI

- Create an account: https://pipecat.daily.co/sign-up
- Install + log in:
  ```bash
  uv tool install pipecat-ai-cli
  pc cloud auth login          # opens a browser
  pc cloud organizations list  # note your org name
  ```

## 2. Get your Pipecat Cloud API key

Pipecat Cloud dashboard → Settings → API Keys → create one. This is the
`PIPECAT_CLOUD_API_KEY` Cekura uses to start sessions against your agent.

## 3. Deploy the proctor bot

From `server/`:
```bash
pc cloud secrets set catchgpt-proctor-secrets --file .env --region us-east
pc cloud deploy --dockerfile Dockerfile.pcc                 # builds + deploys bot.py in us-east
```
`pcc-deploy.toml` already names the agent `catchgpt-proctor`. The deploy entry is
`bot.py` (reuses run_bot; questions come from the session's Agent Configuration
JSON, with a built-in default exam fallback).

`Dockerfile.pcc` uses Pipecat Cloud's `dailyco/pipecat-base` image on purpose.
Do not add `server/Dockerfile`: Railway treats that as the web app image and
will stop using Railpack. Also do not add a custom `CMD` for cloud deploys: the base image provides the reserved
`POST /bot` endpoint that Pipecat Cloud calls to start sessions.

## 4. Put the Pipecat Cloud creds in .env

```
PIPECAT_CLOUD_API_KEY=<the key from step 2>
PIPECAT_AGENT_NAME=catchgpt-proctor
```

## 5. Create the Cekura agent + scenarios (one command)

```bash
uv run cekura_client.py setup
```
Creates the Cekura agent (provider self_hosted + transcript_provider pipecat,
with your Pipecat Cloud creds) and two labeled scenarios: `honest-student` and
`cheat-scripted-ai`. IDs are saved to `cekura_state.json`.

## 6. Run the loop

```bash
uv run cekura_client.py run            # Cekura simulates both students vs the bot
uv run cekura_client.py poll <RUN_ID>  # wait until status is complete
uv run tune_detection.py --from-cekura <RUN_ID>   # score + retune detection
```
`tune_detection` labels each transcript by scenario (honest=0, cheat=1), scores
them with the real detector, and rewrites `detection_config.json`
(flag_threshold + follow-up aggressiveness) to maximize F1. Re-run rounds and
watch `eval_log.json` — that's "detection getting stronger" with numbers.

## Notes / honest caveats
- The Daily transport path in `bot.py` only exercises on Pipecat Cloud (a
  Daily-backed session doesn't exist locally). The local SmallWebRTC path
  (`proctor_server.py`) is tested.
- Cekura's simulated students are themselves LLM-driven; an AI-text detector can
  read AI-ish phrasing in *both* classes. The follow-up questioning (where a
  scripted student stalls and an honest one reasons) is what creates real
  separation — keep follow-up aggressiveness at medium/high for the eval.
