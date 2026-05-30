FROM dailyco/pipecat-base:latest

# Pipecat Cloud automatically uses `Dockerfile` when present. Keep this file
# aligned with `Dockerfile.pcc` so `pc cloud deploy` builds the bot image, not
# the Railway web-server image.
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

COPY ./bot.py bot.py
COPY ./bot_proctor.py bot_proctor.py
COPY ./detector.py detector.py
COPY ./exam_store.py exam_store.py
COPY ./config.py config.py
COPY ./nemotron_llm.py nemotron_llm.py
COPY ./nvidia_stt.py nvidia_stt.py
COPY ./detection_config.json detection_config.json
