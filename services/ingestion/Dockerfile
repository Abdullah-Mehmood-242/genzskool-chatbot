# syntax=docker/dockerfile:1.7
# Build from the repo root:  docker build -f services/ingestion/Dockerfile .
ARG PYTHON_IMAGE=python:3.12-slim

FROM ${PYTHON_IMAGE} AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY services/ingestion/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

FROM ${PYTHON_IMAGE} AS runtime
RUN useradd --create-home --uid 10001 app
COPY --from=builder /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/libs:/app/service \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CONFIG_DIR=/app/config \
    PROMPTS_DIR=/app/prompts \
    SERVICE_PORT=8001
WORKDIR /app
COPY libs/common /app/libs/common
COPY config /app/config
COPY prompts /app/prompts
COPY services/ingestion/app /app/service/app
USER app
EXPOSE 8001
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/healthz', timeout=2)"
CMD ["sh", "-c", "exec uvicorn app.main:create_app --factory --host 0.0.0.0 --port ${SERVICE_PORT} --no-access-log"]
