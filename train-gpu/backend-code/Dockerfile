FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VENV_PATH=/opt/venv

WORKDIR /build

RUN python -m venv ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:${PATH}"

COPY requirements-runtime.txt ./
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements-runtime.txt \
    && find ${VENV_PATH} -type d -name "__pycache__" -prune -exec rm -rf {} + \
    && find ${VENV_PATH} -type f -name "*.pyc" -delete


FROM builder AS training-builder

COPY requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements-ml.txt \
    && find ${VENV_PATH} -type d -name "__pycache__" -prune -exec rm -rf {} + \
    && find ${VENV_PATH} -type f -name "*.pyc" -delete


FROM caddy:2.8-alpine AS caddy-bin


FROM python:3.11-slim AS training

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VENV_PATH=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    HOME=/tmp \
    TRAIN_MODE=true \
    CRON_MODE=false

WORKDIR /app

RUN addgroup --system app \
    && adduser --system --ingroup app app

COPY --from=training-builder /opt/venv /opt/venv
COPY src ./src
COPY ml ./ml
COPY entrypoint.sh ./entrypoint.sh
COPY cronjob.sh ./cronjob.sh

RUN chmod +x ./entrypoint.sh ./cronjob.sh \
    && find /app -type d -name "__pycache__" -prune -exec rm -rf {} +

USER app

CMD ["/app/entrypoint.sh"]


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VENV_PATH=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    HOME=/tmp \
    XDG_CONFIG_HOME=/config \
    XDG_DATA_HOME=/data \
    PORT=8000 \
    TRAIN_MODE=false \
    CRON_MODE=false \
    UVICORN_WORKERS=1

WORKDIR /app

RUN addgroup --system app \
    && adduser --system --ingroup app app

COPY --from=builder /opt/venv /opt/venv
COPY --from=caddy-bin /usr/bin/caddy /usr/bin/caddy

COPY src ./src
COPY Caddyfile /etc/caddy/Caddyfile
COPY entrypoint.sh ./entrypoint.sh
COPY cronjob.sh ./cronjob.sh

RUN chmod +x ./entrypoint.sh ./cronjob.sh \
    && find /app -type d -name "__pycache__" -prune -exec rm -rf {} +

EXPOSE 80 443

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD /bin/sh -c "if [ \"$CRON_MODE\" = \"true\" ]; then exit 0; else python -c \"import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')\"; fi"

USER app

CMD ["/app/entrypoint.sh"]
