############################
# Stage 1: Builder
############################
FROM python:3.12-slim AS builder

ARG DEBIAN_FRONTEND=noninteractive

ENV \
  POETRY_VERSION=2.1.2 \
  POETRY_NO_INTERACTION=1 \
  POETRY_VIRTUALENVS_CREATE=false \
  PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  TZ=US

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      curl \
      build-essential \
      git \
      tzdata \
      ffmpeg \
    && pip install --no-cache-dir "poetry==$POETRY_VERSION" \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock ./
COPY ./configs ./configs
COPY ./src ./src
COPY ./scripts ./scripts

RUN poetry install --only main


############################
# Stage 2: Runtime
############################
FROM python:3.12-slim AS runtime

ARG DEBIAN_FRONTEND=noninteractive

ENV \
  PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  TZ=US \
  PATH="/opt/google-cloud-sdk/bin:${PATH}"

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      tzdata \
      ca-certificates \
      bash \
      ffmpeg \
      curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz \
      | tar -xz -C /opt \
    && /opt/google-cloud-sdk/install.sh --quiet --usage-reporting=false

RUN useradd -m appuser

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY --from=builder /app/configs ./configs
COPY --from=builder /app/src ./src
COPY --from=builder /app/scripts ./scripts
COPY --from=builder /app/scripts/console_fetch.py ./console_fetch.py

RUN chown -R appuser:appuser /app \
    && chmod +x scripts/startup.sh

USER appuser

EXPOSE 8080

CMD ["bash", "scripts/startup.sh"]
