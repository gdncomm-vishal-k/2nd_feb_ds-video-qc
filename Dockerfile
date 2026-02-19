############################
# Stage 1: Builder
############################
FROM python:3.12-slim AS builder

ARG DEBIAN_FRONTEND=noninteractive

ENV \
  POETRY_VERSION=1.2.1 \
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

RUN poetry export \
      --only main \
      --without-hashes \
      -f requirements.txt \
      -o requirements.txt

COPY ./configs ./configs
COPY ./image_qc ./image_qc
COPY ./main_components ./main_components
COPY ./src ./src
COPY ./startup_script.sh ./startup_script.sh


############################
# Stage 2: Runtime
############################
FROM python:3.12-slim AS runtime

ARG DEBIAN_FRONTEND=noninteractive

ENV \
  PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  TZ=US

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      tzdata \
      ca-certificates \
      bash \
      ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m appuser

COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=builder /app/configs ./configs
COPY --from=builder /app/image_qc ./image_qc
COPY --from=builder /app/main_components ./main_components
COPY --from=builder /app/src ./src
COPY --from=builder /app/startup_script.sh ./startup_script.sh

RUN chown -R appuser:appuser /app \
    && chmod +x startup_script.sh

USER appuser

EXPOSE 8080

CMD ["bash", "startup_script.sh"]
