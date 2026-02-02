## Dockerfile will be changed
############################
# Stage 1: Builder
############################
FROM python:3.9-slim AS builder

ARG DEBIAN_FRONTEND=noninteractive

ENV \
  POETRY_VERSION=1.2.1 \
  POETRY_NO_INTERACTION=1 \
  POETRY_VIRTUALENVS_CREATE=false \
  PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  TZ=US

WORKDIR /app

# Install build tools + ffmpeg (builder needs it for validation/tests if any)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      curl \
      build-essential \
      git \
      tzdata \
      ffmpeg \
    && pip install --no-cache-dir "poetry==$POETRY_VERSION" \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency metadata first (better cache usage)
COPY pyproject.toml poetry.lock ./

# Export main dependencies only (no dev deps)
RUN poetry export \
      --only main \
      --without-hashes \
      -f requirements.txt \
      -o requirements.txt

# Copy application source
COPY ./configs ./configs
COPY ./image_qc ./image_qc
COPY ./main_components ./main_components
COPY ./src ./src
COPY ./startup_script.sh ./startup_script.sh

############################
# Stage 2: Runtime
############################
FROM python:3.9-slim AS runtime

ARG DEBIAN_FRONTEND=noninteractive

ENV \
  PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  TZ=US

WORKDIR /app

# Install ONLY runtime OS dependencies (ffmpeg is REQUIRED)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      tzdata \
      ca-certificates \
      bash \
      ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m appuser

# Install Python dependencies
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --from=builder /app/configs ./configs
COPY --from=builder /app/image_qc ./image_qc
COPY --from=builder /app/main_components ./main_components
COPY --from=builder /app/src ./src
COPY --from=builder /app/startup_script.sh ./startup_script.sh

# Set permissions
RUN chown -R appuser:appuser /app \
    && chmod +x startup_script.sh

USER appuser

EXPOSE 8080

CMD ["bash", "startup_script.sh"]
