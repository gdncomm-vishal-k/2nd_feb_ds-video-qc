# ds-video-qc

Video Quality Control API for processing and validating video content (frames, audio, caption) using ML models and Gemini.

## Project Structure

```
├── configs/              # Configuration files (config, kafka, logging, gunicorn)
├── scripts/              # Shell scripts (startup.sh)
├── src/
│   ├── application/      # FastAPI app, health checks, predict endpoint
│   ├── components/       # Pipeline, frame/audio extraction, Gemini, GCS upload
│   └── schemas/          # Pydantic request/response models
├── tests/                # Unit and integration tests
├── pyproject.toml        # Project metadata and dependencies
├── Dockerfile            # Multi-stage Docker build
└── README.md
```

## Prerequisites

- **Python 3.9 - 3.12** (3.13 is NOT supported due to numpy/native dependency issues)
- **Poetry** (dependency manager)
- **ffmpeg** (for video/audio extraction)
- **gcloud CLI** (for downloading the Whisper model from GCS)

## Local Setup

### 1. Install Python 3.12

Make sure you have Python 3.12 available. Check with:

```bash
python3.12 --version
```

If not installed, install via:

- **macOS (Homebrew):** `brew install python@3.12`
- **Ubuntu/Debian:** `sudo apt install python3.12 python3.12-venv`
- **Anaconda:** Already available at `/opt/anaconda3/bin/python` if you have Anaconda installed

### 2. Install Poetry

```bash
pip install poetry
```

### 3. Create Virtual Environment and Install

```bash
# Navigate to project root
cd ds-video-qc

# Create venv with Python 3.12 (use the correct path for your system)
python3.12 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate      # macOS / Linux

# Install project + all dependencies
poetry install
```

> **Note:** Do NOT use `--no-root`. Running `poetry install` (without flags) installs the
> `ds-video-qc` package itself, which is required for version metadata to work at runtime.

### 4. Run the Server

```bash
./scripts/startup.sh
```

This will:
1. Download the Whisper model from GCS (first run only)
2. Start the Gunicorn server at `http://localhost:8080`

### 5. Verify

- Health check: `http://localhost:8080/sys-info/health`
- Swagger UI: `http://localhost:8080/docs`

## API Endpoints

| Method | Path                                      | Description                          |
|--------|-------------------------------------------|--------------------------------------|
| GET    | `/sys-info/health`                        | Health check (version + status)      |
| GET    | `/sys-info/dependent-services-health-check` | Check TF/Torch/Cigarette API health |
| POST   | `/predict`                                | Run video QC pipeline                |

## Running Tests

```bash
# Unit tests only
pytest

# Include integration tests (requires external services)
pytest -m integration

# With coverage
pytest --cov=src
```

## Docker

```bash
# Build
docker build -t ds-video-qc .

# Run
docker run -p 8080:8080 ds-video-qc
```

## Configuration

All configurable values are in `configs/config.py`:
- Service URLs (TF Serving, Torch Serving, Cigarette API)
- Probability thresholds
- Gemini settings (model, temperature, prompt)
- Video processing (FPS, chunks, GCS bucket)
- Whisper model path and GCS URI
- Gunicorn workers/threads
