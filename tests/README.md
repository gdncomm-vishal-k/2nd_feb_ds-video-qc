# Tests

## Run all tests

```bash
pytest
# or
pytest tests/ -v
```

## Run with coverage

```bash
pytest tests/ --cov=src --cov=configs --cov-report=term-missing
```

For an HTML report:

```bash
pytest tests/ --cov=src --cov=configs --cov-report=html
```

## Test layout

- **test_schemas.py** – Request/response validation (VideoQCRequest, VideoQCResponse, etc.)
- **test_config.py** – Config constants
- **test_logging.py** – `simple_logger` decorator (sync/async)
- **test_app.py** – FastAPI health and predict endpoints (mocked dependencies)
- **test_pipeline.py** – Pipeline and `delete_video_folder`
- **test_frame_extraction.py** – Video name, frame number, distinct frames, download, process (mocked)
- **test_frame_validation.py** – Audio/caption validation (mocked)
- **test_gcs_upload.py** – GCS URL helper and upload (mocked auth/session)
- **test_gemini_call.py** – Prompt builder and Gemini validation (mocked)
- **test_transcript.py** – Transcript extraction (mocked Whisper)
- **test_image_qc_clients.py** – HTTP clients for TF/Torch/Cigarette APIs
- **test_image_qc_postprocessing.py** – Postprocessing helpers (watermark, blur, logo, etc.)
- **test_image_qc_predictions.py** – Batch prediction and summarization

External services (TF serving, Torch serving, Cigarette API, Gemini, GCS, Whisper) are mocked so tests run without network or heavy dependencies.
