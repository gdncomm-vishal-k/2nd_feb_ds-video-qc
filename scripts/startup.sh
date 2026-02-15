#!/bin/bash

# Go to project root directory
cd "$(dirname "$0")/.."

# Create model directory
mkdir -p ./model

# Download model from GCS if not exists (paths from configs/config.py)
WHISPER_GCS_URI=$(python -c "import sys; sys.path.insert(0, '.'); from configs.config import WHISPER_GCS_URI; print(WHISPER_GCS_URI)")
WHISPER_MODEL_PATH=$(python -c "import sys; sys.path.insert(0, '.'); from configs.config import WHISPER_MODEL_PATH; print(WHISPER_MODEL_PATH)")
if [ ! -d "$WHISPER_MODEL_PATH" ]; then
    echo "Downloading Whisper model from $WHISPER_GCS_URI ..."
    gcloud storage cp -r "$WHISPER_GCS_URI" "$(dirname "$WHISPER_MODEL_PATH")/"
fi

# Start the app
echo "Starting server at http://localhost:8080"
echo "Swagger UI: http://localhost:8080/docs"
gunicorn --config configs/gunicorn_config.py src.application.app:app