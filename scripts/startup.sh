#!/bin/bash
# This makes the script exit immediately if any command fails. Without it, the script would continue even if gsutil or mkdir fails.
set -e

# Go to project root directory
cd "$(dirname "$0")/.."

# Create model directory
mkdir -p ./model

# Download model from GCS if not exists
if [ ! -d "./model/faster_whisper_turbo_v3_large" ]; then
    echo "Downloading Whisper model..."
    gsutil -m cp -r "gs://test-images-image-qc/Video_QC/models/faster_whisper_turbo_v3_large" ./model/
fi

# Start the app
echo "Starting server at http://localhost:8000"
echo "Swagger UI: http://localhost:8000/docs"
uvicorn src.application.app:app --host 0.0.0.0 --port 8000 --reload
