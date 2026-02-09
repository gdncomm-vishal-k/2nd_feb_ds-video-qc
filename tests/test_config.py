"""Tests for config module (import and key constants)."""
import pytest

from configs.config import (
    TF_SERVING_API_HEALTH_CHECK_URL,
    TORCH_SERVING_API_HEALTH_CHECK_URL,
    CIGARETTE_API_HEALTH_CHECK_URL,
    PREDICTIONS,
    VIDEO_QC_PREDICTION_MAP,
    VIDEO_QC_TEMP_FOLDER,
    VIDEO_QC_CHUNKS,
    VIDEO_QC_FPS,
    PROB_THRESHOLD,
    BLUR_LOW_HIGH,
    IMAGE_QC_PREDICTION_BATCH_SIZE,
)


def test_health_check_urls_defined():
    assert "http" in TF_SERVING_API_HEALTH_CHECK_URL
    assert "http" in TORCH_SERVING_API_HEALTH_CHECK_URL
    assert "http" in CIGARETTE_API_HEALTH_CHECK_URL


def test_predictions_list():
    assert isinstance(PREDICTIONS, list)
    assert len(PREDICTIONS) > 0
    assert "blur" in PREDICTIONS or "nsfw" in PREDICTIONS


def test_video_qc_prediction_map():
    assert isinstance(VIDEO_QC_PREDICTION_MAP, dict)


def test_video_qc_config():
    assert VIDEO_QC_TEMP_FOLDER
    assert VIDEO_QC_CHUNKS > 0
    assert VIDEO_QC_FPS > 0


def test_prob_threshold_and_blur():
    assert isinstance(PROB_THRESHOLD, dict)
    assert isinstance(BLUR_LOW_HIGH, list)
    assert len(BLUR_LOW_HIGH) == 2


def test_batch_size():
    assert IMAGE_QC_PREDICTION_BATCH_SIZE > 0
