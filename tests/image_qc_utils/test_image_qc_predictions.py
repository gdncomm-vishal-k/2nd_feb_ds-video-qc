"""
Tests for src/components/image_qc_utils/image_qc_predictions.py.
No mocks – pure functions use real config; async API calls are integration (real services + image URLs).
"""

import pytest

from configs.config import IMAGE_QC_PREDICTION_BATCH_SIZE, PREDICTIONS, VIDEO_QC_PREDICTION_MAP
from components.image_qc_utils.image_qc_predictions import (
    call_cigarette_api,
    call_tf_serving,
    call_torch_serving,
    filter_predictions,
    get_frame_number,
    get_per_frame_results,
    group_consecutive_frames,
    merge_ocr_text_results_for_cigarette_models,
    override_cigarette_with_torch_ocr,
    predict_frames_in_batches,
    split_into_batches,
    summarize_video_qc_results,
)


# ---------------------------------------------------------------------------
# split_into_batches
# ---------------------------------------------------------------------------
def test_split_into_batches():
    items = list(range(10))
    batches = list(split_into_batches(items, 3))
    assert batches == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]


def test_split_into_batches_empty():
    assert list(split_into_batches([], IMAGE_QC_PREDICTION_BATCH_SIZE)) == []


# ---------------------------------------------------------------------------
# get_frame_number
# ---------------------------------------------------------------------------
def test_get_frame_number():
    assert get_frame_number("frame_0005.jpg") == 5
    assert get_frame_number("frame_123.jpg") == 123
    assert get_frame_number("path/frame_1.png") == 1


def test_get_frame_number_no_match():
    assert get_frame_number("other.jpg") == 0


# ---------------------------------------------------------------------------
# group_consecutive_frames
# ---------------------------------------------------------------------------
def test_group_consecutive_frames():
    assert group_consecutive_frames([1, 2, 3, 5, 9]) == ["1-3", "5", "9"]
    assert group_consecutive_frames([1]) == ["1"]
    assert group_consecutive_frames([1, 2]) == ["1-2"]


def test_group_consecutive_frames_empty():
    assert group_consecutive_frames([]) == []


def test_group_consecutive_frames_unsorted():
    assert group_consecutive_frames([5, 1, 3, 2]) == ["1-3", "5"]


# ---------------------------------------------------------------------------
# merge_ocr_text_results_for_cigarette_models (same logic as postprocessing)
# ---------------------------------------------------------------------------
def test_merge_ocr_text_results_for_cigarette_models():
    assert merge_ocr_text_results_for_cigarette_models([], None, None) is None
    assert merge_ocr_text_results_for_cigarette_models([0], None, False) == 0
    assert merge_ocr_text_results_for_cigarette_models([1], True, True) == 100


# ---------------------------------------------------------------------------
# override_cigarette_with_torch_ocr
# ---------------------------------------------------------------------------
def test_override_cigarette_with_torch_ocr():
    batch_results = [
        {
            "torch_serving": (
                {},  # dict not used in override
                [50, 60],  # cigarette_ocr_probs per frame
                [],
                [],
            ),
            "tf_serving": {},
            "cigarette": [
                [{"predictionType": "cigarette_prediction", "confidence": 0, "ocr_flag": True}],
                [{"predictionType": "cigarette_prediction", "confidence": 0, "ocr_flag": None}],
            ],
        }
    ]
    result = override_cigarette_with_torch_ocr(batch_results)
    assert result is batch_results
    assert result[0]["cigarette"][0][0]["confidence"] == 100  # [50], True, True -> 100
    assert result[0]["cigarette"][1][0]["confidence"] == 100  # [60], 0, None -> 100 (any(ocr) and ocr_flag None)


# ---------------------------------------------------------------------------
# filter_predictions
# ---------------------------------------------------------------------------
def test_filter_predictions():
    frame_result = {
        "torch_serving": {
            "text": {"predictionType": "text_predictions", "present": True, "confidence": 80},
            "logo": {"predictionType": "logo_predictions", "present": False, "confidence": 0},
        },
        "tf_serving": {
            "wtmk": {"predictionType": "watermark_predictions", "present": False, "confidence": 0},
        },
        "cigarette_service": [
            {"predictionType": "cigarette_prediction", "confidence": 0},
        ],
    }
    filtered = filter_predictions(frame_result, PREDICTIONS, VIDEO_QC_PREDICTION_MAP)
    assert "text_predictions" in filtered["torch_serving"] or "text" in filtered["torch_serving"]
    assert filtered["torch_serving"]["text"]["predictionType"] == "text_predictions"
    assert "logo_predictions" in filtered["torch_serving"] or "logo" in filtered["torch_serving"]
    assert len(filtered["cigarette_service"]) <= len(frame_result["cigarette_service"])


# ---------------------------------------------------------------------------
# get_per_frame_results
# ---------------------------------------------------------------------------
def test_get_per_frame_results():
    frame_urls = ["http://example.com/frame_1.jpg", "http://example.com/frame_2.jpg"]
    torch_dict = {
        "text": [{"predictionType": "text_predictions", "present": False}, {"predictionType": "text_predictions", "present": False}],
        "logo": [{"predictionType": "logo_predictions", "present": False}, {"predictionType": "logo_predictions", "present": False}],
        "keras_logo": [{"predictionType": "pharma_prescription", "present": False}, {"predictionType": "pharma_prescription", "present": False}],
        "narkotika_logo": [{"predictionType": "pharma_banned", "present": False}, {"predictionType": "pharma_banned", "present": False}],
    }
    batch_results = [
        {
            "torch_serving": (torch_dict, [None, None], [None, None], [None, None]),
            "tf_serving": {
                "wtmk": [{"predictionType": "watermark_predictions", "present": False}, {"predictionType": "watermark_predictions", "present": False}],
                "nsfw": [{"predictionType": "nsfw_predictions", "present": False}, {"predictionType": "nsfw_predictions", "present": False}],
                "blur": [{"predictionType": "blur_predictions", "present": False}, {"predictionType": "blur_predictions", "present": False}],
            },
            "cigarette": [
                [{"predictionType": "cigarette_prediction", "confidence": 0}],
                [{"predictionType": "cigarette_prediction", "confidence": 0}],
            ],
        }
    ]
    per_frame = get_per_frame_results(frame_urls, batch_results)
    assert len(per_frame) == 2
    assert frame_urls[0] in per_frame and frame_urls[1] in per_frame
    for url, data in per_frame.items():
        assert "torch_serving" in data and "tf_serving" in data and "cigarette_service" in data


# ---------------------------------------------------------------------------
# summarize_video_qc_results
# ---------------------------------------------------------------------------
def test_summarize_video_qc_results():
    per_frame_results = {
        "http://example.com/frame_1.jpg": {
            "torch_serving": {"text": {"predictionType": "text_predictions", "present": True}},
            "tf_serving": {"blur": {"predictionType": "blur_predictions", "present": True}},
            "cigarette_service": [],
        },
        "http://example.com/frame_3.jpg": {
            "torch_serving": {},
            "tf_serving": {},
            "cigarette_service": [{"predictionType": "cigarette_prediction", "confidence": 60}],
        },
    }
    summary = summarize_video_qc_results(per_frame_results)
    assert set(summary.keys()) == set(PREDICTIONS)
    for pred in PREDICTIONS:
        assert isinstance(summary[pred], list)
    # text_predictions and blur should have frame refs; cigarette may have
    assert any(len(summary[p]) > 0 for p in PREDICTIONS)


def test_summarize_video_qc_results_with_url_mapping():
    per_frame_results = {
        "http://gcs/frame_0001.jpg": {
            "torch_serving": {"text": {"predictionType": "text_predictions", "present": True}},
            "tf_serving": {},
            "cigarette_service": [],
        },
    }
    url_mapping = {"http://gcs/frame_0001.jpg": [1, 2, 3]}
    summary = summarize_video_qc_results(per_frame_results, url_mapping)
    assert set(summary.keys()) == set(PREDICTIONS)
    # text_predictions should include 1, 2, 3 from mapping
    assert summary["text_predictions"] == ["1-3"]


# ---------------------------------------------------------------------------
# Async API calls – empty list returns error/fallback without hitting services
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_call_torch_serving_empty_list():
    """call_torch_serving([]) returns error_resp structure without calling API (empty lists)."""
    result = await call_torch_serving([])
    assert result is not None
    torch_dict, cigar, alcohol, guns = result
    assert "text" in torch_dict and "logo" in torch_dict
    assert torch_dict["text"] == []  # empty list when image_path_list is empty
    assert len(cigar) == 0 and len(alcohol) == 0 and len(guns) == 0


@pytest.mark.asyncio
async def test_call_tf_serving_empty_list():
    """call_tf_serving([]) returns error_resp structure without calling API (empty lists)."""
    result = await call_tf_serving([])
    assert result is not None
    assert "wtmk" in result and "nsfw" in result and "blur" in result
    assert result["wtmk"] == []  # empty list when image_path_list is empty


@pytest.mark.asyncio
async def test_predict_frames_in_batches_empty_raises():
    """predict_frames_in_batches([]) raises RuntimeError."""
    with pytest.raises(RuntimeError) as exc_info:
        await predict_frames_in_batches([])
    assert "No frame URLs" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Async API calls – integration (real image URLs; set SAMPLE_IMAGE_URLS if needed)
# ---------------------------------------------------------------------------
SAMPLE_IMAGE_URLS = []  # Set to e.g. ["https://storage.googleapis.com/.../frame_1.jpg"] to run integration


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(len(SAMPLE_IMAGE_URLS) == 0, reason="SAMPLE_IMAGE_URLS not set")
async def test_call_cigarette_api_integration():
    """call_cigarette_api with real image URLs returns prediction or fallback."""
    result = await call_cigarette_api(
        product_name="", description="", max_price=0, brand="", image_path_list=SAMPLE_IMAGE_URLS
    )
    assert isinstance(result, list)
    assert any(p.get("predictionType") == "cigarette_prediction" for p in result)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(len(SAMPLE_IMAGE_URLS) == 0, reason="SAMPLE_IMAGE_URLS not set")
async def test_call_torch_serving_integration():
    """call_torch_serving with real image URLs returns predictions or error_resp."""
    result = await call_torch_serving(SAMPLE_IMAGE_URLS)
    assert result is not None
    torch_dict, _, _, _ = result
    assert "text" in torch_dict and "logo" in torch_dict


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(len(SAMPLE_IMAGE_URLS) == 0, reason="SAMPLE_IMAGE_URLS not set")
async def test_call_tf_serving_integration():
    """call_tf_serving with real image URLs returns predictions or error_resp."""
    result = await call_tf_serving(SAMPLE_IMAGE_URLS)
    assert result is not None
    assert "wtmk" in result and "nsfw" in result and "blur" in result
