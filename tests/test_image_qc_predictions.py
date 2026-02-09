"""Tests for image_qc_utils.image_qc_predictions."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from configs.config import PREDICTIONS, VIDEO_QC_PREDICTION_MAP
from components.image_qc_utils.image_qc_predictions import (
    call_cigarette_api,
    call_torch_serving,
    call_tf_serving,
    merge_ocr_text_results_for_cigarette_models,
    override_cigarette_with_torch_ocr,
    split_into_batches,
    predict_frames_in_batches,
    get_per_frame_results,
    filter_predictions,
    get_frame_number,
    group_consecutive_frames,
    summarize_video_qc_results,
)


# --- call_cigarette_api ---


@pytest.mark.asyncio
async def test_call_cigarette_api_success():
    with patch("components.image_qc_utils.image_qc_predictions.cigarette_api_post", new_callable=AsyncMock) as mock_post:
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"predictionType": "cigarette_prediction", "confidence": 0}]
        mock_post.return_value = mock_resp
        result = await call_cigarette_api("", "", 0, "", ["http://img.jpg"])
        assert result == [{"predictionType": "cigarette_prediction", "confidence": 0}]


@pytest.mark.asyncio
async def test_call_cigarette_api_fallback():
    with patch("components.image_qc_utils.image_qc_predictions.cigarette_api_post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = Exception("api down")
        result = await call_cigarette_api("", "", 0, "", ["http://img.jpg"])
        assert isinstance(result, list)
        assert any("cigarette" in str(p.get("predictionType", "")) for p in result)


# --- call_torch_serving ---


@pytest.mark.asyncio
async def test_call_torch_serving_empty_list():
    result = await call_torch_serving([])
    assert result[0]["text"] == []
    assert result[0]["logo"] == []


@pytest.mark.asyncio
async def test_call_torch_serving_success():
    with patch("components.image_qc_utils.image_qc_predictions.torch_serving_post", new_callable=AsyncMock) as mock_post:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "classes": [0],
            "logo_probs": [0.1],
            "medicine_logo_probs": [None],
            "cigarette_probs": [None],
            "alcohol_probs": [None],
            "guns_probs": [None],
        }
        mock_post.return_value = mock_resp
        result = await call_torch_serving(["http://img.jpg"])
        assert "text" in result[0]
        assert "logo" in result[0]


@pytest.mark.asyncio
async def test_call_torch_serving_exception():
    with patch("components.image_qc_utils.image_qc_predictions.torch_serving_post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = Exception("err")
        result = await call_torch_serving(["http://img.jpg"])
        assert result[0]["text"][0]["present"] is None


# --- call_tf_serving ---


@pytest.mark.asyncio
async def test_call_tf_serving_success():
    with patch("components.image_qc_utils.image_qc_predictions.tf_serving_post", new_callable=AsyncMock) as mock_post:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "watermark_result": [0.1],
            "nsfw_result": [0.1],
            "blur_result": [2.0],
        }
        mock_post.return_value = mock_resp
        result = await call_tf_serving(["http://img.jpg"])
        assert "wtmk" in result
        assert "nsfw" in result
        assert "blur" in result


@pytest.mark.asyncio
async def test_call_tf_serving_exception():
    with patch("components.image_qc_utils.image_qc_predictions.tf_serving_post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = Exception("err")
        result = await call_tf_serving(["http://img.jpg"])
        assert result["wtmk"][0]["present"] is None


# --- merge_ocr_text_results_for_cigarette_models (in predictions module) ---


def test_merge_ocr_text_results_for_cigarette_models():
    assert merge_ocr_text_results_for_cigarette_models([1], 50, True) == 100
    assert merge_ocr_text_results_for_cigarette_models([None], None, None) == None


# --- override_cigarette_with_torch_ocr ---


def test_override_cigarette_with_torch_ocr():
    batch_results = [{
        "torch_serving": (None, [80]),
        "cigarette": [[{"predictionType": "cigarette_prediction", "confidence": 0, "ocr_flag": True}]],
    }]
    out = override_cigarette_with_torch_ocr(batch_results)
    assert out == batch_results


# --- split_into_batches ---


def test_split_into_batches():
    from components.image_qc_utils.image_qc_predictions import IMAGE_QC_PREDICTION_BATCH_SIZE
    items = list(range(35))
    batches = list(split_into_batches(items, IMAGE_QC_PREDICTION_BATCH_SIZE))
    assert len(batches) == 2
    assert len(batches[0]) == IMAGE_QC_PREDICTION_BATCH_SIZE
    assert len(batches[1]) == 5


# --- get_frame_number ---


def test_get_frame_number():
    assert get_frame_number("https://storage.googleapis.com/b/frame_0005.jpg") == 5
    assert get_frame_number("frame_0042.jpg") == 42
    assert get_frame_number("no_match") == 0


# --- group_consecutive_frames ---


def test_group_consecutive_frames():
    assert group_consecutive_frames([1, 2, 3, 5, 9]) == ["1-3", "5", "9"]
    assert group_consecutive_frames([1]) == ["1"]
    assert group_consecutive_frames([]) == []


# --- filter_predictions ---


def test_filter_predictions():
    frame_result = {
        "torch_serving": {
            "text": {"predictionType": "text_predictions", "present": True},
            "logo": {"predictionType": "logo_predictions", "present": False},
        },
        "tf_serving": {"blur": {"predictionType": "blur_predictions", "present": False}},
        "cigarette_service": [{"predictionType": "cigarette_prediction", "confidence": 0}],
    }
    out = filter_predictions(frame_result, PREDICTIONS, VIDEO_QC_PREDICTION_MAP)
    assert "torch_serving" in out
    assert "tf_serving" in out
    assert "cigarette_service" in out


# --- get_per_frame_results ---


def test_get_per_frame_results():
    frame_urls = ["http://f1.jpg", "http://f2.jpg"]
    batch_results = [{
        "torch_serving": ({
            "text": [{"predictionType": "text_predictions", "present": False}, {"predictionType": "text_predictions", "present": False}],
            "logo": [{"predictionType": "logo_predictions", "present": False}, {"predictionType": "logo_predictions", "present": False}],
            "keras_logo": [{"predictionType": "pharma_prescription", "present": False}, {"predictionType": "pharma_prescription", "present": False}],
            "narkotika_logo": [{"predictionType": "pharma_banned", "present": False}, {"predictionType": "pharma_banned", "present": False}],
        }, [], [], []),
        "tf_serving": {
            "wtmk": [{"predictionType": "watermark_predictions", "present": False}, {"predictionType": "watermark_predictions", "present": False}],
            "nsfw": [{"predictionType": "nsfw_predictions", "present": False}, {"predictionType": "nsfw_predictions", "present": False}],
            "blur": [{"predictionType": "blur_predictions", "present": False}, {"predictionType": "blur_predictions", "present": False}],
        },
        "cigarette": [
            [{"predictionType": "cigarette_prediction", "confidence": 0}],
            [{"predictionType": "cigarette_prediction", "confidence": 0}],
        ],
    }]
    out = get_per_frame_results(frame_urls, batch_results)
    assert len(out) == 2
    assert "http://f1.jpg" in out
    assert out["http://f1.jpg"]["torch_serving"]["text"]["predictionType"] == "text_predictions"


# --- summarize_video_qc_results ---


def test_summarize_video_qc_results():
    per_frame = {
        "http://f1.jpg": {
            "torch_serving": {"text": {"predictionType": "text_predictions", "present": True}},
            "tf_serving": {},
            "cigarette_service": [],
        },
    }
    url_mapping = {"http://f1.jpg": [1, 2, 3]}
    out = summarize_video_qc_results(per_frame, url_mapping)
    assert set(out.keys()) == set(PREDICTIONS)
    assert "1-3" in out.get("text_predictions", []) or "1" in out.get("text_predictions", [])


# --- predict_frames_in_batches ---


@pytest.mark.asyncio
async def test_predict_frames_in_batches_no_urls_raises():
    with pytest.raises(RuntimeError) as exc_info:
        await predict_frames_in_batches([])
    assert "No frame URLs" in str(exc_info.value)


@pytest.mark.asyncio
async def test_predict_frames_in_batches_success():
    with patch("components.image_qc_utils.image_qc_predictions.call_torch_serving", new_callable=AsyncMock) as mock_torch:
        with patch("components.image_qc_utils.image_qc_predictions.call_tf_serving", new_callable=AsyncMock) as mock_tf:
            with patch("components.image_qc_utils.image_qc_predictions.call_cigarette_api", new_callable=AsyncMock) as mock_cig:
                mock_torch.return_value = (
                    {
                        "text": [{"predictionType": "text_predictions", "present": False}],
                        "logo": [{"predictionType": "logo_predictions", "present": False}],
                        "keras_logo": [{"predictionType": "pharma_prescription", "present": False}],
                        "narkotika_logo": [{"predictionType": "pharma_banned", "present": False}],
                    },
                    [None], [None], [None],
                )
                mock_tf.return_value = {
                    "wtmk": [{"predictionType": "watermark_predictions", "present": False}],
                    "nsfw": [{"predictionType": "nsfw_predictions", "present": False}],
                    "blur": [{"predictionType": "blur_predictions", "present": False}],
                }
                mock_cig.return_value = [{"predictionType": "cigarette_prediction", "confidence": 0}]
                result = await predict_frames_in_batches(
                    ["http://frame_0001.jpg"],
                    url_mapping={"http://frame_0001.jpg": [1]},
                )
                assert set(result.keys()) == set(PREDICTIONS)
