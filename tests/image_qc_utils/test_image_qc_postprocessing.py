"""
Tests for src/components/image_qc_utils/image_qc_postprocessing.py.
No mocks – uses real config (PROB_THRESHOLD, BLUR_LOW_HIGH, etc.) and real input examples.
"""

import pytest

from components.image_qc_utils.image_qc_postprocessing import (
    get_blur_response,
    get_explicit_response,
    get_logo_response,
    get_medicine_logo_response,
    get_restricted_keyword_response,
    get_text_ocr_response,
    get_watermark_response,
    merge_ocr_text_results_for_cigarette_models,
    merge_ocr_text_results_for_restriction_models,
    trim_score,
)


# ---------------------------------------------------------------------------
# trim_score
# ---------------------------------------------------------------------------
def test_trim_score_clips_to_bounds():
    assert trim_score(0.5) == 0.5
    assert trim_score(-0.5, 0, 1) == 0
    assert trim_score(1.5, 0, 1) == 1
    assert trim_score(3, 0, 5) == 3


# ---------------------------------------------------------------------------
# get_watermark_response – PROB_THRESHOLD["wtmk"] = 0.5
# ---------------------------------------------------------------------------
def test_get_watermark_response():
    # prob >= 0.5 -> present True; prob < 0.5 -> present False; None/negative -> None
    result = get_watermark_response([0.3, 0.6, None, -1])
    assert len(result) == 4
    assert result[0]["predictionType"] == "watermark_predictions"
    assert result[0]["present"] is False
    assert result[0]["confidence"] == 30.0
    assert result[1]["present"] is True
    assert result[1]["confidence"] == 60.0
    assert result[2]["present"] is None and result[2]["confidence"] is None
    assert result[3]["present"] is None and result[3]["confidence"] is None


# ---------------------------------------------------------------------------
# get_explicit_response – PROB_THRESHOLD["nsfw"] = 0.5
# ---------------------------------------------------------------------------
def test_get_explicit_response():
    result = get_explicit_response([0.1, 0.5])
    assert len(result) == 2
    assert result[0]["predictionType"] == "nsfw_predictions"
    assert result[0]["present"] is False
    assert result[1]["present"] is True


# ---------------------------------------------------------------------------
# get_blur_response – blur score [1, 5], BLUR_LOW_HIGH [0, 5]
# ---------------------------------------------------------------------------
def test_get_blur_response():
    result = get_blur_response([1.0, 3.0, None])
    assert len(result) == 3
    assert result[0]["predictionType"] == "blur_predictions"
    assert result[0]["present"] is not None
    assert result[0]["confidence"] is not None
    assert result[2]["present"] is None and result[2]["confidence"] is None


# ---------------------------------------------------------------------------
# get_text_ocr_response
# ---------------------------------------------------------------------------
def test_get_text_ocr_response():
    result = get_text_ocr_response([0, 1, None])
    assert len(result) == 3
    assert result[0]["predictionType"] == "text_predictions"
    assert result[0]["present"] is False
    assert result[1]["present"] is True
    assert result[2]["present"] is None and result[2]["confidence"] is None


# ---------------------------------------------------------------------------
# get_logo_response – PROB_THRESHOLD["logo"] = 0.5
# ---------------------------------------------------------------------------
def test_get_logo_response():
    result = get_logo_response([0.4, 0.6])
    assert len(result) == 2
    assert result[0]["predictionType"] == "logo_predictions"
    assert result[0]["present"] is False
    assert result[1]["present"] is True


# ---------------------------------------------------------------------------
# get_medicine_logo_response – keras_logo, narkotika_logo thresholds 50
# ---------------------------------------------------------------------------
def test_get_medicine_logo_response():
    # result: list of dicts with keras_predictions, narkotika_predictions
    proba_result_list = [
        {"keras_predictions": 60, "narkotika_predictions": 30},
        None,
    ]
    keras_resp, narkotika_resp = get_medicine_logo_response(proba_result_list)
    assert len(keras_resp) == 2
    assert len(narkotika_resp) == 2
    assert keras_resp[0]["predictionType"] == "pharma_prescription"
    assert keras_resp[0]["present"] is True
    assert narkotika_resp[0]["predictionType"] == "pharma_banned"
    assert narkotika_resp[0]["present"] is False
    assert keras_resp[1]["present"] is None and narkotika_resp[1]["present"] is None


# ---------------------------------------------------------------------------
# merge_ocr_text_results_for_restriction_models
# ---------------------------------------------------------------------------
def test_merge_ocr_text_results_for_restriction_models():
    assert merge_ocr_text_results_for_restriction_models([], None) is None
    assert merge_ocr_text_results_for_restriction_models([0, 0], None) == 0
    assert merge_ocr_text_results_for_restriction_models([1], False) == 100
    assert merge_ocr_text_results_for_restriction_models([0], True) == 100


# ---------------------------------------------------------------------------
# merge_ocr_text_results_for_cigarette_models
# ---------------------------------------------------------------------------
def test_merge_ocr_text_results_for_cigarette_models():
    assert merge_ocr_text_results_for_cigarette_models([], None, None) is None
    assert merge_ocr_text_results_for_cigarette_models([0, 0], None, False) == 0
    assert merge_ocr_text_results_for_cigarette_models([1], True, True) == 100
    assert merge_ocr_text_results_for_cigarette_models([1], None, None) == 100


# ---------------------------------------------------------------------------
# get_restricted_keyword_response – needs response with keywordRecommendations
# ---------------------------------------------------------------------------
def test_get_restricted_keyword_response():
    # Minimal response: keywordRecommendations with keywordType in RESTRICTION_MODEL_KEYWORDS
    response = {
        "keywordRecommendations": [
            {"keywordType": "Cigarette", "recommendation": ""},
            {"keywordType": "Alcohol", "recommendation": ""},
        ]
    }
    dsj_restricted_results = [
        {"prediction_type": "pharma_prescription", "predictions": [{"confidence": 60}]},
    ]
    # cigarette 60 >= 50 -> True; alcohol 0 >= 50 -> False
    result = get_restricted_keyword_response(
        response, dsj_restricted_results, cigarette_result=60, alcohol_result=0, guns_result=None
    )
    assert "keywordRecommendations" in result
    # Cigarette should get recommendation from restriction_model_resp
    cig = next(k for k in result["keywordRecommendations"] if k["keywordType"] == "Cigarette")
    alc = next(k for k in result["keywordRecommendations"] if k["keywordType"] == "Alcohol")
    assert cig["recommendation"] in ("valid-detection", "not-sure")
    assert alc["recommendation"] in ("valid-detection", "not-sure")
