"""Tests for image_qc_utils.image_qc_postprocessing."""
import pytest
from components.image_qc_utils.image_qc_postprocessing import (
    get_watermark_response,
    get_explicit_response,
    trim_score,
    get_blur_response,
    get_text_ocr_response,
    get_logo_response,
    get_medicine_logo_response,
    merge_ocr_text_results_for_restriction_models,
    merge_ocr_text_results_for_cigarette_models,
    get_restricted_keyword_response,
)


def test_get_watermark_response():
    out = get_watermark_response([0.1, 0.6, None, -1])
    assert len(out) == 4
    assert out[0]["predictionType"] == "watermark_predictions"
    assert out[0]["present"] is False
    assert out[1]["present"] is True
    assert out[2]["present"] is None
    assert out[3]["present"] is None


def test_get_explicit_response():
    out = get_explicit_response([0.3, 0.6])
    assert len(out) == 2
    assert out[0]["predictionType"] == "nsfw_predictions"
    assert out[0]["present"] is False
    assert out[1]["present"] is True


def test_trim_score():
    assert trim_score(0.5, 0, 1) == 0.5
    assert trim_score(-1, 0, 1) == 0
    assert trim_score(2, 0, 1) == 1


def test_get_blur_response():
    out = get_blur_response([None, -1, 3.0, 1.0])
    assert len(out) == 4
    assert out[0]["present"] is None
    assert out[1]["present"] is None
    assert out[2]["predictionType"] == "blur_predictions"
    assert "present" in out[2]
    assert "confidence" in out[2]


def test_get_text_ocr_response():
    out = get_text_ocr_response([0, 1, None])
    assert len(out) == 3
    assert out[0]["predictionType"] == "text_predictions"
    assert out[0]["present"] is False
    assert out[1]["present"] is True
    assert out[2]["present"] is None


def test_get_logo_response():
    out = get_logo_response([0.2, 0.8])
    assert len(out) == 2
    assert out[0]["predictionType"] == "logo_predictions"


def test_get_medicine_logo_response():
    keras, narkotika = get_medicine_logo_response([
        None,
        {"keras_predictions": 60, "narkotika_predictions": 40},
    ])
    assert len(keras) == 2
    assert len(narkotika) == 2
    assert keras[0]["present"] is None
    assert keras[1]["present"] is True
    assert narkotika[1]["present"] is False


def test_merge_ocr_text_results_for_restriction_models():
    assert merge_ocr_text_results_for_restriction_models([1], True) == 100
    assert merge_ocr_text_results_for_restriction_models([], True) == 100
    assert merge_ocr_text_results_for_restriction_models([None, None], None) is None
    assert merge_ocr_text_results_for_restriction_models([None], None) is None
    assert merge_ocr_text_results_for_restriction_models([0], None) == 0


def test_merge_ocr_text_results_for_cigarette_models():
    assert merge_ocr_text_results_for_cigarette_models([1], None, True) == 100
    assert merge_ocr_text_results_for_cigarette_models([1], None, None) == 100
    assert merge_ocr_text_results_for_cigarette_models([], True, False) == 100
    assert merge_ocr_text_results_for_cigarette_models([None, None], None, False) is None
    assert merge_ocr_text_results_for_cigarette_models([None], None, False) is None
    assert merge_ocr_text_results_for_cigarette_models([0], None, False) == 0


def test_get_restricted_keyword_response():
    response = {
        "keywordRecommendations": [
            {"keywordType": "Cigarette", "recommendation": None, "validateByDs": True},
            {"keywordType": "Other", "recommendation": None},
        ]
    }
    dsj_restricted = []
    result = get_restricted_keyword_response(
        response, dsj_restricted,
        cigarette_result=60,
        alcohol_result=0,
        guns_result=0,
    )
    assert "keywordRecommendations" in result
    assert result["keywordRecommendations"][0]["keywordType"] == "Cigarette"


def test_get_restricted_keyword_response_with_dsj_and_validate_by_ds():
    response = {
        "keywordRecommendations": [
            {"keywordType": "Doctor's Prescription", "recommendation": None, "validateByDs": True},
            {"keywordType": "UnknownType", "recommendation": None, "validateByDs": True},
        ]
    }
    dsj_restricted = [
        {"prediction_type": "pharma_prescription", "predictions": [{"confidence": 60}]},
    ]
    result = get_restricted_keyword_response(
        response, dsj_restricted,
        cigarette_result=0, alcohol_result=0, guns_result=0,
    )
    assert result["keywordRecommendations"][0]["recommendation"] in ("valid-detection", "not-sure")
    assert result["keywordRecommendations"][1]["recommendation"] == "not-sure"
