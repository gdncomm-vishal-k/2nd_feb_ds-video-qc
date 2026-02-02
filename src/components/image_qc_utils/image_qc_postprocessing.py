import sys
from pathlib import Path

# Add root to path for configs imports (go up 3 levels: image_qc_utils -> components -> src -> root)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from configs.config import PROB_THRESHOLD, BLUR_LOW_HIGH, KEYWORD_PROB_THRESHOLD, RESTRICTION_MODEL_KEYWORDS, PREDICTION_MAP
from typing import List, Dict

def get_watermark_response(proba_result) -> List[Dict]:
    watermark_response: List[Dict] = []
    for prob in proba_result:
        response = {
            "predictionType": "watermark_predictions",
            "present": None if prob is None or prob < 0 else prob >= PROB_THRESHOLD["wtmk"],
            "confidence": None if prob is None or prob < 0 else round(float(prob) * 100, 2),
        }
        watermark_response.append(response)

    return watermark_response

def get_explicit_response(proba_result) -> List[Dict]:
    text_response: List[Dict] = []
    for prob in proba_result:
        response = {
            "predictionType": "nsfw_predictions",
            "present": None if prob is None or prob < 0 else prob >= PROB_THRESHOLD["nsfw"],
            "confidence": None if prob is None or prob < 0 else round(float(prob) * 100, 2),
        }
        text_response.append(response)

    return text_response


def trim_score(score, low=0, high=1) -> float:
    score = max(score, low)
    score = min(score, high)
    return score


def get_blur_response(proba_result) -> List[Dict]:
    """
    blur model returns score from [1, 5]
    :param proba_result: float value range [1, 5]
    :return:
    """
    blur_response: List[Dict] = []
    for score in proba_result:
        response = {
            "predictionType": "blur_predictions"
        }
        if score is None or score < 0:
            response["present"] = None
            response["confidence"] = None
        else:
            low, high = BLUR_LOW_HIGH
            score = trim_score(score, low, high)
            score = 1 - score / 5
            assert 0 <= score <= 5, f"score must be in range [0, 5] but is {score}"
            response["present"] = score < PROB_THRESHOLD["blur"]
            response["confidence"] = round(float(score) * 100, 2)

        blur_response.append(response)

    return blur_response


def get_text_ocr_response(bool_result) -> List[Dict]:
    ocr_response: List[Dict] = []
    for flag in bool_result:
        response = {
            "predictionType": "text_predictions",
            "present": None if flag is None or flag < 0 else bool(flag),
            "confidence": None if flag is None or flag < 0 else round(float(flag) * 100, 2)
        }
        ocr_response.append(response)

    return ocr_response


def get_logo_response(proba_result):
    logo_response: List[Dict] = []
    for prob in proba_result:
        response = {
            "predictionType": "logo_predictions",
            "present": None if prob is None or prob < 0 else prob >= PROB_THRESHOLD["logo"],
            "confidence": None if prob is None or prob < 0 else round(float(prob) * 100, 2),
        }
        logo_response.append(response)

    return logo_response


def get_medicine_logo_response(proba_result_list):
    keras_responses = []
    narkotika_responses = []
    for result in proba_result_list:
        if result is None:
            keras_present, keras_confidence = None, None
            narkotika_present, narkotika_confidence = None, None
        else:
            keras_present, keras_confidence = result["keras_predictions"] > PROB_THRESHOLD["keras_logo"], result[
                "keras_predictions"]
            narkotika_present, narkotika_confidence = result["narkotika_predictions"] > PROB_THRESHOLD[
                "narkotika_logo"], result["narkotika_predictions"]
        keras_responses.append({
            "predictionType": "pharma_prescription",
            "present": keras_present,
            "confidence": keras_confidence
        })
        narkotika_responses.append({
            "predictionType": "pharma_banned",
            "present": narkotika_present,
            "confidence": narkotika_confidence
        })
    return keras_responses, narkotika_responses

def merge_ocr_text_results_for_restriction_models(ocr_result, text_result):
    image_ocr_none = [i is None for i in ocr_result]
    if any(ocr_result) or text_result:
        return 100
    if all(image_ocr_none) and text_result is None:
        return None
    return 0
    
def merge_ocr_text_results_for_cigarette_models(ocr_result, text_result, ocr_flag):
    image_ocr_none = [i is None for i in ocr_result]
    if (any(ocr_result) and ocr_flag) or (any(ocr_result) and ocr_flag is None) or text_result:
        return 100
    if all(image_ocr_none) and text_result is None:
        return None
    return 0


def get_restricted_keyword_response(response, dsj_restricted_results, cigarette_result, alcohol_result, guns_result):
    restriction_model_resp = {}
    for prediction in dsj_restricted_results:
        pred_type = prediction.get("prediction_type")
        if pred_type not in KEYWORD_PROB_THRESHOLD:
            continue
        restriction_model_resp[prediction["prediction_type"]] = any(
            list(map(lambda x: x["confidence"] is not None and x["confidence"] >= KEYWORD_PROB_THRESHOLD[
                prediction["prediction_type"]],
                     prediction["predictions"])))
    restriction_model_resp["cigarette"] = cigarette_result is not None and cigarette_result >= KEYWORD_PROB_THRESHOLD[
        "cigarette"]
    restriction_model_resp["alcohol"] = alcohol_result is not None and alcohol_result >= KEYWORD_PROB_THRESHOLD[
        "alcohol"]
    restriction_model_resp["guns"] = guns_result is not None and guns_result >= KEYWORD_PROB_THRESHOLD["guns"]
    for keyword_dict in response["keywordRecommendations"]:
        if keyword_dict.get("keywordType", None) in RESTRICTION_MODEL_KEYWORDS:
            keyword_dict["recommendation"] = "valid-detection" \
                if restriction_model_resp[PREDICTION_MAP[keyword_dict["keywordType"]]] \
                else \
                "not-sure"
        if keyword_dict.get("validateByDs") and keyword_dict.get("validateByDs") == True and not keyword_dict.get("recommendation"):
            keyword_dict["recommendation"] = "not-sure"
    return response