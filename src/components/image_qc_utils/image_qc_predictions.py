import sys
from pathlib import Path
import re
# Add src/ to path for components imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
# Add root to path for configs imports (go up 3 levels: image_qc_utils -> components -> src -> root)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import httpx
import asyncio
import logging
from typing import Dict, List, Union, Tuple
from configs.config import PREDICTIONS, VIDEO_QC_PREDICTION_MAP
from configs.config import IMAGE_QC_PREDICTION_BATCH_SIZE
from components.image_qc_utils.clients import cigarette_api_post, tf_serving_post, torch_serving_post
from components.image_qc_utils.image_qc_postprocessing import get_watermark_response, get_explicit_response, get_blur_response, get_text_ocr_response, get_logo_response, get_medicine_logo_response, get_restricted_keyword_response

async def call_cigarette_api(product_name, description, max_price, brand, image_path_list: List[str]) -> list:
    body = {
        "productName": product_name,
        "description": description,
        "price": max_price,
        "brand": brand,
        "images": image_path_list
    }
    try:
        response: httpx.Response = await cigarette_api_post(body)
        prediction = response.json()
        logging.info(f"Prediction from cigarette API: {prediction}")
        return prediction
    except Exception as e:
        fallback = [
            dict(predictionType="cigarette_prediction", confidence=None, ocr_flag=None),
            dict(predictionType="alcohol_prediction", confidence=None),
            dict(predictionType="guns_prediction", confidence=None),
        ]
        logging.error(f"Fallback prediction from cigarette API: {fallback}")
        return fallback


async def call_torch_serving(image_path_list: List[str]) -> Union[Tuple[Dict, List, List, List], None]:
    error_resp = ({
                      "text": [{
                          "predictionType": "text_predictions",
                          "present": None,
                          "confidence": None
                      }] * len(image_path_list),

                      "logo": [{
                          "predictionType": "logo_predictions",
                          "present": None,
                          "confidence": None
                      }] * len(image_path_list),

                      "keras_logo": [{
                          "predictionType": "pharma_prescription",
                          "present": None,
                          "confidence": None
                      }] * len(image_path_list),

                      "narkotika_logo": [{
                          "predictionType": "pharma_banned",
                          "present": None,
                          "confidence": None
                      }] * len(image_path_list),
                  }, [None] * len(image_path_list), [None] * len(image_path_list), [None] * len(image_path_list))

    if len(image_path_list) == 0:
        return error_resp

    try:
        torch_response = await torch_serving_post(body={"images": image_path_list})
        torch_predictions = torch_response.json()
        keras, narkotika = get_medicine_logo_response(torch_predictions["medicine_logo_probs"])
    except Exception as e:
        logging.error(f"Error in torch serving: {e}")
        return error_resp
    else:
        logging.info(f"Torch predictions: {torch_predictions}")
        return dict(
            text=get_text_ocr_response(torch_predictions["classes"]),
            logo=get_logo_response(torch_predictions["logo_probs"]),
            keras_logo=keras,
            narkotika_logo=narkotika
        ), torch_predictions["cigarette_probs"], torch_predictions["alcohol_probs"], torch_predictions["guns_probs"]


async def call_tf_serving(image_path_list: List[str]):
    error_resp = {
        "wtmk": [{
            "predictionType": "watermark_predictions",
            "present": None,
            "confidence": None
        }] * len(image_path_list),

        "nsfw": [{
            "predictionType": "nsfw_predictions",
            "present": None,
            "confidence": None
        }] * len(image_path_list),

        "blur": [{
            "predictionType": "blur_predictions",
            "present": None,
            "confidence": None
        }] * len(image_path_list)
    }
    
    try:
        tf_response = await tf_serving_post(body={"images": image_path_list})
        tf_predictions = tf_response.json()
    except Exception as e:
        logging.error(f"Error in tf serving: {e}")
        return error_resp
    else:
        logging.info(f"TF predictions: {tf_predictions}")
        return dict(
            wtmk=get_watermark_response(tf_predictions["watermark_result"]),
            nsfw=get_explicit_response(tf_predictions["nsfw_result"]),
            blur=get_blur_response(tf_predictions["blur_result"])
        )
    
def merge_ocr_text_results_for_cigarette_models(ocr_result, text_result, ocr_flag):
    image_ocr_none = [i is None for i in ocr_result]
    if (any(ocr_result) and ocr_flag) or (any(ocr_result) and ocr_flag is None) or text_result:
        return 100
    if all(image_ocr_none) and text_result is None:
        return None
    return 0


## Check this logic again properly
def override_cigarette_with_torch_ocr(batch_results):

    for batch in batch_results:
        cigarette_ocr_probs = batch["torch_serving"][1] or []
        
        for idx, frame_api in enumerate(batch["cigarette"]):
            cigarette_ocr = cigarette_ocr_probs[idx] if idx < len(cigarette_ocr_probs) else None
            
            for pred in frame_api:
                if pred["predictionType"] == "cigarette_prediction":
                    pred["confidence"] = merge_ocr_text_results_for_cigarette_models(
                        [cigarette_ocr], pred.get("confidence"), pred.get("ocr_flag")
                    )
                    break
    
    return batch_results

def split_into_batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


async def predict_frames_in_batches(frame_urls, url_mapping=None):
    if not frame_urls:
        raise RuntimeError("No frame URLs provided")

    all_batch_results = []

    for batch in split_into_batches(frame_urls, IMAGE_QC_PREDICTION_BATCH_SIZE):
        tasks = [
            call_torch_serving(batch),
            call_tf_serving(batch),
            *[
                call_cigarette_api(
                    product_name="",
                    description="",
                    max_price=0,
                    brand="",
                    image_path_list=[url],
                )
                for url in batch
            ],
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for res in results:
            if isinstance(res, Exception):
                raise RuntimeError("Prediction batch failed") from res

        all_batch_results.append(
            {
                "torch_serving": results[0],
                "tf_serving": results[1],
                "cigarette": results[2:],
            }
        )

    per_frame_results = get_per_frame_results(frame_urls, override_cigarette_with_torch_ocr(all_batch_results))
    return summarize_video_qc_results(per_frame_results, url_mapping)


def get_per_frame_results(frame_urls, batch_results):
    per_frame_results = {}
    frame_idx = 0
    
    for batch in batch_results:
        torch_dict = batch["torch_serving"][0]  # {text, logo, keras_logo, narkotika_logo}
        tf_dict = batch["tf_serving"]           # {wtmk, nsfw, blur}
        cigarette_list = batch["cigarette"]     # list of per-frame results
        
        num_frames = len(cigarette_list)
        
        for i in range(num_frames):
            
            frame_result = {
                "torch_serving": {
                    "text": torch_dict["text"][i],
                    "logo": torch_dict["logo"][i],
                    "keras_logo": torch_dict["keras_logo"][i],
                    "narkotika_logo": torch_dict["narkotika_logo"][i],
                },
                "tf_serving": {
                    "wtmk": tf_dict["wtmk"][i],
                    "nsfw": tf_dict["nsfw"][i],
                    "blur": tf_dict["blur"][i],
                },
                "cigarette_service": cigarette_list[i]
            }
            
            filtered_result = filter_predictions(frame_result, PREDICTIONS, VIDEO_QC_PREDICTION_MAP)
            
            per_frame_results[frame_urls[frame_idx]] = filtered_result
            frame_idx += 1
    
    return per_frame_results


def filter_predictions(frame_result, predictions_list, prediction_map):
    """
    Filter frame results to only include predictions from the PREDICTIONS list.
    """
    filtered = {
        "torch_serving": {},
        "tf_serving": {},
        "cigarette_service": []
    }
    
    # Filter torch_serving
    for key, value in frame_result["torch_serving"].items():
        pred_type = value.get("predictionType", "")
        if prediction_map.get(pred_type) in predictions_list:
            filtered["torch_serving"][key] = value
    
    # Filter tf_serving
    for key, value in frame_result["tf_serving"].items():
        pred_type = value.get("predictionType", "")
        if prediction_map.get(pred_type) in predictions_list:
            filtered["tf_serving"][key] = value
    
    # Filter cigarette_service
    for pred in frame_result["cigarette_service"]:
        pred_type = pred.get("predictionType", "")
        if prediction_map.get(pred_type) in predictions_list:
            filtered["cigarette_service"].append(pred)
    
    return filtered


def get_frame_number(frame_name):
    """Extract frame number from frame name like 'frame_0005.jpg' -> 5"""
    match = re.search(r'frame_(\d+)', frame_name)
    return int(match.group(1)) if match else 0


def group_consecutive_frames(frame_numbers):
    """
    Group consecutive frame numbers into ranges.
    [1,2,3,5,9] -> ["1-3", "5", "9"]
    """
    if not frame_numbers:
        return []
    
    frame_numbers = sorted(frame_numbers)
    result = []
    start = frame_numbers[0]
    end = frame_numbers[0]
    
    for num in frame_numbers[1:]:
        if num == end + 1:
            end = num
        else:
            # Save previous range
            if start == end:
                result.append(str(start))
            else:
                result.append(f"{start}-{end}")
            start = num
            end = num
    
    # Save last range
    if start == end:
        result.append(str(start))
    else:
        result.append(f"{start}-{end}")
    
    return result


def summarize_video_qc_results(per_frame_results, url_mapping=None):
    
    summary = {pred: [] for pred in PREDICTIONS}
    flagged_frames = {pred: [] for pred in PREDICTIONS}
    
    for frame_url, frame_data in per_frame_results.items():
        # Get all original frame numbers for this distinct frame
        if url_mapping and frame_url in url_mapping:
            original_frames = url_mapping[frame_url]
        else:
            original_frames = [get_frame_number(frame_url)]
        
        # Check torch_serving predictions
        for key, value in frame_data.get("torch_serving", {}).items():
            if value.get("present"):
                pred_type = value.get("predictionType", "")
                mapped = VIDEO_QC_PREDICTION_MAP.get(pred_type)
                if mapped in flagged_frames:
                    flagged_frames[mapped].extend(original_frames)
        
        # Check tf_serving predictions
        for key, value in frame_data.get("tf_serving", {}).items():
            if value.get("present"):
                pred_type = value.get("predictionType", "")
                mapped = VIDEO_QC_PREDICTION_MAP.get(pred_type)
                if mapped in flagged_frames:
                    flagged_frames[mapped].extend(original_frames)
        
        # Check cigarette_service predictions
        for pred in frame_data.get("cigarette_service", []):
            confidence = pred.get("confidence")
            if confidence is not None and confidence > 0:
                pred_type = pred.get("predictionType", "")
                mapped = VIDEO_QC_PREDICTION_MAP.get(pred_type)
                if mapped in flagged_frames:
                    flagged_frames[mapped].extend(original_frames)
    
    # Group consecutive frames into ranges (sorted and deduplicated)
    for pred in PREDICTIONS:
        unique_frames = sorted(set(flagged_frames[pred]))
        summary[pred] = group_consecutive_frames(unique_frames)
    
    return summary