import vertexai
import logging
from typing import Optional
from vertexai.generative_models import GenerationConfig, GenerativeModel
from configs.config import PROJECT_ID, REGION, GEMINI_API_KEY, GEMINI_SYSTEM_PROMPT, GEMINI_MODEL_NAME, GEMINI_MAX_OUTPUT_TOKENS, GEMINI_TEMPERATURE
from configs.logging import simple_logger
vertexai.init(project=PROJECT_ID, location=REGION, api_key=GEMINI_API_KEY)

@simple_logger()    
def get_prompt_ready(prompt: str = None, input_text: str = None) -> str:
    return f"""
    {prompt}
    predict for the input -> {input_text}
    """
@simple_logger()    
async def validate_text_with_gemini(
    prompt: str = GEMINI_SYSTEM_PROMPT,
    model_name: str = GEMINI_MODEL_NAME,
    input_text: str = None,
) -> Optional[str]:
    logging.info(f"Validating text with Gemini: {input_text}")
    if input_text is None or input_text.lower().strip() == "":
        return "NO INPUT TEXT"

    prompt = get_prompt_ready(prompt, input_text)

    try:
        logging.info(f"Getting model: {model_name}")
        model = GenerativeModel(
            model_name=model_name,
            generation_config=GenerationConfig(
                temperature=GEMINI_TEMPERATURE,
                max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
            ),
        )

        response = await model.generate_content_async(prompt)

        if response and response.candidates:
            model_response = response.candidates[0].content.parts[0].text
            logging.info(f"Response from Gemini: {model_response}")
            return model_response
        return "NO RESPONSE FROM GEMINI"

    except Exception as e:
        raise RuntimeError("Text generation failed") from e
