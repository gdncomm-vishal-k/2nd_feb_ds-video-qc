REQUESTS_TIMEOUT = 60

TF_SERVING_REST_ADDR = "http://ds-image-qc-tf-serving.qa2-sg.cld/predict/all_models"
TF_SERVING_API_HEALTH_CHECK_URL = "http://ds-image-qc-tf-serving.qa2-sg.cld/sys-info/health"


TORCH_SERVING_REST_ADDR = "http://ds-image-qc-torch-serving.qa2-sg.cld/predict/all_models"
TORCH_SERVING_API_HEALTH_CHECK_URL = "http://ds-image-qc-torch-serving.qa2-sg.cld/sys-info/health"


CIGARETTE_API_REST_ADDR = "http://ds-cigarette-prediction-api.qa2-sg.cld/predict"
CIGARETTE_API_HEALTH_CHECK_URL = "http://ds-cigarette-prediction-api.qa2-sg.cld/sys_info/health_check"


PROB_THRESHOLD = {
    "blur": 2.5,
    "nsfw": 0.5,
    "text": 0.5,
    "wtmk": 0.5,
    "logo": 0.5,
    "keras_logo": 50,
    "narkotika_logo": 50
}

BLUR_LOW_HIGH = [0, 5]

KEYWORD_PROB_THRESHOLD = {
    "pharma_prescription": 50,
    "pharma_banned": 50,
    "google_restricted":50,
    "cigarette": 50,
    "alcohol": 50,
    "guns": 50
}

RESTRICTION_MODEL_KEYWORDS = [
    "Doctor's Prescription",
    "Illegal drugs",
    "Cigarette",
    "Google Voucher",
    "Alcohol",
    "Guns"
]

PREDICTION_MAP = {
    "Doctor's Prescription": "pharma_prescription",
    "Illegal drugs": "pharma_banned",
    "Google Voucher": "google_restricted",
    "Cigarette": "cigarette",
    "Alcohol": "alcohol",
    "Guns": "guns"
}

IMAGE_QC_PREDICTION_BATCH_SIZE = 30
ENABLE_ILLEGAL_CIGARETTE_PREDICTION = True

PROJECT_ID = ("data-science-prod-218306")
GEMINI_API_KEY = 'API_KEY_REMOVED'
GEMINI_MODEL_NAME = 'gemini-2.0-flash-lite'
REGION = "us-central1"
GEMINI_MAX_OUTPUT_TOKENS = 100
GEMINI_TEMPERATURE = 0
GEMINI_SYSTEM_PROMPT = '''
You are a content compliance specialist for Blibli, an Indonesian e-commerce platform.
 
## Your Primary Task:
- Your Primary Task is to analyse the input text(caption and transcript of an audio) and check if the text compiles with the Blibli's content policies.
    ## Context Understanding:
        - The caption and transcript can be in Bahasa or English.
        - The transcript is not always perfect, so interpretation should be tolerant of minor transcription errors (e.g., spelling, grammar, filler words).
        - The transcript is extracted from user-submitted videos on the platform, typically from product reviews or unboxing content.
 
    ## Content Compliance Rules:
        The text **must NOT contain** any of the following,
        1. **Competitor Names:**
            - Tokopedia, Lazada, Shopee, Zalora, etc.
        2. **Political Party Names or Figures:**
            - National Awakening Party (PKB), Great Indonesia Movement Party (Gerindra), Indonesian Democratic Party of Struggle (PDI-P)Jokowi,
            Prabowo, Partai Komunis, Kampret, Kafir, etc.
        3. **Abusive, profane, or NSFW words (including slang, domain fragments, or numeric leetspeak):**
            - .co.id, .net, .org, 8.23E+10, 69 58, adf.ly, akulaku, aliexpress, anjink, anjinng, asu, bacot, bajingan,
            bangke, bangsat, bego, bejad, bejat, bhinneka, bispak, bitch, bl, blackmarket, blackmarketphonsel, brengsek,
            buka lapak, bukalapak, bullshit, carefour, crap, cunt, damn, dick, dizalora, dongo, dungu, eek, elevenia,
            facebook, fuck, fucker, gibran, gmail.com, goblok, grazera, http, https, https://m.facebook.com/foxdistroshop,
            hypermart, IG;blackmarketphonsel, instagram, invite, itil, jablay, jamban, jancok, jancuk, jd.id, jdid, jembut,
            jokowi, kafir, kampang, kampret, keparat, khontol, kimbe, kimbek, komunis, kontol, ktp, kunyuk, laknat, lazada,
            lonte, maling, mataharimall, memek, motherfucker, ngehe, ngentot, olshop, pecun, peler, pembohong, penis, perek,
            pin BB, PIN BBM;DA919E55, pler, prabowo, pussy, qareg, RP, Rp., setan, shit, shopee, sialan, siup, suck, sucks,
            tai, taik, tit, toket, tokopedia, tokped, tolol, toped, w3bsite, whatsapp, whore, wtf, www.Blackmarketphonsel.com,
            yahoo.com, zalora.

        **Output Format:**
        {{
            "CAPTION_QC_FLAG": "Return CAPTION_HAS_ISSUE if the input breaks any content rules, otherwise return CAPTION_NO_ISSUE.",
            "CAPTION_QC_REASON": "<reason behind why the text is valid or invalid under 30 words>",
            "AUDIO_QC_FLAG": "Return AUDIO_HAS_ISSUE if the input breaks any content rules, otherwise return AUDIO_NO_ISSUE.",
            "AUDIO_QC_REASON": "<reason behind why the text is valid or invalid under 30 words>",
        }}
'''


PREDICTIONS = [
    "pharma_banned",
    "pharma_prescription",
    "competitor_logo",
    "nsfw",
    "cigarette",
    "alcohol",
    "guns",
    "blur",
    "text_predictions",
    "water_mark",
]

# Maps API prediction types to PREDICTIONS list names
VIDEO_QC_PREDICTION_MAP = {
    "text_predictions": "text_predictions",
    "pharma_prescription": "pharma_prescription",
    "pharma_banned": "pharma_banned",
    "logo_predictions": "competitor_logo",
    "nsfw_predictions": "nsfw",
    "blur_predictions": "blur",
    "watermark_predictions": "water_mark",
    "cigarette_prediction": "cigarette",
    "alcohol_prediction": "alcohol",
    "guns_prediction": "guns",
}

# Video QC Pipeline Config
VIDEO_QC_TEMP_FOLDER = "./temp_videos"
VIDEO_QC_CHUNKS = 8
VIDEO_QC_BUCKET = "test-images-image-qc"
VIDEO_QC_GCS_FOLDER = "Video_QC/Code_testing"
VIDEO_QC_FPS = 1

# Whisper Model Config
DEVICE = "cpu"
WHISPER_MODEL_PATH = "./model/faster_whisper_turbo_v3_large"
WHISPER_GCS_BUCKET = "test-images-image-qc"
WHISPER_GCS_MODEL_PATH = "Video_QC/models/faster_whisper_turbo_v3_large"

# Logging Config
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

FRAME_DIFF_THRESHOLD = 30 # min=0 max=255
FRAME_RESIZE_TO = (128, 128)