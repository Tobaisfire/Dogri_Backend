import io
import json
import logging
import os
import pickle
import re
import sys
import time
import types
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from indicnlp.tokenize import indic_tokenize
from nltk.probability import LidstoneProbDist
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model
from transformers import AutoModelForTokenClassification, AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

logger = logging.getLogger("dogri_backend")
logging.basicConfig(level=logging.INFO)




BASE_MODEL_OPTIONS = ["SVM", "HMM", "BiLSTM", "mBERT", "Gemma-FineTuned", "phi-4-fine-tuned"]
HYBRID_MODEL_COMBINATIONS = {
    "SVM+BiLSTM": ("SVM", "BiLSTM"),
    "HMM+BiLSTM": ("HMM", "BiLSTM"),
    "SVM+mBERT": ("SVM", "mBERT"),
    "HMM+mBERT": ("HMM", "mBERT"),
}

MODEL_METRICS = {
    "SVM": {"accuracy": 0.80, "f1": 0.81},
    "HMM": {"accuracy": 0.77, "f1": 0.77},
    "BiLSTM": {"accuracy": 0.82, "f1": 0.80},
    "mBERT": {"accuracy": 0.83, "f1": 0.83},
    "Gemma-FineTuned": {"accuracy": 0.80, "f1": 0.78},
    "phi-4-fine-tuned": {"accuracy": 0.90, "f1": 0.89},
    "SVM+BiLSTM": {"accuracy": 0.8210, "precision": 0.8242, "recall": 0.8210, "f1": 0.8187},
    "HMM+BiLSTM": {"accuracy": 0.8210, "precision": 0.8242, "recall": 0.8210, "f1": 0.8187},
    "SVM+mBERT": {"accuracy": 0.8350, "precision": 0.8172, "recall": 0.8350, "f1": 0.8175},
    "HMM+mBERT": {"accuracy": 0.8350, "precision": 0.8172, "recall": 0.8350, "f1": 0.8175},
}

MODELS_CACHE: Optional[Dict] = None
DEFAULT_SENTENCE_FOLDER = os.environ.get(
    "DOGRI_SENTENCE_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "dataset-sent")
)
DEFAULT_SENTENCE_EXTENSIONS = (".txt", ".xml")
MANUAL_SENTENCE_CACHE: Optional[List[str]] = None
ANALYTICS_REQUIRED_COLUMNS = [
    "file_id",
    "category",
    "Sub-category",  # note: capital 'S' to match original dataset
    "address",
    "sentence_index",
    "Sentences",
    "token_index",
    "token",
    "Tagg",
    "confidence",
    "explanation",
]
DEFAULT_ANALYTICS_DATASET = os.environ.get(
    "DOGRI_ANALYTICS_DATASET",
    os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "Final_pos_tagged_200k(no error).xlsx"),
)
ANALYTICS_CACHE: Optional[Dict[str, Any]] = None


def lidstone_estimator(gamma, bins):
    return LidstoneProbDist(gamma, bins)


# Ensure pickled HMM tagger can find lidstone_estimator under __main__
if "__main__" not in sys.modules:
    sys.modules["__main__"] = types.ModuleType("__main__")
sys.modules["__main__"].lidstone_estimator = lidstone_estimator


def _extract_sentences_from_xml_root(root) -> List[str]:
    sentences = []
    for paragraph in root.findall(".//p"):
        text = (paragraph.text or "").strip()
        if text:
            sentences.extend(split_dogri_sentences(text))
    return sentences


def parse_xml_content(content: str) -> List[str]:
    try:
        root = ET.fromstring(content)
        return _extract_sentences_from_xml_root(root)
    except Exception:
        return []


def parse_xml_file(path: str) -> List[str]:
    try:
        logger.info("Parsing XML file: %s", path)
        tree = ET.parse(path)
        logger.info("tree: %s", tree)
        logger.info("tree.getroot(): %s", tree.getroot())
        return _extract_sentences_from_xml_root(tree.getroot())
    except Exception as exc:
        logger.warning("Failed to parse XML file %s: %s", path, exc)
        return []


def split_dogri_sentences(text: str) -> List[str]:
    if not text:
        return []
    normalized = text.replace("\r", "\n")
    segments = re.split(r'(?<=[|।])', normalized)
    sentences = []
    for segment in segments:
        cleaned = segment.strip()
        cleaned = cleaned[:-1]
        if cleaned:
            sentences.append(cleaned+" |")
    return sentences


def _series_to_chart(series: pd.Series) -> List[Dict[str, Any]]:
    if series is None or series.empty:
        return []
    total = int(series.sum())
    rows = []
    for label, value in series.items():
        safe_label = str(label) if pd.notna(label) and str(label).strip() else "Unknown"
        safe_value = int(value)
        percentage = round((safe_value / total) * 100, 2) if total else 0
        rows.append({"label": safe_label, "value": safe_value, "percentage": percentage})
    return rows


def _load_excel_from_path(path: str) -> tuple[pd.DataFrame, str]:
    excel = pd.ExcelFile(path)
    if not excel.sheet_names:
        raise ValueError("Excel file does not contain any sheets.")
    sheet_name = excel.sheet_names[0]
    frame = excel.parse(sheet_name=sheet_name)
    return frame, sheet_name


def _load_excel_from_bytes(raw_bytes: bytes) -> tuple[pd.DataFrame, str]:
    buffer = io.BytesIO(raw_bytes)
    excel = pd.ExcelFile(buffer)
    if not excel.sheet_names:
        raise ValueError("Uploaded Excel does not contain any sheets.")
    sheet_name = excel.sheet_names[0]
    frame = excel.parse(sheet_name=sheet_name)
    return frame, sheet_name


def _validate_analytics_frame(frame: pd.DataFrame):
    missing = [col for col in ANALYTICS_REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {', '.join(missing)}")


def _compute_analytics_payload(
    frame: pd.DataFrame,
    *,
    source: str,
    file_name: Optional[str],
    sheet_name: Optional[str],
) -> Dict[str, Any]:
    if frame.empty:
        raise ValueError("Dataset is empty after reading the first sheet.")

    _validate_analytics_frame(frame)
    working = frame.copy()
    for col in ["category", "sub-category", "Tagg"]:
        if col in working.columns:
            working[col] = working[col].fillna("Unknown")

    total_tokens = int(len(working))
    total_sentences = int(working["Sentences"].nunique()) if "Sentences" in working.columns else 0
    unique_categories = int(working["category"].nunique()) if "category" in working.columns else 0
    unique_sub_categories = int(working["sub-category"].nunique()) if "sub-category" in working.columns else 0

    tag_distribution = _series_to_chart(working["Tagg"].value_counts()) if "Tagg" in working.columns else []
    token_by_category = (
        _series_to_chart(working.groupby("category")["token"].count()) if "category" in working.columns else []
    )
    sentence_distribution = (
        _series_to_chart(working.groupby("category")["Sentences"].nunique())
        if "category" in working.columns and "Sentences" in working.columns
        else []
    )
    sub_category_distribution = (
        _series_to_chart(working["Sub-category"].value_counts()) if "Sub-category" in working.columns else []
    )

    generated_at = datetime.utcnow().isoformat() + "Z"
    notes = (
        "Only the first sheet is processed. Ensure the uploaded Excel strictly follows the required column schema."
    )

    return {
        "source": source,
        "file_name": file_name,
        "sheet_name": sheet_name,
        "generated_at": generated_at,
        "total_tokens": total_tokens,
        "total_sentences": total_sentences,
        "unique_categories": unique_categories,
        "unique_sub_categories": unique_sub_categories,
        "tag_distribution": tag_distribution,
        "token_by_category": token_by_category,
        "sentence_distribution": sentence_distribution,
        "sub_category_distribution": sub_category_distribution,
        "required_columns": ANALYTICS_REQUIRED_COLUMNS,
        "notes": notes,
    }


def load_default_analytics_dataset(force_reload: bool = False) -> Dict[str, Any]:
    global ANALYTICS_CACHE
    if ANALYTICS_CACHE is not None and not force_reload:
        return ANALYTICS_CACHE
    if not DEFAULT_ANALYTICS_DATASET or not os.path.isfile(DEFAULT_ANALYTICS_DATASET):
        raise FileNotFoundError(f"Default analytics dataset not found at {DEFAULT_ANALYTICS_DATASET}")
    frame, sheet_name = _load_excel_from_path(DEFAULT_ANALYTICS_DATASET)
    payload = _compute_analytics_payload(
        frame,
        source="default",
        file_name=os.path.basename(DEFAULT_ANALYTICS_DATASET),
        sheet_name=sheet_name,
    )
    ANALYTICS_CACHE = payload
    return payload


def process_uploaded_analytics_file(raw_bytes: bytes, file_name: Optional[str]) -> Dict[str, Any]:
    if not raw_bytes:
        raise ValueError("Uploaded file is empty.")
    frame, sheet_name = _load_excel_from_bytes(raw_bytes)
    return _compute_analytics_payload(
        frame,
        source="uploaded",
        file_name=file_name,
        sheet_name=sheet_name,
    )


def load_manual_sentence_corpus(force_reload: bool = False) -> List[str]:
    """Load all sentences from dataset-sent directory or fallback samples."""
    global MANUAL_SENTENCE_CACHE
    if MANUAL_SENTENCE_CACHE is not None and not force_reload:
        return MANUAL_SENTENCE_CACHE

    sentences: List[str] = []
    folder = os.path.abspath(DEFAULT_SENTENCE_FOLDER)
    if os.path.isdir(folder):
        logger.info("Loading manual sentences from %s", folder)
        for file_name in sorted(os.listdir(folder)):
            lower_name = file_name.lower()
            logger.info("file_name: %s", file_name)
            if not lower_name.endswith(DEFAULT_SENTENCE_EXTENSIONS):
                logger.info("file_name does not end with DEFAULT_SENTENCE_EXTENSIONS: %s", file_name)
                continue
            path = os.path.join(folder, file_name)
            try:
                if lower_name.endswith(".xml"):
                    sentences.extend(parse_xml_file(path))
                    content = " ".join(sentences)
                    sentences.extend(split_dogri_sentences(content))
                else:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                    sentences.extend(split_dogri_sentences(content))
            except Exception as exc:
                logger.warning("Failed to read %s: %s", path, exc)
    else:
        logger.warning("Manual sentence folder not found: %s", folder)

    if not sentences:
        sentences = [
            "छड़े 12 . 6 फीसदी भारतीय कौंपनियें दा छंटनी उप्पर बचार : सर्वेक्षण दिल्ली , 22 फरवरी |",
            "अरुणा ने फ्ही बी कोई जवाब नेईं दित्ता अरुणा।",
        ]

    MANUAL_SENTENCE_CACHE = sentences
    logger.info("Manual sentence corpus size: %d", len(MANUAL_SENTENCE_CACHE))
    return MANUAL_SENTENCE_CACHE


def append_manual_sentences(new_sentences: List[str]) -> int:
    global MANUAL_SENTENCE_CACHE
    sentences = load_manual_sentence_corpus()
    added = 0
    for sentence in new_sentences:
        trimmed = sentence.strip()
        if trimmed:
            sentences.append(trimmed)
            added += 1
    MANUAL_SENTENCE_CACHE = sentences
    return added


def load_models():
    global MODELS_CACHE
    if MODELS_CACHE is not None:
        return MODELS_CACHE

    model_path = "C:/Users/KEVAL/Desktop/post_dogri/models"
    svm_model = joblib.load(os.path.join(model_path, "svm_model_200k.pkl"))
    hmm_tagger = pickle.load(open(os.path.join(model_path, "hmm_pos_tagger.pkl"), "rb"))
    bilstm_model = load_model(os.path.join(model_path, "bilstm2", "bilstm_pos_tagger.keras"))
    bert_tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, "mbert"))
    bert_model = AutoModelForTokenClassification.from_pretrained(os.path.join(model_path, "mbert"))

    with open(os.path.join(model_path, "bilstm2", "token_encoder.pkl"), "rb") as f:
        token_encoder = pickle.load(f)
    with open(os.path.join(model_path, "bilstm2", "tag_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)

    with open(os.path.join(model_path, "mbert", "mbert_dogri_label_mappings.json"), "r", encoding="utf-8") as f:
        label_mappings = json.load(f)
    id2tag = label_mappings["id2tag"]

    token_lookup = {token: idx for idx, token in enumerate(token_encoder.classes_)}
    unk_index = token_lookup.get("<UNK>")
    if unk_index is None:
        logger.warning("'<UNK>' token not found in encoder; falling back to index 0 for unknown tokens.")
        unk_index = 0

    gemini_client = None
    if GEMINI_AVAILABLE:
        try:
            from dotenv import load_dotenv

            load_dotenv()
            api_key = (
                os.getenv("GOOGLE_API_KEY")
                or os.getenv("google_api")
                or os.getenv("GEMINI_API_KEY")
                or os.getenv("GEMINI_API")
            )
            if api_key and api_key.strip():
                gemini_client = genai.Client(api_key=api_key.strip())
            else:
                logger.warning("Gemini API key not found. Gemini-based models will be disabled.")
        except Exception as exc:
            logger.warning("Could not initialize Gemini client: %s", exc)

    MODELS_CACHE = {
        "svm": svm_model,
        "hmm": hmm_tagger,
        "bilstm": bilstm_model,
        "bert_tokenizer": bert_tokenizer,
        "bert": bert_model,
        "token_encoder": token_encoder,
        "token_lookup": token_lookup,
        "token_unk_index": unk_index,
        "label_encoder": label_encoder,
        "id2tag": id2tag,
        "metrics": MODEL_METRICS,
        "gemini_client": gemini_client,
    }
    return MODELS_CACHE


def extract_features(tokens, index):
    token = tokens[index]
    features = {
        "token": token,
        "prefix1": token[:1] if len(token) > 0 else "",
        "prefix2": token[:2] if len(token) > 1 else token[:1] if len(token) > 0 else "",
        "suffix1": token[-1:] if len(token) > 0 else "",
        "suffix2": token[-2:] if len(token) > 1 else token[-1:] if len(token) > 0 else "",
        "length": len(token),
    }
    features["prev_token"] = tokens[index - 1] if index > 0 else "<START>"
    features["next_token"] = tokens[index + 1] if index < len(tokens) - 1 else "<END>"
    return features


def predict_svm(tokens, models):
    start_time = time.time()
    features_list = [extract_features(tokens, i) for i in range(len(tokens))]
    predictions = [models["svm"].predict([features])[0] for features in features_list]
    elapsed_time = time.time() - start_time
    return predictions, elapsed_time


def predict_hmm(tokens, models):
    start_time = time.time()
    predictions = [tag for _, tag in models["hmm"].tag(tokens)]
    elapsed_time = time.time() - start_time
    return predictions, elapsed_time


def predict_bilstm(tokens, models):
    start_time = time.time()
    lookup = models.get("token_lookup")
    unk_index = models.get("token_unk_index", 0)
    if lookup is None:
        raise RuntimeError("Token lookup not available for BiLSTM model.")
    token_encoded = [lookup.get(t, unk_index) for t in tokens]
    X_input = np.array(token_encoded).reshape(len(tokens), 1)
    y_pred = models["bilstm"].predict(X_input)
    predictions = models["label_encoder"].inverse_transform(np.argmax(y_pred, axis=2).flatten())
    elapsed_time = time.time() - start_time
    return predictions.tolist(), elapsed_time


def predict_bert(tokens, models):
    start_time = time.time()
    inputs = models["bert_tokenizer"](tokens, return_tensors="pt", is_split_into_words=True, truncation=True, padding=True)
    with torch.no_grad():
        outputs = models["bert"](**inputs).logits
    predictions = torch.argmax(outputs, dim=-1).squeeze().tolist()
    word_ids = inputs.word_ids()
    if isinstance(predictions, int):
        predictions = [predictions]
    if word_ids is None:
        word_ids = []
    aligned_predictions = []
    previous_word_idx = None
    for word_idx, pred in zip(word_ids, predictions):
        if word_idx is not None and word_idx != previous_word_idx:
            if word_idx < len(tokens):
                tag = models["id2tag"].get(str(pred), "X")
                aligned_predictions.append((tokens[word_idx], tag))
        previous_word_idx = word_idx
    if not aligned_predictions and tokens:
        aligned_predictions = [(token, "X") for token in tokens]
    tag_list = [tag for _, tag in aligned_predictions]
    if len(tag_list) < len(tokens):
        tag_list.extend(["X"] * (len(tokens) - len(tag_list)))
    elif len(tag_list) > len(tokens):
        tag_list = tag_list[: len(tokens)]
    elapsed_time = time.time() - start_time
    return tag_list, elapsed_time


def predict_gemma(tokens, models):
    start_time = time.time()
    if not GEMINI_AVAILABLE or models.get("gemini_client") is None:
        return ["X"] * len(tokens), 0.0
    try:
        sentence = " ".join(tokens)
        prompt = f"""You are a dogri langauge expert.
You are given a sentence in dogri langauge.
you need to provide pos tags for the sentence from the given list of pos tags.

pos tags:
• Noun (N): NC (Common Noun), NP (Proper Noun), NV (Verbal Noun), NST (Spatiotemporal Noun)
• Verb (V): VM (Main verb), VA (Auxiliary verb)
• Pronoun (P): PPR (Pronominal)
• Nominal Modifiers (J): JJ (Adjective), JQ (Quantifier)
• Demonstratives (D): DAB (Absolute Demonstrative)
• Adverb (A): AMN (Manner Adverb)
• Postposition (PP): PP (Postposition)
• Particles (C): CCD (Coordinating Particle)
• Numeral (NUM): NUMR (Real Numbers)
• Residual (RD): RDF (Foreign Word)
• Punctuation (PU): PU (Punctuation)

{sentence}

Example:
INPUT: अरुणा ने फ्ही बी कोई जवाब नेईं दित्ता अरुणा |
OUTPUT: POS TAG : अरुणा/ N_NP   ने/ PP_PP   फ्ही/ A_AMN   बी/ J_JQ   कोई/ J_JQ   जवाब/ N_NC   नेईं/ V_VA   दित्ता/ V_VM   अरुणा/ N_NP   ।/ PU_PU

Provide the most appropriate POS tag in the following format with no additional text or commentary:
OUTPUT:
POS TAG :"""
        client = models["gemini_client"]
        response = client.models.generate_content(model="gemma-3-27b-it", contents=prompt)
        response_text = response.text if hasattr(response, "text") else str(response)
        pos_line = None
        for line in response_text.split("\n"):
            if "POS TAG" in line or "OUTPUT" in line:
                pos_line = line
                break
        if pos_line is None:
            return ["X"] * len(tokens), time.time() - start_time
        matches = re.findall(r"(\S+?)\s*/\s*([A-Z_]+)", pos_line)
        predictions = []
        if matches:
            tag_map = {token: tag for token, tag in matches}
            for token in tokens:
                tag = tag_map.get(token)
                if tag:
                    predictions.append(tag)
                else:
                    match = next((v for k, v in tag_map.items() if k.strip() == token.strip()), None)
                    predictions.append(match or "X")
        else:
            predictions = ["X"] * len(tokens)
        if len(predictions) < len(tokens):
            predictions.extend(["X"] * (len(tokens) - len(predictions)))
        elif len(predictions) > len(tokens):
            predictions = predictions[: len(tokens)]
        elapsed_time = time.time() - start_time
        return predictions, elapsed_time
    except Exception as exc:
        logger.warning("Gemma prediction failed: %s", exc)
        return ["X"] * len(tokens), 0.0


def predict_phi4(tokens, models):
    return predict_gemma(tokens, models)


def normalize_model_name(name: str) -> str:
    if not name:
        return ""
    name = name.strip()
    for option in BASE_MODEL_OPTIONS + list(HYBRID_MODEL_COMBINATIONS.keys()):
        if option.lower() == name.lower():
            return option
    return name


def get_model_predictions(model_name, tokens, models):
    if model_name == "SVM":
        return predict_svm(tokens, models)
    if model_name == "HMM":
        return predict_hmm(tokens, models)
    if model_name == "BiLSTM":
        return predict_bilstm(tokens, models)
    if model_name == "mBERT":
        preds, elapsed = predict_bert(tokens, models)
        return preds, elapsed
    if model_name == "Gemma-FineTuned":
        return predict_gemma(tokens, models)
    if model_name == "phi-4-fine-tuned":
        return predict_phi4(tokens, models)
    raise ValueError(f"Unsupported model: {model_name}")


def create_hybrid_predictions(model1_preds, model2_preds, priority_model=2):
    if len(model1_preds) != len(model2_preds):
        min_len = min(len(model1_preds), len(model2_preds))
        model1_preds = model1_preds[:min_len]
        model2_preds = model2_preds[:min_len]
    hybrid_preds = []
    agreement_count = 0
    for pred1, pred2 in zip(model1_preds, model2_preds):
        if pred1 == pred2:
            hybrid_preds.append(pred1)
            agreement_count += 1
        else:
            hybrid_preds.append(pred1 if priority_model == 1 else pred2)
    agreement_rate = agreement_count / len(model1_preds) if model1_preds else 0
    return hybrid_preds, agreement_rate


class PredictRequest(BaseModel):
    sentence: str = Field(..., description="Input sentence in Dogri")
    models: List[str] = Field(default_factory=lambda: BASE_MODEL_OPTIONS.copy(), description="List of models to run")


class PredictResponse(BaseModel):
    tokens: List[str]
    results: Dict[str, List[str]]
    timings: Dict[str, float]
    agreements: Dict[str, float] = {}


class SinglePredictRequest(BaseModel):
    sentence: str = Field(..., description="Input sentence in Dogri")


class SinglePredictResponse(BaseModel):
    tokens: List[str]
    tags: List[str]
    timing: float
    agreement: Optional[float] = None
    naming: str


class ManualSentenceUpload(BaseModel):
    sentences: List[str]


class FileModelResult(BaseModel):
    tags: List[str]
    timing: float
    agreement: Optional[float] = None


class FileSentenceResult(BaseModel):
    sentence: str
    tokens: List[str]
    models: Dict[str, FileModelResult]


class FilePredictResponse(BaseModel):
    download_name: str
    content: str
    sentences: List[FileSentenceResult]


def run_model_inference(
    sentence: str,
    target_model: str,
    models_cache: Dict,
    tokens_override: Optional[List[str]] = None,
    base_cache: Optional[Dict[str, tuple]] = None,
):
    model_name = normalize_model_name(target_model)
    if model_name not in BASE_MODEL_OPTIONS and model_name not in HYBRID_MODEL_COMBINATIONS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {target_model}")

    if tokens_override is not None:
        tokens = tokens_override
        cleaned_sentence = sentence.strip()
        if not cleaned_sentence and not tokens:
            raise HTTPException(status_code=400, detail="Sentence must not be empty.")
    else:
        cleaned_sentence = sentence.strip()
        if not cleaned_sentence:
            raise HTTPException(status_code=400, detail="Sentence must not be empty.")
        tokens = indic_tokenize.trivial_tokenize(cleaned_sentence)
        if not tokens:
            tokens = cleaned_sentence.split()
        if not tokens:
            raise HTTPException(status_code=400, detail="Unable to tokenize sentence.")

    required_models = set()
    if model_name in BASE_MODEL_OPTIONS:
        required_models.add(model_name)
    else:
        required_models.update(HYBRID_MODEL_COMBINATIONS[model_name])

    if base_cache is None:
        base_cache = {}
    base_predictions = {}
    base_timings = {}
    for base in required_models:
        if base in base_cache:
            preds, elapsed = base_cache[base]
        else:
            preds, elapsed = get_model_predictions(base, tokens, models_cache)
            base_cache[base] = (preds, elapsed)
        base_predictions[base] = preds
        base_timings[base] = elapsed

    if model_name in BASE_MODEL_OPTIONS:
        return tokens, base_predictions[model_name], base_timings[model_name], None

    base1, base2 = HYBRID_MODEL_COMBINATIONS[model_name]
    preds1 = base_predictions.get(base1)
    preds2 = base_predictions.get(base2)
    if preds1 is None or preds2 is None:
        raise HTTPException(status_code=500, detail=f"Hybrid components missing for {model_name}")
    hybrid_preds, agreement = create_hybrid_predictions(preds1, preds2, priority_model=2)
    timing = base_timings.get(base1, 0) + base_timings.get(base2, 0)
    return tokens, hybrid_preds, timing, agreement


app = FastAPI(title="Dogri Tagging Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    load_models()
    load_manual_sentence_corpus()
    try:
        load_default_analytics_dataset()
        logger.info("Default analytics dataset loaded and cached.")
    except Exception as exc:  # pragma: no cover - best-effort logging
        logger.warning("Failed to preload analytics dataset: %s", exc)
    logger.info("Models loaded and ready.")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/models")
def list_models(include_metrics: bool = False):
    payload = {
        "base": BASE_MODEL_OPTIONS,
        "hybrid": list(HYBRID_MODEL_COMBINATIONS.keys()),
    }
    if include_metrics:
        payload["metrics"] = MODEL_METRICS
    return payload


@app.get("/manual/sentences")
def list_manual_sentences(limit: int = 200, offset: int = 0):
    data = load_manual_sentence_corpus()
    safe_limit = max(1, min(1000, limit))
    safe_offset = max(0, offset)
    sliced = data[safe_offset : safe_offset + safe_limit]
    return {"sentences": sliced, "total": len(data), "limit": safe_limit, "offset": safe_offset}


@app.post("/manual/sentences")
def add_manual_sentences(payload: ManualSentenceUpload):
    added = append_manual_sentences(payload.sentences)
    total = len(load_manual_sentence_corpus())
    return {"added": added, "total": total}


@app.get("/analytics")
def get_default_analytics(force_reload: bool = False):
    try:
        return load_default_analytics_dataset(force_reload=force_reload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/analytics/upload")
async def upload_analytics(file: UploadFile = File(...)):
    raw_bytes = await file.read()
    try:
        payload = process_uploaded_analytics_file(raw_bytes, file.filename)
        # Make uploaded dataset the active analytics until process restart or another upload
        global ANALYTICS_CACHE
        ANALYTICS_CACHE = payload
        return payload
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    models_cache = load_models()
    sentence = request.sentence
    results = {}
    timings = {}
    agreements = {}
    naming = {}
    tokens_reference: Optional[List[str]] = None
    base_cache: Dict[str, tuple] = {}

    normalized_models = [normalize_model_name(name) for name in request.models]
    if not normalized_models:
        raise HTTPException(status_code=400, detail="No models requested.")

    for model_name in normalized_models:
        tokens, tags, timing, agreement = run_model_inference(
            sentence,
            model_name,
            models_cache,
            tokens_override=tokens_reference,
            base_cache=base_cache,
        )
        if tokens_reference is None:
            tokens_reference = tokens
        results[model_name] = tags
        timings[model_name] = timing
        naming[model_name] = model_name
        if agreement is not None:
            agreements[model_name] = agreement

    return PredictResponse(tokens=tokens_reference, results=results, naming=naming, timings=timings, agreements=agreements)


def _format_file_tagged_output(file_results: List[Dict]) -> str:
    lines = []
    for idx, entry in enumerate(file_results, start=1):
        lines.append(f"Sentence {idx}: {entry['sentence']}")
        tokens = entry.get("tokens") or []
        for model_name, payload in entry.get("models", {}).items():
            tags = payload.get("tags") or []
            token_pairs = " ".join(f"{tok}/{tag}" for tok, tag in zip(tokens, tags))
            lines.append(f"{model_name}: {token_pairs}".strip())
        lines.append("")
    return "\n".join(lines).strip()


@app.post("/predict/file", response_model=FilePredictResponse)
async def predict_from_file(models: str = Form(...), file: UploadFile = File(...)):
    logger.info("/predict/file invoked; raw models payload: %s", models)
    models_cache = load_models()
    if not models:
        raise HTTPException(status_code=400, detail="Models payload is required.")
    try:
        parsed_models = json.loads(models)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Models payload must be a JSON array.")
    if isinstance(parsed_models, str):
        parsed_models = [parsed_models]
    if not isinstance(parsed_models, list) or not parsed_models:
        raise HTTPException(status_code=400, detail="At least one model must be specified.")

    normalized_models = [normalize_model_name(name) for name in parsed_models if name]
    normalized_models = [name for name in normalized_models if name]
    if not normalized_models:
        raise HTTPException(status_code=400, detail="No valid models found in payload.")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    decoded_text = None
    for encoding in ("utf-8", "utf-8-sig", "utf-16"):
        try:
            decoded_text = raw_bytes.decode(encoding).strip()
            if decoded_text:
                break
        except UnicodeDecodeError:
            continue
    if not decoded_text:
        raise HTTPException(status_code=400, detail="Unable to decode uploaded file content.")

    sentences = split_dogri_sentences(decoded_text)
    if not sentences:
        sentences = [decoded_text]

    base_name = os.path.splitext(file.filename or "input")[0]
    download_name = f"{base_name}_tagged.txt"

    sentence_payloads: List[Dict] = []
    for sentence in sentences:
        tokens_reference: Optional[List[str]] = None
        base_cache: Dict[str, tuple] = {}
        model_results: Dict[str, Dict] = {}
        for model_name in normalized_models:
            tokens, tags, timing, agreement = run_model_inference(
                sentence,
                model_name,
                models_cache,
                tokens_override=tokens_reference,
                base_cache=base_cache,
            )
            if tokens_reference is None:
                tokens_reference = tokens
            model_results[model_name] = {
                "tags": tags,
                "timing": timing,
                "agreement": agreement,
                "Model_Used":model_name
            }
        sentence_payloads.append(
            {
                "sentence": sentence,
                "tokens": tokens_reference or [],
                "models": model_results,
            }
        )

    content = _format_file_tagged_output(sentence_payloads)
    return FilePredictResponse(download_name=download_name, content=content, sentences=sentence_payloads)


@app.post("/predict/{model_name}", response_model=SinglePredictResponse)
def predict_single(model_name: str, request: SinglePredictRequest):
    models_cache = load_models()
    normalized_name = normalize_model_name(model_name)

    tokens, tags, timing, agreement = run_model_inference(
        request.sentence, normalized_name, models_cache, base_cache={}
    )

    return SinglePredictResponse(
        tokens=tokens,
        tags=tags,
        timing=timing,
        agreement=agreement,
        naming=normalized_name

    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


