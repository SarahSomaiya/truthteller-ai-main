"""
utils.py
────────
File parsing, feature extraction, and text analysis helpers.

FIXES vs original:
  • Replaced PyPDF2 (abandoned) with pypdf (actively maintained)
  • Removed calls to load_ml_model() / load_simple_model() which
    were undefined in this file (they live in app.py)
  • predict_comprehensive() now uses the stored LR weights from the
    JSON model instead of hard-coded rule thresholds
  • analyze_text() cascades cleanly: comprehensive → fallback
  • extract_comprehensive_features() now returns 21 features
    consistently (added missing features that caused index errors)
"""

import os
import json
import math
import re
import numpy as np
from collections import Counter

# pypdf replaces the abandoned PyPDF2 — drop-in compatible API
try:
    from pypdf import PdfReader
except ImportError:
    try:
        import PyPDF2
        class PdfReader:   # shim for legacy installs
            def __init__(self, f):
                self._r = PyPDF2.PdfReader(f)
            @property
            def pages(self):
                return self._r.pages
    except ImportError:
        PdfReader = None

from docx import Document
from pptx import Presentation
from werkzeug.utils import secure_filename
import joblib

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# ─── NLTK data ────────────────────────────────────────────────────────────────
for _resource, _path in [
    ("tokenizers/punkt", "punkt"),
    ("corpora/stopwords", "stopwords"),
]:
    try:
        nltk.data.find(_resource)
    except LookupError:
        try:
            nltk.download(_path, quiet=True)
        except Exception:
            pass

# ─── Constants ────────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {"pdf", "docx", "pptx"}

# Feature columns expected by the comprehensive model
FEATURE_COLS = [
    "prompt_complexity_score",
    "perplexity_score",
    "burstiness_index",
    "syntactic_variability",
    "semantic_coherence_score",
    "lexical_diversity_ratio",
    "readability_grade_level",
    "generation_confidence_score",
]

# ─── Model loading ────────────────────────────────────────────────────────────
COMPREHENSIVE_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "models", "comprehensive_model.json"
)

comprehensive_model = None


def load_comprehensive_model():
    """Load the comprehensive ML model from JSON."""
    global comprehensive_model
    try:
        if os.path.exists(COMPREHENSIVE_MODEL_PATH):
            with open(COMPREHENSIVE_MODEL_PATH, "r") as f:
                comprehensive_model = json.load(f)
            print("✓ Comprehensive model loaded successfully")
            print(f"  CV accuracy: {comprehensive_model.get('cv_accuracy', 'n/a'):.4f}")
            return True
        else:
            print("⚠  Comprehensive model not found — run train_comprehensive_model.py first")
            return False
    except Exception as e:
        print(f"✗ Error loading comprehensive model: {e}")
        return False


# Load on import
load_comprehensive_model()


# ─── Logistic-regression inference ───────────────────────────────────────────
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))


def predict_logistic(X, weights, bias):
    """Logistic regression forward pass. Returns probability array."""
    z = np.dot(X, weights) + bias
    return _sigmoid(z)


# ─── File helpers ─────────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    if PdfReader is None:
        print("No PDF library available. Install: pip install pypdf")
        return text
    try:
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error extracting PDF: {e}")
    return text


def extract_text_from_docx(file_path: str) -> str:
    text = ""
    try:
        doc = Document(file_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        print(f"Error extracting DOCX: {e}")
    return text


def extract_text_from_pptx(file_path: str) -> str:
    text = ""
    try:
        prs = Presentation(file_path)
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"
    except Exception as e:
        print(f"Error extracting PPTX: {e}")
    return text


def extract_text_from_file(file_path: str, file_extension: str) -> str:
    ext = file_extension.lower().lstrip(".")
    if ext == "pdf":
        return extract_text_from_pdf(file_path)
    elif ext == "docx":
        return extract_text_from_docx(file_path)
    elif ext == "pptx":
        return extract_text_from_pptx(file_path)
    return ""


# ─── Feature extraction from raw text ────────────────────────────────────────
def extract_comprehensive_features(text: str) -> list:
    """
    Extract 21 linguistic features from raw text.
    Returns a list of 21 floats (zeros for empty text).

    Feature index map:
     0  avg_word_length
     1  word_length_std
     2  lexical_diversity
     3  repetition_ratio
     4  unique_words
     5  avg_sent_length
     6  sent_length_std
     7  sent_length_variation
     8  punctuation_ratio
     9  capitalization_ratio
    10  function_word_ratio
    11  bigram_repetition
    12  trigram_repetition
    13  burstiness
    14  perplexity
    15  avg_sent_overlap
    16  total_words
    17  num_sentences
    18  punctuation_count
    19  capital_words
    20  function_word_count
    """
    N_FEATURES = 21
    zeros = [0.0] * N_FEATURES

    if not text or not text.strip():
        return zeros

    words = re.findall(r"\b\w+\b", text.lower())
    raw_words = re.findall(r"\b\w+\b", text)  # preserve case for capitalisation
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

    if not words:
        return zeros

    n = len(words)

    # 0-1: word length stats
    wlens = [len(w) for w in words]
    avg_word_length = float(np.mean(wlens))
    word_length_std = float(np.std(wlens))

    # 2-4: lexical diversity
    word_freq = Counter(words)
    unique_words = len(word_freq)
    lexical_diversity = unique_words / n
    most_common_freq = word_freq.most_common(1)[0][1]
    repetition_ratio = most_common_freq / n

    # 5-7: sentence stats
    if sentences:
        slens = [len(re.findall(r"\b\w+\b", s)) for s in sentences]
        avg_sent_length   = float(np.mean(slens))
        sent_length_std   = float(np.std(slens))
        sent_length_variation = sent_length_std / avg_sent_length if avg_sent_length > 0 else 0.0
    else:
        avg_sent_length = sent_length_std = sent_length_variation = 0.0

    # 8-10: punctuation / capitalisation / function words
    punctuation_count = len(re.findall(r"[.!?]", text))
    punctuation_ratio = punctuation_count / n

    capital_words = sum(1 for w in raw_words if w and w[0].isupper())
    capitalization_ratio = capital_words / n

    _FUNCTION_WORDS = frozenset([
        "the","a","an","and","or","but","in","on","at","to",
        "for","of","with","by","is","are","was","were","be",
    ])
    function_word_count = sum(1 for w in words if w in _FUNCTION_WORDS)
    function_word_ratio = function_word_count / n

    # 11-12: bigram / trigram repetition
    bigrams  = [f"{words[i]} {words[i+1]}"         for i in range(n - 1)]
    trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(n - 2)]

    bigram_repetition = (
        Counter(bigrams).most_common(1)[0][1] / len(bigrams) if bigrams else 0.0
    )
    trigram_repetition = (
        Counter(trigrams).most_common(1)[0][1] / len(trigrams) if trigrams else 0.0
    )

    # 13: burstiness (CV of unigram frequencies)
    freq_vals = list(word_freq.values())
    mean_f = float(np.mean(freq_vals))
    burstiness = float(np.std(freq_vals)) / mean_f if mean_f > 0 else 0.0

    # 14: entropy-based perplexity
    entropy = -sum((c / n) * math.log2(c / n) for c in word_freq.values() if c > 0)
    perplexity = float(2 ** entropy)

    # 15: avg Jaccard overlap between consecutive sentences
    if len(sentences) > 1:
        sw = [set(re.findall(r"\b\w+\b", s.lower())) for s in sentences]
        overlaps = []
        for i in range(len(sw) - 1):
            u = len(sw[i] | sw[i + 1])
            overlaps.append(len(sw[i] & sw[i + 1]) / u if u > 0 else 0.0)
        avg_overlap = float(np.mean(overlaps))
    else:
        avg_overlap = 0.0

    return [
        avg_word_length, word_length_std, lexical_diversity,
        repetition_ratio, float(unique_words),
        avg_sent_length, sent_length_std, sent_length_variation,
        punctuation_ratio, capitalization_ratio, function_word_ratio,
        bigram_repetition, trigram_repetition, burstiness, perplexity,
        avg_overlap, float(n), float(len(sentences)),
        float(punctuation_count), float(capital_words), float(function_word_count),
    ]


# ─── Inference helpers ────────────────────────────────────────────────────────
def _predict_with_lr_weights(features_scaled: np.ndarray, model: dict):
    """Use the stored LR weights from the JSON model."""
    weights = np.array(model["weights"])
    bias    = float(model["bias"])
    prob_ai = float(_sigmoid(np.dot(features_scaled, weights) + bias))
    pred    = "AI" if prob_ai >= 0.5 else "Human"
    conf    = max(prob_ai, 1.0 - prob_ai)
    return pred, round(conf, 3)


# ─── Main public API ──────────────────────────────────────────────────────────
def analyze_text(text: str) -> dict:
    """
    Analyse text and return {"prediction": "AI"|"Human", "confidence": float}.

    Priority:
      1. comprehensive_model (JSON with LR weights) — dataset features
      2. fallback heuristics
    """
    if not text or not text.strip():
        return {"prediction": "Unknown", "confidence": 0.0}

    if comprehensive_model is not None:
        try:
            # The JSON model was trained on the 8 pre-computed dataset features.
            # At inference we don't have those, so we approximate them from text.
            text_feats = _approximate_dataset_features(text)
            mean = np.array(comprehensive_model["mean"])
            std  = np.array(comprehensive_model["std"])
            # avoid divide-by-zero
            std  = np.where(std == 0, 1.0, std)
            feat_scaled = (np.array(text_feats) - mean) / std

            pred, conf = _predict_with_lr_weights(feat_scaled, comprehensive_model)
            return {"prediction": pred, "confidence": conf}

        except Exception as e:
            print(f"Comprehensive model inference error: {e}")

    # Fallback
    return _analyze_text_fallback(text)


def _approximate_dataset_features(text: str) -> list:
    """
    Map raw text → approximate values for the 8 dataset feature columns.
    These won't be perfect but let us re-use the trained LR weights.
    """
    words     = re.findall(r"\b\w+\b", text.lower())
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    n = max(len(words), 1)
    word_freq = Counter(words)

    # lexical_diversity_ratio
    lex_div = len(word_freq) / n

    # perplexity_score (entropy-based, scaled to ~10-100 range)
    entropy = -sum((c / n) * math.log2(c / n) for c in word_freq.values() if c > 0)
    perplexity = min(float(2 ** entropy), 100.0)

    # burstiness_index
    freq_vals = list(word_freq.values())
    mean_f = float(np.mean(freq_vals))
    burstiness = float(np.std(freq_vals)) / mean_f if mean_f > 0 else 0.0
    burstiness = min(burstiness, 1.0)

    # syntactic_variability — proxy: sentence-length CV
    if len(sentences) > 1:
        slens = [len(re.findall(r"\b\w+\b", s)) for s in sentences]
        m = np.mean(slens)
        syn_var = float(np.std(slens)) / m if m > 0 else 0.5
        syn_var = min(syn_var, 1.0)
    else:
        syn_var = 0.5

    # semantic_coherence_score — proxy: avg Jaccard overlap
    if len(sentences) > 1:
        sw = [set(re.findall(r"\b\w+\b", s.lower())) for s in sentences]
        olaps = []
        for i in range(len(sw) - 1):
            u = len(sw[i] | sw[i + 1])
            olaps.append(len(sw[i] & sw[i + 1]) / u if u > 0 else 0.0)
        coherence = float(np.mean(olaps))
    else:
        coherence = 0.5

    # prompt_complexity_score — proxy: type-token ratio normalised
    prompt_complexity = min(lex_div, 1.0)

    # readability_grade_level — proxy: avg words per sentence (rough Flesch-Kincaid proxy)
    avg_sent_len = n / max(len(sentences), 1)
    avg_word_len = np.mean([len(w) for w in words]) if words else 5.0
    readability  = max(0.0, min(0.39 * avg_sent_len + 11.8 * avg_word_len - 15.59, 20.0))

    # generation_confidence_score — proxy: bigram repeat density
    bigrams = [f"{words[i]}_{words[i+1]}" for i in range(n - 1)]
    if bigrams:
        bg_unique_ratio = len(set(bigrams)) / len(bigrams)
        gen_conf = 1.0 - bg_unique_ratio   # more repetition → higher "AI confidence"
    else:
        gen_conf = 0.5

    return [
        prompt_complexity,
        perplexity,
        burstiness,
        syn_var,
        coherence,
        lex_div,
        readability,
        gen_conf,
    ]


def _analyze_text_fallback(text: str) -> dict:
    """Heuristic fallback when no model is available."""
    words = re.findall(r"\b\w+\b", text.lower())
    n     = max(len(words), 1)

    # AI indicators
    word_freq    = Counter(words)
    lex_div      = len(word_freq) / n
    bigrams      = [f"{words[i]} {words[i+1]}" for i in range(n - 1)]
    bg_rep       = 1 - (len(set(bigrams)) / len(bigrams)) if bigrams else 0.0
    entropy      = -sum((c / n) * math.log2(c / n) for c in word_freq.values() if c > 0)
    perplexity   = 2 ** entropy

    ai_score = 0.0
    if lex_div < 0.4:    ai_score += 0.2
    if bg_rep  > 0.1:    ai_score += 0.2
    if perplexity < 50:  ai_score += 0.2

    prob_ai  = 0.5 + ai_score * 0.5
    pred     = "AI" if prob_ai >= 0.5 else "Human"
    conf     = round(min(max(prob_ai, 1.0 - prob_ai), 0.95), 3)
    return {"prediction": pred, "confidence": conf}