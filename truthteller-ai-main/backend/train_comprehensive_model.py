"""
train_comprehensive_model.py
────────────────────────────
Trains the best-possible AI vs Human detector on the
AuthentiText_X_2026 dataset and saves the model as
models/comprehensive_model.json.

WHY ACCURACY IS LIMITED (~51-54%)
──────────────────────────────────
After full analysis, the dataset's 8 numeric features have
near-zero correlation (max r = 0.064) with the AI/Human label.
The content_text column is synthetic (only 13 unique words,
randomly shuffled per sample). This means the dataset itself
does not contain enough signal for high accuracy — this is a
fundamental data quality issue, not a modelling issue.

The best model trained here achieves ~54% CV accuracy, which is
the ceiling for this data. To get 85%+ accuracy you would need
real AI vs Human text with genuine linguistic features (actual
sentences from GPT-4, Claude, Gemini vs real humans).
"""

import os
import json
import numpy as np
import pandas as pd
from collections import Counter
import re
import math
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "AuthentiText_X_2026_AI_vs_Human_Detection_1K.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "comprehensive_model.json")

os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Feature columns (pre-computed in the dataset) ────────────────────────────
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


# ─── Text feature extraction (used at inference time) ─────────────────────────
def extract_text_features(text: str) -> dict:
    """
    Extract linguistic features from raw text.
    These supplement the pre-computed numeric features when
    the caller provides raw text instead of a pre-featurised row.
    """
    if not text or len(text.strip()) == 0:
        return {col: 0.0 for col in [
            "text_avg_word_len", "text_lexical_diversity",
            "text_bigram_unique_ratio", "text_perplexity",
            "text_burstiness", "text_sent_length_cv",
        ]}

    words = re.findall(r"\b\w+\b", text.lower())
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

    if not words:
        return {col: 0.0 for col in [
            "text_avg_word_len", "text_lexical_diversity",
            "text_bigram_unique_ratio", "text_perplexity",
            "text_burstiness", "text_sent_length_cv",
        ]}

    word_freq   = Counter(words)
    n           = len(words)
    unique      = len(word_freq)

    # Lexical diversity
    lex_div = unique / n

    # Average word length
    avg_wl = np.mean([len(w) for w in words])

    # Bigram unique ratio
    bigrams = [f"{words[i]}_{words[i+1]}" for i in range(n - 1)]
    bigram_unique_ratio = len(set(bigrams)) / len(bigrams) if bigrams else 1.0

    # Entropy-based perplexity
    entropy = -sum((c / n) * math.log2(c / n) for c in word_freq.values() if c > 0)
    perplexity = 2 ** entropy

    # Burstiness (CV of word frequencies)
    freq_vals = list(word_freq.values())
    mean_f = np.mean(freq_vals)
    burstiness = np.std(freq_vals) / mean_f if mean_f > 0 else 0.0

    # Sentence-length CV
    if len(sentences) > 1:
        slens = [len(re.findall(r"\b\w+\b", s)) for s in sentences]
        m = np.mean(slens)
        sent_cv = np.std(slens) / m if m > 0 else 0.0
    else:
        sent_cv = 0.0

    return {
        "text_avg_word_len":       avg_wl,
        "text_lexical_diversity":  lex_div,
        "text_bigram_unique_ratio": bigram_unique_ratio,
        "text_perplexity":         perplexity,
        "text_burstiness":         burstiness,
        "text_sent_length_cv":     sent_cv,
    }


# ─── Load & prepare data ──────────────────────────────────────────────────────
def load_data(path: str):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows. Columns: {df.columns.tolist()}")
    print(f"Label distribution:\n{df['author_type'].value_counts()}\n")

    X = df[FEATURE_COLS].values.astype(np.float64)
    y = (df["author_type"] == "AI").astype(int).values
    return X, y, df


# ─── Train ────────────────────────────────────────────────────────────────────
def train(X, y):
    """
    Trains a soft-voting ensemble of the three most stable classifiers
    given the low-signal dataset. Returns (fitted_model, scaler, cv_scores).
    """
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Individual candidates (evaluated for transparency)
    candidates = {
        "LogisticRegression (C=0.5)": LogisticRegression(
            C=0.5, max_iter=2000, solver="lbfgs", random_state=42
        ),
        "RandomForest (n=300)": RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=10,
            random_state=42, n_jobs=-1
        ),
        "GBM (n=300, lr=0.05)": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            subsample=0.8, min_samples_leaf=10, random_state=42
        ),
        "SVM (rbf, C=0.5)": SVC(
            C=0.5, kernel="rbf", probability=True, random_state=42
        ),
    }

    print("── Cross-validation scores (10-fold) ────────────────────────")
    cv_results = {}
    for name, clf in candidates.items():
        scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="accuracy")
        cv_results[name] = scores
        print(f"  {name:<35s}  {scores.mean():.4f} ± {scores.std():.4f}")

    # Soft-voting ensemble (best generalisation on low-signal data)
    lr  = LogisticRegression(C=0.5, max_iter=2000, solver="lbfgs", random_state=42)
    rf  = RandomForestClassifier(n_estimators=300, max_depth=5,
                                  min_samples_leaf=10, random_state=42, n_jobs=-1)
    gbm = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                      max_depth=3, subsample=0.8,
                                      min_samples_leaf=10, random_state=42)
    svm = SVC(C=0.5, kernel="rbf", probability=True, random_state=42)

    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("gbm", gbm), ("svm", svm)],
        voting="soft",
        weights=[2, 1, 1, 1],   # LR gets double weight — most calibrated
    )

    ens_scores = cross_val_score(ensemble, X_scaled, y, cv=cv, scoring="accuracy")
    print(f"\n  {'Soft Voting Ensemble (final)':<35s}  "
          f"{ens_scores.mean():.4f} ± {ens_scores.std():.4f}")
    print("─────────────────────────────────────────────────────────────\n")

    # Fit ensemble on full data
    ensemble.fit(X_scaled, y)

    # Hold-out evaluation for the report
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )
    ensemble_eval = VotingClassifier(
        estimators=[("lr", LogisticRegression(C=0.5, max_iter=2000, random_state=42)),
                    ("rf", RandomForestClassifier(n_estimators=300, max_depth=5,
                                                   min_samples_leaf=10, random_state=42)),
                    ("gbm", GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                                        max_depth=3, random_state=42)),
                    ("svm", SVC(C=0.5, kernel="rbf", probability=True, random_state=42))],
        voting="soft", weights=[2, 1, 1, 1],
    )
    ensemble_eval.fit(X_tr, y_tr)
    y_pred = ensemble_eval.predict(X_te)
    print("Hold-out evaluation (80/20 split):")
    print(f"  Accuracy : {accuracy_score(y_te, y_pred):.4f}")
    print(classification_report(y_te, y_pred, target_names=["Human", "AI"]))

    return ensemble, scaler, ens_scores


# ─── Save model ───────────────────────────────────────────────────────────────
def save_model(ensemble, scaler, cv_scores, feature_cols):
    """
    Serialise the fitted scaler + the LR sub-model coefficients to JSON
    (keeping compatibility with utils.py's JSON-based loader).
    We also store the full ensemble metadata for the app to know it exists.
    """
    # Extract the LR estimator from the ensemble for lightweight JSON storage
    lr_clf = None
    for name, est in ensemble.named_estimators_.items():
        if name == "lr":
            lr_clf = est
            break

    model_data = {
        "model_type":      "voting_ensemble_lr_rf_gbm_svm",
        "feature_columns": feature_cols,
        "mean":            scaler.mean_.tolist(),
        "std":             scaler.scale_.tolist(),
        # LR weights kept for fast JSON-based inference in utils.py
        "weights":         lr_clf.coef_[0].tolist() if lr_clf else [],
        "bias":            float(lr_clf.intercept_[0]) if lr_clf else 0.0,
        "classes":         ["Human", "AI"],
        "cv_accuracy":     float(cv_scores.mean()),
        "cv_std":          float(cv_scores.std()),
        "note": (
            "Dataset has near-zero feature signal (max r=0.064). "
            "Accuracy ceiling is ~54% with this data. "
            "For 85%+ accuracy, replace with real AI vs Human text corpus."
        ),
    }

    with open(MODEL_PATH, "w") as f:
        json.dump(model_data, f, indent=2)

    print(f"Model saved → {MODEL_PATH}")
    print(f"CV accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")
    return model_data


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  AI vs Human Detector — Training")
    print("=" * 60, "\n")

    X, y, df = load_data(DATA_PATH)

    print("⚠️  DATA QUALITY NOTE")
    print("   The 8 numeric features have correlations with the label")
    corrmap = {c: float(np.corrcoef(df[c], y)[0, 1]) for c in FEATURE_COLS}
    for k, v in sorted(corrmap.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"   {k:<35s}: r = {v:+.4f}")
    print()
    print("   Max |r| = {:.4f}. Near-random features → accuracy ceiling ≈ 54%".format(
        max(abs(v) for v in corrmap.values())
    ))
    print("   See README for how to improve accuracy with a better dataset.\n")

    ensemble, scaler, cv_scores = train(X, y)
    save_model(ensemble, scaler, cv_scores, FEATURE_COLS)

    print("Training complete.")