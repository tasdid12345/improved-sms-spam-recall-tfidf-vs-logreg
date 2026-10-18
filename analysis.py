"""Question: can a class-weighted word + character TF-IDF linear model with a
CV-tuned threshold beat a standard word-TF-IDF logistic regression on spam
recall? Data: UCI SMS Spam Collection (id 228) via the ucimlrepo package."""
import json
import socket
import sys
import time

import matplotlib
matplotlib.use("Agg")  # headless plotting on CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_curve,
                             recall_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_predict, train_test_split)
from sklearn.pipeline import make_pipeline, make_union

SEED = 42
MAX_ROWS = 200_000          # subsample cap (dataset is ~5.6k rows, so inactive)
PRECISION_FLOOR = 0.90      # threshold must keep out-of-fold spam precision >= this
SOURCE_URL = "https://archive.ics.uci.edu/dataset/228/sms+spam+collection"
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def load_data():
    """Download (with timeout + retry/backoff) and return texts and 0/1 labels (1 = spam)."""
    socket.setdefaulttimeout(60)  # 60 s timeout on all network calls (ucimlrepo has no timeout arg)
    data = None
    for attempt in range(4):
        try:
            data = fetch_ucirepo(id=228)
            break
        except Exception as e:  # covers HTTP 429/5xx and connection errors
            wait = 2 ** attempt
            log(f"download failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    if data is None:
        sys.exit("could not download the SMS Spam Collection")
    label_col = data.targets.columns[0]
    df = pd.concat([data.features, data.targets], axis=1)
    text_col = [c for c in df.columns if c != label_col and df[c].dtype == object][0]
    df = df[[text_col, label_col]].dropna()
    if len(df) > MAX_ROWS:  # same subsample rule would apply to both models
        df = df.sample(MAX_ROWS, random_state=SEED)
    y = (df[label_col].astype(str).str.strip().str.lower() == "spam").astype(int).to_numpy()
    X = df[text_col].astype(str).to_numpy()
    return X, y


def evaluate(y_true, y_pred):
    """Shared evaluation: spam recall (main), accuracy, macro-F1, spam F1."""
    return {
        "recall": float(recall_score(y_true, y_pred, pos_label=1)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "f1": float(f1_score(y_true, y_pred, pos_label=1)),
    }


def make_improved():
    """Word (1-2) + char_wb (2-5) TF-IDF, class-balanced logistic regression."""
    feats = make_union(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2),
    )
    return make_pipeline(feats, LogisticRegression(class_weight="balanced", max_iter=1000))


X, y = load_data()
log(f"loaded {len(X)} messages, spam share {y.mean():.3f}")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
log(f"train {len(Xtr)}, test {len(Xte)} (stratified 80/20, seed {SEED})")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# ---- BASELINE: standard word TF-IDF + default logistic regression, threshold 0.5 ----
baseline = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
baseline.fit(Xtr, ytr)
base_res = evaluate(yte, baseline.predict(Xte))
log(f"baseline test recall {base_res['recall']:.4f}")

# ---- IMPROVED: CV-tuned C (by average precision), CV-tuned threshold ----
search = GridSearchCV(make_improved(), {"logisticregression__C": [0.3, 1, 3, 10, 30]},
                      scoring="average_precision", cv=skf, n_jobs=-1)
search.fit(Xtr, ytr)
log(f"best C by 5-fold CV: {search.best_params_['logisticregression__C']}")

# Out-of-fold probabilities on the TRAINING set only -> pick threshold
oof = cross_val_predict(clone(search.best_estimator_), Xtr, ytr, cv=skf,
                        method="predict_proba", n_jobs=-1)[:, 1]
prec, rec, thr = precision_recall_curve(ytr, oof)
feasible = prec[:-1] >= PRECISION_FLOOR
if feasible.any():  # highest recall among thresholds meeting the precision floor
    i = np.flatnonzero(feasible)[np.argmax(rec[:-1][feasible])]
else:
    log("no OOF threshold reaches the precision floor; falling back to max OOF F1")
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    i = int(np.argmax(f1[:-1]))
threshold = float(thr[i])
log(f"chosen threshold {threshold:.4f} (OOF precision {prec[i]:.3f}, recall {rec[i]:.3f})")

improved = search.best_estimator_  # refit on full training set by GridSearchCV
p_te = improved.predict_proba(Xte)[:, 1]
imp_res = evaluate(yte, (p_te >= threshold).astype(int))
log(f"improved test recall {imp_res['recall']:.4f}")

if yte.sum() == 0 or not np.isfinite(imp_res["recall"]) or not np.isfinite(base_res["recall"]):
    sys.exit("metrics are not meaningful; aborting")

# ---- Optional chart ----
import os
os.makedirs("figures", exist_ok=True)
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["baseline", "improved"], [base_res["recall"], imp_res["recall"]], color=["grey", "tab:blue"])
ax.set_ylabel("spam recall (test)")
ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig("figures/recall_comparison.png", dpi=120)

# ---- Results ----
results = {
    "question": "Can a class-weighted word and character TF-IDF linear model with a cross-validated threshold beat a standard word-TF-IDF logistic regression on spam recall for real SMS messages?",
    "method": "UCI SMS Spam Collection via ucimlrepo; stratified 80/20 split (seed 42); all tuning by 5-fold CV on the training part; test set used once for both models.",
    "dataset": {
        "name": "UCI SMS Spam Collection (id 228)",
        "source_url": SOURCE_URL,
        "n_rows": int(len(X)),
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "same_as_paper": False,
        "baseline_is_papers_method": False,
    },
    "metric_name": "spam-class recall on the held-out test split",
    "higher_is_better": True,
    "baseline": "Word TF-IDF + default logistic regression (threshold 0.5), no tuning.",
    "improved": "Word (1-2 gram) + char_wb (2-5 gram) TF-IDF, class_weight='balanced' logistic regression, C chosen by 5-fold CV on average precision, decision threshold chosen on out-of-fold training predictions to maximise spam recall subject to precision >= 0.90.",
    "metrics": {
        "baseline_score": base_res["recall"],
        "improved_score": imp_res["recall"],
        "improvement": imp_res["recall"] - base_res["recall"],
        "baseline_accuracy": base_res["accuracy"],
        "improved_accuracy": imp_res["accuracy"],
        "baseline_macro_f1": base_res["macro_f1"],
        "improved_macro_f1": imp_res["macro_f1"],
        "baseline_f1": base_res["f1"],
        "improved_f1": imp_res["f1"],
        "chosen_C": float(search.best_params_["logisticregression__C"]),
        "chosen_threshold": threshold,
        "n_spam_test": int(yte.sum()),
    },
    "notes": "Single split, so the test estimate has wide uncertainty (about 75 spam messages in test). Both models are deterministic given seed 42 (lbfgs, fixed CV folds), so no multi-seed std is reported. The threshold is tuned on 80%-data out-of-fold models but applied to a model refit on all training data, which can shift calibration slightly. The UCI SMS data is a real-message stand-in, not the paper's synthetic LLM-generated data, and the baseline is a classical linear model, not the paper's fine-tuned small language model.",
}
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
log("wrote results.json")
