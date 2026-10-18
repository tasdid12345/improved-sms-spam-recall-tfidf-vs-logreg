# SMS spam: class-weighted word + char TF-IDF vs. word TF-IDF baseline

Question: can a class-weighted word + character TF-IDF linear model with a CV-tuned threshold beat a standard word-TF-IDF logistic regression on spam recall?

Data: UCI SMS Spam Collection (id 228). `analysis.py` first tries the `ucimlrepo` package. If that repository entry is not importable (as happened in CI), it downloads the same file directly from the UCI archive.

Method: stratified 80/20 split with seed 42. The baseline is word TF-IDF with default logistic regression at threshold 0.5. The improved model uses word (1-2) plus char_wb (2-5) TF-IDF, a balanced logistic regression with C chosen by 5-fold CV, and a threshold chosen on out-of-fold training predictions subject to a precision floor of 0.90. The test set is used once for each model.

Run: `python analysis.py` writes `results.json` and `figures/recall_comparison.png`.

## Results (latest run)

On the UCI SMS Spam Collection held-out test split (1,115 messages), spam-class recall rose from 81.2% for a standard word-TF-IDF logistic regression to 97.3% for my class-weighted word and character TF-IDF model with a CV-tuned threshold, though this is a single split.

| Metric | Value |
|---|---|
| baseline_score | 0.8121 |
| improved_score | 0.9732 |
| improvement | 0.1611 |
| baseline_accuracy | 0.9749 |
| improved_accuracy | 0.9857 |
| baseline_macro_f1 | 0.941 |
| improved_macro_f1 | 0.9697 |
| baseline_f1 | 0.8963 |
| improved_f1 | 0.9477 |
| chosen_C | 10 |
| chosen_threshold | 0.1434 |
| n_spam_test | 149 |

Run on 2026-10-18. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Single split, so the test estimate has wide uncertainty (about 149 spam messages in test). Both models are deterministic given seed 42 (lbfgs, fixed CV folds), so no multi-seed std is reported. The threshold is tuned on 80%-data out-of-fold models but applied to a model refit on all training data, which can shift calibration slightly. The UCI SMS data is a real-message stand-in, not the paper's synthetic LLM-generated data, and the baseline is a classical linear model, not the paper's fine-tuned small language model. The data was loaded from the UCI archive file directly when the ucimlrepo download for id 228 is unavailable; the content is the same file.

## Reference

This project is motivated by the following paper:

> A ElZemity, J Sylvester, B Arief, R De Lemos (2026). *Agentic Knowledge Distillation: Autonomous Training of Small Language Models for SMS Threat Detection*. arXiv preprint arXiv. https://arxiv.org/abs/2602.10869

Paper author: Prof. Budi Arief, University of Kent (b.arief@kent.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
