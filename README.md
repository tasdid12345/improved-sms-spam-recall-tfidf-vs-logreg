# SMS spam: class-weighted word + char TF-IDF vs. word TF-IDF baseline

Question: can a class-weighted word + character TF-IDF linear model with a CV-tuned threshold beat a standard word-TF-IDF logistic regression on spam recall?

Data: UCI SMS Spam Collection (id 228). `analysis.py` first tries the `ucimlrepo` package. If that repository entry is not importable (as happened in CI), it downloads the same file directly from the UCI archive.

Method: stratified 80/20 split with seed 42. The baseline is word TF-IDF with default logistic regression at threshold 0.5. The improved model uses word (1-2) plus char_wb (2-5) TF-IDF, a balanced logistic regression with C chosen by 5-fold CV, and a threshold chosen on out-of-fold training predictions subject to a precision floor of 0.90. The test set is used once for each model.

Run: `python analysis.py` writes `results.json` and `figures/recall_comparison.png`.

## Reference

This project is motivated by the following paper:

> A ElZemity, J Sylvester, B Arief, R De Lemos (2026). *Agentic Knowledge Distillation: Autonomous Training of Small Language Models for SMS Threat Detection*. arXiv preprint arXiv. https://arxiv.org/abs/2602.10869

Paper author: Prof. Budi Arief, University of Kent (b.arief@kent.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
