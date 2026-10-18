# Class-weighted word + char TF-IDF vs word TF-IDF for SMS spam recall

Motivated by "Agentic Knowledge Distillation: Autonomous Training of Small Language Models for SMS Threat Detection" (2026), which uses LLM-generated synthetic SMS data to train small language models for spam detection and reports accuracy and recall. Missing spam is the costly error in that setting, so recall is the main metric here. This experiment asks a cheaper question that the paper does not test: whether a classical linear model with tuned class weighting and a CV-chosen threshold can beat a plain baseline on spam recall.

**Question:** Can a class-weighted word and character TF-IDF linear model with a cross-validated threshold beat a standard word-TF-IDF logistic regression on spam recall for real SMS messages?

**Dataset:** UCI SMS Spam Collection (dataset id 228, about 5.6k labelled messages), downloaded via the `ucimlrepo` package. This is a real-message dataset and is not the paper's own synthetic data (`same_as_paper` is false).

**Baseline:** word TF-IDF with default logistic regression, threshold 0.5. This stands in for the paper's fine-tuned small language model, which is too heavy for CPU.

**Improved model:** word and character n-gram TF-IDF, class-balanced logistic regression, C tuned by 5-fold CV, and a decision threshold chosen on out-of-fold training predictions to maximise spam recall at precision of at least 0.90.

**How to run:**

```
pip install -r requirements.txt
python analysis.py
```

**Results:** results are written to `results.json` (and `figures/recall_comparison.png`).

## Reference

This project is motivated by the following paper:

> A ElZemity, J Sylvester, B Arief, R De Lemos (2026). *Agentic Knowledge Distillation: Autonomous Training of Small Language Models for SMS Threat Detection*. arXiv preprint arXiv. https://arxiv.org/abs/2602.10869

Paper author: Prof. Budi Arief, University of Kent (b.arief@kent.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
