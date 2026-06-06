# 🔒 OSS Threat Detection Toolkit

![Python](https://img.shields.io/badge/python-3.8+-blue)
![Accuracy](https://img.shields.io/badge/accuracy-97%25-brightgreen)
![Dataset](https://img.shields.io/badge/dataset-999%20incidents-orange)
![License](https://img.shields.io/badge/license-MIT-green)

A taxonomy-aligned LLM framework for automated detection and classification of open source software (OSS) supply chain threats. This repository accompanies a research paper submitted to AsiaCCS 2027.

---

## 🎯 What This Does

Classifies OSS supply chain security incidents into five attack categories using a taxonomy-aligned GPT-4 prompting strategy, achieving **97% accuracy** on 999 verified real-world incidents.

---

## 📂 Repository Structure

```
oss-threat-data/
├── data/
│   └── oss_threat_dataset_beta.csv   # 999 verified OSS threat incidents
├── scripts/
│   ├── baseline_experiments.py       # TF-IDF and ML baseline classifiers
│   └── opensource_llm_experiments.py # Open source LLM experiments (Ollama)
├── results/
│   ├── confusion_matrix_baseline.png # Confusion matrix figure
│   └── method_comparison.png         # Method comparison figure
├── .github/workflows/
│   ├── evaluate.yml                  # Auto-runs on data/script changes
│   └── run.yml                       # Runs baselines on push to main
└── requirements.txt
```

---

## 📊 Dataset

999 verified OSS supply chain incidents across 5 attack categories:

| Category | Description | Count |
|---|---|---|
| AV-200 | Typosquatting — malicious package mimics legitimate name | 200 |
| AV-300 | Trojan Source — Unicode characters hide malicious code | 198 |
| AV-400 | Malicious Builds — malicious code in install/build process | 224 |
| AV-410 | Pipeline Poisoning — CI/CD pipeline compromised | 192 |
| AV-509 | Dependency Confusion — public package overrides private | 185 |

Sources include GitHub Security Advisories, CISA alerts, NHS England Digital, Unit42, Sonatype, Socket.dev, and Snyk covering incidents from 2018 to 2026.

---

## 📈 Results

| Method | Accuracy | Macro F1 |
|---|---|---|
| Random Classifier | 19.9% | 19.6% |
| Majority Class | 22.4% | 7.3% |
| TF-IDF + Naïve Bayes | 81.4% | 81.4% |
| TF-IDF + Logistic Regression | 82.3% | 82.3% |
| TF-IDF + LinearSVC | 81.9% | 81.9% |
| Mistral 7B (zero-shot) | 65.7% | 63.6% |
| **GPT-4 Taxonomy-Aligned LLM (Ours)** | **97.0%** | **97.0%** |

Our taxonomy-aligned LLM approach outperforms the strongest traditional baseline by **+14.7 percentage points**.

---

## 🚀 Reproduce

```bash
# Install dependencies
pip install -r requirements.txt

# Run baseline experiments
python scripts/baseline_experiments.py

# Run open source LLM experiments (requires Ollama)
ollama pull mistral
python scripts/opensource_llm_experiments.py
```

---

## 📄 Citation

If you use this dataset or code please cite:

```bibtex
@inproceedings{niloy2027oss,
  title={Taxonomy-Aligned LLM Framework for OSS Supply Chain Threat Detection},
  author={Md. Robiul Islam Niloy},
  booktitle={Proceedings of the ACM Asia Conference on Computer and Communications Security (AsiaCCS)},
  year={2027}
}
```

---
