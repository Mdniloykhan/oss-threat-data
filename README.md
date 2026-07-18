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
│   └── oss_threat_dataset_beta.csv        # 999 verified OSS threat incidents
├── scripts/
│   ├── baseline_experiments.py            # TF-IDF and ML baseline classifiers
│   ├── opensource_llm_experiments.py      # Open source LLM experiments (Ollama)
│   ├── finetune_llama3.py                 # Llama 3.1 8B QLoRA fine-tuning
│   └── finetune_securitybert_v2.py        # SecRoBERTa fine-tuning
├── results/
│   ├── confusion_matrix_baseline.png      # Confusion matrix figure
│   ├── method_comparison.png              # Method comparison figure
│   ├── llama3_results.json                # Llama 3.1 8B experiment results
│   └── securitybert_results.json          # SecRoBERTa experiment results
├── data/
│   ├── irr_sample.csv                     # IRR validation sample
│   ├── irr_session1.csv                   # IRR session 1 labels
│   └── irr_session2.csv                   # IRR session 2 labels
├── .github/workflows/
│   ├── evaluate.yml                       # Auto-runs on data/script changes
│   └── run.yml                            # Runs baselines on push to main
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

Sources include GitHub Security Advisories, CISA alerts, NHS England Digital, Unit42, Sonatype, Socket.dev, Checkmarx, Snyk, Datadog Security Labs, and Phylum covering incidents from 2018 to 2026.

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
| Llama 3.1 8B (fine-tuned) | 70.5% | 70.4% |
| SecRoBERTa (fine-tuned) | 77.5% | 77.8% |
| **GPT-4 Taxonomy-Aligned (Ours)** | **97.0%** | **97.0%** |

> **Key Finding:** Fine-tuned neural models (SecRoBERTa 77.5%, Llama 3.1 8B 70.5%) underperform simple TF-IDF classifiers (82.3%), demonstrating that taxonomy-aligned prompting rather than model scale or fine-tuning is the critical performance factor.

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

# Run Llama 3.1 8B fine-tuning (requires HuggingFace access)
python scripts/finetune_llama3_v2.py

# Run SecRoBERTa fine-tuning
python scripts/finetune_securitybert_v2.py
```

---

## 📄 Citation

If you use this dataset or code please cite:

```bibtex
@inproceedings{niloy2027oss,
  title={Autonomous OSS Threat Detection via Taxonomy-Aligned LLMs},
  author={Anonymous Author(s)},
  booktitle={Proceedings of the ACM Asia Conference on Computer 
             and Communications Security (AsiaCCS)},
  year={2027}
}
```

---

## 📋 Requirements

```
transformers>=4.40.0
peft>=0.9.0
trl>=1.8.0
datasets>=2.18.0
accelerate>=0.27.0
bitsandbytes>=0.43.0
scikit-learn>=1.3.0
pandas>=2.0.0
torch>=2.1.0
matplotlib>=3.7.0
seaborn>=0.12.0
requests>=2.31.0
tqdm>=4.66.0
scipy>=1.11.0
```

---

## 🔒 License

MIT License — see LICENSE file for details.
