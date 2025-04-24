# 🔒 OSS Threat Detection Toolkit

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Accuracy](https://img.shields.io/badge/accuracy-97%25-yellowgreen)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15270679.svg)](https://zenodo.org/records/15274842)

This project uses machine learning to detect and classify **supply chain threats** in open source software (OSS). It includes a dataset, Python scripts, and automated evaluation.

---

## 📁 What's in this project?

- `data/oss_threat_dataset.csv` → labeled threat data
- `scripts/evaluate.py` → shows label stats
- `scripts/evaluate_with_predictions.py` → checks prediction accuracy
- `.github/workflows/evaluate.yml` → runs scripts automatically on every change
- `evaluation_report.md` → saved report from model
- `README.md` → this file

---

## 🚀 How to Use

### Step 1: Clone this project

```bash
git clone https://github.com/Mdniloykhan/oss-threat-data.git
cd oss-threat-data
