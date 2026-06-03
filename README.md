# OSS Threat Detection

Dataset and experiments for taxonomy-aligned LLM-based 
detection of open source software supply chain threats.

## Dataset
999 verified OSS supply chain incidents across 5 attack categories:
- AV-200: Typosquatting
- AV-300: Trojan Source  
- AV-400: Malicious Builds
- AV-410: Pipeline Poisoning
- AV-509: Dependency Confusion

## Results
| Method | Accuracy |
|---|---|
| TF-IDF + Logistic Regression | 82.3% |
| Mistral 7B (zero-shot) | 65.7% |
| GPT-4 Taxonomy-Aligned (Ours) | 97.0% |

## Reproduce
pip install -r requirements.txt
python scripts/baseline_experiments.py
