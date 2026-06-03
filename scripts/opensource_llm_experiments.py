# ============================================================
# OSS Threat Detection — Open Source LLM Experiments
# For AsiaCCS 2027 Paper
# ============================================================
# INSTRUCTIONS:
# 1. Run: pip install pandas requests tqdm
# 2. Install Ollama: https://ollama.ai
# 3. Pull models:
#    ollama pull llama3
#    ollama pull mistral
# 4. Run: python scripts/opensource_llm_experiments.py
# ============================================================
# NOTE: This uses Ollama to run models locally — FREE
# No API key needed. No cost.
# Llama 3 requires ~4GB RAM, Mistral requires ~4GB RAM
# ============================================================

import pandas as pd
import numpy as np
import requests
import json
import time
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("OPEN SOURCE LLM EXPERIMENTS")
print("=" * 60)

# ============================================================
# STEP 1 — LOAD DATASET
# ============================================================

valid_labels = ['AV-200', 'AV-300', 'AV-400', 'AV-410', 'AV-509']
DATASET_PATH = 'data/oss_threat_dataset_beta.csv'

print(f"\nLoading: {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH)
df = df[df['label'].isin(valid_labels)].copy()
df = df.dropna(subset=['description', 'label'])
if 'id' in df.columns:
    df = df.drop_duplicates(subset=['id'])

print(f"Loaded {len(df)} clean entries")

# ============================================================
# STEP 2 — SAMPLE FOR EXPERIMENT
# ============================================================
# Running 999 entries through a local LLM takes hours.
# We use a stratified sample of 200 entries (40 per class).
# This is standard practice in LLM evaluation papers.

SAMPLE_SIZE = 200  # Increase if you have time/resources

sampled = df.groupby('label', group_keys=False).apply(
    lambda x: x.sample(min(len(x), SAMPLE_SIZE // len(valid_labels)),
                        random_state=42)
)
sampled = sampled.reset_index(drop=True)

print(f"\nUsing stratified sample: {len(sampled)} entries")
print("Distribution:", sampled['label'].value_counts().sort_index().to_dict())

# ============================================================
# STEP 3 — TAXONOMY PROMPT
# ============================================================

TAXONOMY_DESCRIPTION = """
You are an expert in open source software supply chain security.
Classify the following security incident into exactly one of these categories:

AV-200: Typosquatting — malicious package mimics a legitimate package name
AV-300: Trojan Source — Unicode/invisible characters hide malicious code
AV-400: Malicious Build — malicious code in package install/build process
AV-410: Pipeline Poisoning — CI/CD pipeline is compromised or manipulated
AV-509: Dependency Confusion — public package overrides private internal package

Respond with ONLY the category label (e.g. AV-200). Nothing else.
"""

def build_prompt(title, description):
    text = f"{title}. {description}" if title else description
    return f"{TAXONOMY_DESCRIPTION}\n\nIncident:\n{text}\n\nCategory:"

# ============================================================
# STEP 4 — OLLAMA API CALL
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"

def classify_with_ollama(model_name, title, description, retries=3):
    prompt = build_prompt(title, description)
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 10
        }
    }
    for attempt in range(retries):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=60)
            if response.status_code == 200:
                result = response.json().get('response', '').strip()
                # Extract AV label from response
                for label in valid_labels:
                    if label in result:
                        return label
                return 'UNKNOWN'
        except Exception as e:
            if attempt == retries - 1:
                return 'ERROR'
            time.sleep(2)
    return 'ERROR'

def check_ollama_running():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        return r.status_code == 200
    except:
        return False

def get_available_models():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m['name'] for m in r.json().get('models', [])]
            return models
    except:
        pass
    return []

# ============================================================
# STEP 5 — CHECK OLLAMA AND AVAILABLE MODELS
# ============================================================

print("\n" + "=" * 60)
print("CHECKING OLLAMA STATUS")
print("=" * 60)

if not check_ollama_running():
    print("\nERROR: Ollama is not running!")
    print("Please install and start Ollama:")
    print("  1. Install: https://ollama.ai")
    print("  2. Start:   ollama serve")
    print("  3. Pull:    ollama pull llama3")
    print("  4. Pull:    ollama pull mistral")
    exit(1)

available = get_available_models()
print(f"Ollama is running. Available models: {available}")

# ============================================================
# STEP 6 — RUN EXPERIMENTS ON EACH MODEL
# ============================================================

# Models to test — add or remove based on what you pulled
MODELS_TO_TEST = []
for candidate in ['llama3', 'llama3:8b', 'mistral', 'mistral:7b', 'phi3', 'gemma:7b']:
    if any(candidate in m for m in available):
        MODELS_TO_TEST.append(candidate)

if not MODELS_TO_TEST:
    print("\nNo supported models found. Please pull at least one:")
    print("  ollama pull llama3")
    print("  ollama pull mistral")
    exit(1)

print(f"\nModels to test: {MODELS_TO_TEST}")

all_results = {}

for model in MODELS_TO_TEST:
    print(f"\n{'=' * 60}")
    print(f"RUNNING: {model}")
    print(f"{'=' * 60}")

    predictions = []
    ground_truth = []
    errors = 0

    for _, row in tqdm(sampled.iterrows(), total=len(sampled), desc=model):
        title = str(row.get('title', '')) if 'title' in row else ''
        description = str(row.get('description', ''))
        true_label = str(row['label'])

        pred = classify_with_ollama(model, title, description)

        if pred in valid_labels:
            predictions.append(pred)
            ground_truth.append(true_label)
        else:
            errors += 1

    if not predictions:
        print(f"No valid predictions for {model}")
        continue

    acc = accuracy_score(ground_truth, predictions)
    f1  = f1_score(ground_truth, predictions, average='macro', zero_division=0)

    all_results[model] = {
        'accuracy': acc,
        'f1_macro': f1,
        'n_samples': len(predictions),
        'errors': errors,
        'predictions': predictions,
        'ground_truth': ground_truth
    }

    print(f"\nResults for {model}:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Macro F1 : {f1:.4f}")
    print(f"  Errors   : {errors}")
    print(f"\nDetailed Report:")
    print(classification_report(ground_truth, predictions,
                                 target_names=sorted(valid_labels),
                                 zero_division=0))

# ============================================================
# STEP 7 — FINAL COMPARISON TABLE
# ============================================================

print("\n" + "=" * 60)
print("FINAL COMPARISON TABLE")
print("=" * 60)

# Add your GPT-4 result for comparison
all_results['GPT-4 (Taxonomy-Aligned LLM)'] = {
    'accuracy': 0.97,
    'f1_macro': 0.97,
    'n_samples': 999,
    'errors': 0
}

# Add best baseline for reference
all_results['TF-IDF + Logistic Regression (baseline)'] = {
    'accuracy': 0.8228,
    'f1_macro': 0.8228,
    'n_samples': 999,
    'errors': 0
}

print(f"\n{'Method':<45} {'Accuracy':>10} {'Macro F1':>10} {'Samples':>10}")
print("-" * 76)
for name, res in all_results.items():
    print(f"{name:<45} {res['accuracy']:>10.4f} {res['f1_macro']:>10.4f} {res['n_samples']:>10}")

# ============================================================
# STEP 8 — SAVE RESULTS TO CSV
# ============================================================

results_rows = []
for name, res in all_results.items():
    results_rows.append({
        'method': name,
        'accuracy': res['accuracy'],
        'f1_macro': res['f1_macro'],
        'n_samples': res['n_samples']
    })

results_df = pd.DataFrame(results_rows)
results_df.to_csv('opensource_llm_results.csv', index=False)
print("\nSaved: opensource_llm_results.csv")

# ============================================================
# STEP 9 — KEY NUMBERS FOR PAPER
# ============================================================

print("\n" + "=" * 60)
print("KEY NUMBERS FOR YOUR PAPER")
print("=" * 60)

for name, res in all_results.items():
    if name not in ['GPT-4 (Taxonomy-Aligned LLM)', 'TF-IDF + Logistic Regression (baseline)']:
        diff = (res['accuracy'] - 0.8228) * 100
        sign = '+' if diff > 0 else ''
        print(f"\n{name}:")
        print(f"  Accuracy: {res['accuracy']:.1%}")
        print(f"  vs baseline: {sign}{diff:.1f} pp")

print("\n" + "=" * 60)
print("EXPERIMENT COMPLETE")
print("=" * 60)
print("Next: Add these results to your paper Table 2")
print("      Compare GPT-4 vs open source models vs baseline")
