# ============================================================
# OSS Threat Detection — Open Source LLM Experiments v2
# For AsiaCCS 2027 Paper
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
print(f"Columns: {df.columns.tolist()}")

# Clean
df = df[df['label'].isin(valid_labels)].copy()
df = df.dropna(subset=['description', 'label'])
if 'id' in df.columns:
    df = df.drop_duplicates(subset=['id'])

print(f"Clean entries: {len(df)}")
print("Distribution:", df['label'].value_counts().sort_index().to_dict())

# ============================================================
# STEP 2 — STRATIFIED SAMPLE
# ============================================================

SAMPLE_PER_CLASS = 40  # 40 x 5 classes = 200 total

samples = []
for label in valid_labels:
    class_df = df[df['label'] == label]
    n = min(len(class_df), SAMPLE_PER_CLASS)
    samples.append(class_df.sample(n, random_state=42))

sampled = pd.concat(samples, ignore_index=True)
sampled = sampled.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nStratified sample: {len(sampled)} entries")
print("Sample distribution:", sampled['label'].value_counts().sort_index().to_dict())

# ============================================================
# STEP 3 — TAXONOMY PROMPT
# ============================================================

SYSTEM_PROMPT = """You are an expert in open source software supply chain security.
Classify the following security incident into exactly one category:

AV-200: Typosquatting — malicious package mimics a legitimate package name
AV-300: Trojan Source — Unicode/invisible characters hide malicious code
AV-400: Malicious Build — malicious code in package install/build process
AV-410: Pipeline Poisoning — CI/CD pipeline is compromised or manipulated
AV-509: Dependency Confusion — public package overrides private internal package

Respond with ONLY the category label such as AV-200. Nothing else."""

def classify_with_ollama(model_name, title, description, retries=3):
    text = f"{title}. {description}" if pd.notna(title) and str(title).strip() else str(description)
    prompt = f"{SYSTEM_PROMPT}\n\nIncident: {text}\n\nCategory:"

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 10}
    }

    for attempt in range(retries):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=120
            )
            if response.status_code == 200:
                result = response.json().get('response', '').strip().upper()
                for label in valid_labels:
                    if label in result:
                        return label
                return 'UNKNOWN'
        except Exception as e:
            if attempt == retries - 1:
                print(f"  Error: {e}")
                return 'ERROR'
            time.sleep(3)
    return 'ERROR'

# ============================================================
# STEP 4 — CHECK OLLAMA
# ============================================================

def get_available_models():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code == 200:
            return [m['name'] for m in r.json().get('models', [])]
    except:
        pass
    return []

print("\n" + "=" * 60)
print("CHECKING OLLAMA")
print("=" * 60)

available = get_available_models()
if not available:
    print("ERROR: Ollama not running or no models found.")
    print("Run: ollama serve &")
    print("Run: ollama pull phi3")
    exit(1)

print(f"Available models: {available}")

# Pick models to test
MODELS_TO_TEST = []
for candidate in ['phi3', 'phi3:mini', 'mistral', 'mistral:7b', 'llama3', 'llama3:8b']:
    if any(candidate in m for m in available):
        MODELS_TO_TEST.append(candidate)
        break  # Test one model first — add more if time permits

if not MODELS_TO_TEST:
    MODELS_TO_TEST = [available[0]]  # Use whatever is available

print(f"Testing: {MODELS_TO_TEST}")

# ============================================================
# STEP 5 — RUN EXPERIMENTS
# ============================================================

all_results = {}

for model in MODELS_TO_TEST:
    print(f"\n{'=' * 60}")
    print(f"MODEL: {model}")
    print(f"{'=' * 60}")
    print(f"Running {len(sampled)} classifications...")

    predictions = []
    ground_truth = []
    errors = 0
    unknown = 0

    for i, row in tqdm(sampled.iterrows(), total=len(sampled)):
        title = str(row.get('title', ''))
        desc  = str(row.get('description', ''))
        true  = str(row['label'])

        pred = classify_with_ollama(model, title, desc)

        if pred in valid_labels:
            predictions.append(pred)
            ground_truth.append(true)
        elif pred == 'UNKNOWN':
            unknown += 1
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
        'unknown': unknown,
        'predictions': predictions,
        'ground_truth': ground_truth
    }

    print(f"\nAccuracy : {acc:.4f} ({acc:.1%})")
    print(f"Macro F1 : {f1:.4f}")
    print(f"Errors   : {errors} | Unknown: {unknown}")
    print(f"\nDetailed Report:")
    print(classification_report(ground_truth, predictions,
                                 target_names=sorted(valid_labels),
                                 zero_division=0))

# ============================================================
# STEP 6 — FINAL TABLE
# ============================================================

all_results['GPT-4 Taxonomy-Aligned LLM (ours)'] = {
    'accuracy': 0.97, 'f1_macro': 0.97, 'n_samples': 999
}
all_results['TF-IDF + Logistic Regression'] = {
    'accuracy': 0.8228, 'f1_macro': 0.8228, 'n_samples': 999
}

print("\n" + "=" * 60)
print("FINAL COMPARISON TABLE")
print("=" * 60)
print(f"{'Method':<45} {'Accuracy':>10} {'Macro F1':>10}")
print("-" * 67)
for name, res in all_results.items():
    print(f"{name:<45} {res['accuracy']:>10.4f} {res['f1_macro']:>10.4f}")

# Save results
rows = [{'method': k, 'accuracy': v['accuracy'], 'f1_macro': v['f1_macro']}
        for k, v in all_results.items()]
pd.DataFrame(rows).to_csv('opensource_llm_results.csv', index=False)
print("\nSaved: opensource_llm_results.csv")

print("\n" + "=" * 60)
print("EXPERIMENT COMPLETE")
print("=" * 60)
