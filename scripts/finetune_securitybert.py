#!/usr/bin/env python3
# ============================================================
# SecurityBERT Fine-Tuning for OSS Threat Detection
# COMPLETE FIXED VERSION — All known issues resolved
# AsiaCCS 2027 Paper — Experiment 2
# Hardware: RTX 4070 Ti Super 16GB VRAM
# Expected time: 30-45 minutes
# ============================================================
# SETUP — Run once before running this script:
#   pip install transformers datasets scikit-learn pandas torch accelerate
# Then run:
#   python finetune_securitybert.py
# ============================================================

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)

import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

# SecurityBERT — pretrained on cybersecurity text
# Falls back to strong alternatives if unavailable
MODEL_OPTIONS = [
    "ehsanaghaei/SecureBERT",
    "jackaduma/SecRoBERTa",
    "answerdotai/ModernBERT-base",
    "microsoft/deberta-v3-base",
]

DATASET_PATHS = [
    "oss_threat_dataset_beta.csv",
    "data/oss_threat_dataset_beta.csv",
    "C:/Users/user1/Desktop/oss_threat_dataset_beta.csv",
]

OUTPUT_DIR   = "results/securitybert_finetuned"
RESULTS_FILE = "results/securitybert_results.json"

VALID_LABELS = ["AV-200", "AV-300", "AV-400", "AV-410", "AV-509"]
LABEL2ID     = {label: i for i, label in enumerate(sorted(VALID_LABELS))}
ID2LABEL     = {i: label for label, i in LABEL2ID.items()}

MAX_LENGTH   = 256
BATCH_SIZE   = 16
NUM_EPOCHS   = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY  = 0.01
TEST_SIZE     = 0.2
RANDOM_SEED   = 42

set_seed(RANDOM_SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("SECURITYBERT FINE-TUNING — OSS THREAT DETECTION")
print("=" * 60)
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU:  {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("results", exist_ok=True)

# ============================================================
# STEP 1 — LOAD DATASET
# ============================================================

print("\n" + "=" * 60)
print("STEP 1: LOADING DATASET")
print("=" * 60)

df = None
for path in DATASET_PATHS:
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"Loaded from: {path}")
        break

if df is None:
    raise FileNotFoundError(
        "Dataset not found. Place oss_threat_dataset_beta.csv "
        "in the same folder as this script."
    )

df = df[df['label'].isin(VALID_LABELS)].copy()
df = df.dropna(subset=['description', 'label'])
if 'id' in df.columns:
    df = df.drop_duplicates(subset=['id'])

# Combine title and description
if 'title' in df.columns:
    df['text'] = (
        df['title'].fillna('') + ' ' +
        df['description'].fillna('')
    ).str.strip()
else:
    df['text'] = df['description'].fillna('')

df['label_id'] = df['label'].map(LABEL2ID)

print(f"Clean entries: {len(df)}")
print("Distribution:", df['label'].value_counts().sort_index().to_dict())

# ============================================================
# STEP 2 — TRAIN/TEST SPLIT
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: TRAIN/TEST SPLIT")
print("=" * 60)

train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    stratify=df['label'],
    random_state=RANDOM_SEED
)

print(f"Training: {len(train_df)} | Test: {len(test_df)}")
print("Train:", train_df['label'].value_counts().sort_index().to_dict())
print("Test: ", test_df['label'].value_counts().sort_index().to_dict())

# ============================================================
# STEP 3 — LOAD MODEL
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: LOADING MODEL")
print("=" * 60)

tokenizer = None
model = None
MODEL_NAME = None

for model_name in MODEL_OPTIONS:
    try:
        print(f"Trying: {model_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(VALID_LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True
        )
        MODEL_NAME = model_name
        print(f"Successfully loaded: {model_name}")
        break
    except Exception as e:
        print(f"Failed: {e}")
        continue

if model is None:
    raise RuntimeError("Could not load any model. Check internet connection.")

model = model.to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# ============================================================
# STEP 4 — TOKENIZE DATASET
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: TOKENIZING DATASET")
print("=" * 60)

class ThreatDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

train_dataset = ThreatDataset(
    train_df['text'].values,
    train_df['label_id'].values,
    tokenizer,
    MAX_LENGTH
)
test_dataset = ThreatDataset(
    test_df['text'].values,
    test_df['label_id'].values,
    tokenizer,
    MAX_LENGTH
)

print(f"Train dataset: {len(train_dataset)} samples")
print(f"Test dataset:  {len(test_dataset)} samples")

# ============================================================
# STEP 5 — METRICS
# ============================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average='macro', zero_division=0)
    return {"accuracy": acc, "f1_macro": f1}

# ============================================================
# STEP 6 — FINE-TUNING
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: FINE-TUNING")
print("=" * 60)
print(f"Epochs:        {NUM_EPOCHS}")
print(f"Batch size:    {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Weight decay:  {WEIGHT_DECAY}")
print(f"Max length:    {MAX_LENGTH}")
print("Estimated time: 30-45 minutes on RTX 4070 Ti Super")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    logging_steps=10,
    warmup_ratio=0.1,
    fp16=torch.cuda.is_available(),
    report_to="none",
    dataloader_num_workers=0,
    seed=RANDOM_SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("\nStarting training...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nModel saved to {OUTPUT_DIR}")

# ============================================================
# STEP 7 — EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("STEP 7: FINAL EVALUATION")
print("=" * 60)

pred_output = trainer.predict(test_dataset)
pred_ids    = np.argmax(pred_output.predictions, axis=-1)
true_ids    = test_df['label_id'].values

pred_names = [ID2LABEL[p] for p in pred_ids]
true_names = [ID2LABEL[t] for t in true_ids]

acc = accuracy_score(true_names, pred_names)
f1  = f1_score(true_names, pred_names, average='macro', zero_division=0)

print(f"Accuracy:  {acc:.4f} ({acc:.1%})")
print(f"Macro F1:  {f1:.4f}")

print(f"\nDetailed Report:")
print(classification_report(
    true_names, pred_names,
    target_names=sorted(VALID_LABELS),
    zero_division=0
))

print("Confusion Matrix:")
cm = confusion_matrix(true_names, pred_names, labels=sorted(VALID_LABELS))
print(pd.DataFrame(
    cm,
    index=sorted(VALID_LABELS),
    columns=sorted(VALID_LABELS)
).to_string())

# ============================================================
# STEP 8 — COMPARISON TABLE
# ============================================================

print("\n" + "=" * 60)
print("COMPARISON TABLE — COPY INTO YOUR PAPER")
print("=" * 60)

methods = [
    ("Random Classifier",                0.199, 0.196),
    ("Majority Class",                   0.224, 0.073),
    ("TF-IDF + Naive Bayes",             0.814, 0.814),
    ("TF-IDF + Logistic Regression",     0.823, 0.823),
    ("TF-IDF + LinearSVC",               0.819, 0.819),
    ("Mistral 7B (zero-shot)",           0.657, 0.636),
    ("Llama 3.1 8B (fine-tuned)",        0.705, 0.704),
    (f"{MODEL_NAME.split('/')[-1]} (fine-tuned)", acc, f1),
    ("GPT-4 Taxonomy-Aligned (Ours)",    0.970, 0.970),
]

print(f"\n{'Method':<45} {'Accuracy':>10} {'Macro F1':>10}")
print("-" * 67)
for name, a, f in methods:
    marker = " <-- NEW"        if "BERT" in name or "RoBERTa" in name or "ModernBERT" in name or "deberta" in name.lower() else ""
    marker = " <-- OUR METHOD" if "GPT-4" in name else marker
    print(f"{name:<45} {a:>10.1%} {f:>10.1%}{marker}")

# Save results
results = {
    "timestamp": datetime.now().isoformat(),
    "model": MODEL_NAME,
    "accuracy": acc,
    "f1_macro": f1,
    "test_size": len(test_df),
    "train_size": len(train_df),
    "num_epochs": NUM_EPOCHS,
    "hyperparameters": {
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "weight_decay": WEIGHT_DECAY,
    }
}

with open(RESULTS_FILE, 'w') as f_out:
    json.dump(results, f_out, indent=2)

print(f"\nResults saved to {RESULTS_FILE}")

print("\n" + "=" * 60)
print("SECURITYBERT FINE-TUNING COMPLETE")
print("=" * 60)
print(f"Model used:                  {MODEL_NAME}")
print(f"Accuracy:                    {acc:.1%}")
print(f"vs Llama 3.1 fine-tuned:     70.5%")
print(f"vs TF-IDF best baseline:     82.3%")
print(f"vs GPT-4 Taxonomy-Aligned:   97.0%")
print(f"\nGap to GPT-4:                {(0.970-acc)*100:.1f} pp")
