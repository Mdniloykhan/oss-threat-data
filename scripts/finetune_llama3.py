#!/usr/bin/env python3
# ============================================================
# Llama 3.1 8B Fine-Tuning for OSS Threat Detection
# FIXED VERSION — Corrected hyperparameters to prevent overfitting
# AsiaCCS 2027 Paper
# Hardware: RTX 4070 Ti Super 16GB VRAM
# Expected time: 3-4 hours
# ============================================================

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix
)

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer, SFTConfig

import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION — FIXED HYPERPARAMETERS
# ============================================================

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_PATH = "oss_threat_dataset_beta.csv"
OUTPUT_DIR = "results/llama3_finetuned_v2"
RESULTS_FILE = "results/llama3_results_v2.json"

VALID_LABELS = ["AV-200", "AV-300", "AV-400", "AV-410", "AV-509"]

# FIXED — Key changes from v1
MAX_SEQ_LENGTH  = 512
BATCH_SIZE      = 2        # reduced from 4
GRAD_ACCUM      = 8        # effective batch = 16
LEARNING_RATE   = 2e-5     # reduced 10x from 2e-4 — prevents overfitting
NUM_EPOCHS      = 5        # increased from 3
LORA_R          = 8        # reduced from 16
LORA_ALPHA      = 16       # reduced from 32
LORA_DROPOUT    = 0.1      # increased from 0.05
WEIGHT_DECAY    = 0.01     # added — prevents overfitting
TEST_SIZE       = 0.2
RANDOM_SEED     = 42

print("=" * 60)
print("LLAMA 3.1 8B FINE-TUNING v2 — FIXED HYPERPARAMETERS")
print("=" * 60)
print(f"Model:         {MODEL_NAME}")
print(f"Learning rate: {LEARNING_RATE} (10x smaller than v1)")
print(f"LoRA rank:     {LORA_R} (half of v1)")
print(f"Dropout:       {LORA_DROPOUT} (2x higher than v1)")
print(f"Epochs:        {NUM_EPOCHS}")
print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
if torch.cuda.is_available():
    print(f"GPU:  {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ============================================================
# STEP 1 — LOAD DATASET
# ============================================================

print("\n" + "=" * 60)
print("STEP 1: LOADING DATASET")
print("=" * 60)

paths = [
    DATASET_PATH,
    f"data/{DATASET_PATH}",
    f"C:/Users/user1/Desktop/{DATASET_PATH}",
]

df = None
for path in paths:
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"Loaded from: {path}")
        break

if df is None:
    raise FileNotFoundError(
        f"Dataset not found. Please place {DATASET_PATH} "
        f"in the same folder as this script."
    )

df = df[df['label'].isin(VALID_LABELS)].copy()
df = df.dropna(subset=['description', 'label'])
if 'id' in df.columns:
    df = df.drop_duplicates(subset=['id'])

print(f"Clean entries: {len(df)}")
print("Distribution:", df['label'].value_counts().sort_index().to_dict())

# ============================================================
# STEP 2 — PROMPT FORMAT — FIXED
# ============================================================

SYSTEM_PROMPT = """You are an expert in open source software supply chain security.
Classify the following security incident into exactly one of these categories:

AV-200: Typosquatting — malicious package mimics a legitimate package name
AV-300: Trojan Source — Unicode/invisible characters hide malicious code
AV-400: Malicious Builds — malicious code runs during install/build process
AV-410: Pipeline Poisoning — CI/CD pipeline is compromised or manipulated
AV-509: Dependency Confusion — public package overrides private internal package

Respond with ONLY the category label (e.g. AV-200). Nothing else."""

def create_prompt(title, description, label=None):
    if pd.notna(title) and str(title).strip():
        text = f"{title}. {description}"
    else:
        text = str(description)

    if label:
        # FIXED — clearer separation between instruction and answer
        return (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n"
            f"{SYSTEM_PROMPT}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n"
            f"Classify this OSS security incident:\n\n{text}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
            f"The category is: {label}<|eot_id|>"
        )
    else:
        return (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n"
            f"{SYSTEM_PROMPT}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n"
            f"Classify this OSS security incident:\n\n{text}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
            f"The category is:"
        )

# ============================================================
# STEP 3 — TRAIN/TEST SPLIT
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: TRAIN/TEST SPLIT")
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

train_df['prompt'] = train_df.apply(
    lambda row: create_prompt(
        row.get('title', ''),
        row['description'],
        row['label']
    ), axis=1
)

train_df['completion'] = ''
train_dataset = Dataset.from_pandas(
    train_df[['prompt', 'completion']].reset_index(drop=True)
)

# ============================================================
# STEP 4 — LOAD MODEL WITH 4-BIT QUANTIZATION
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: LOADING MODEL")
print("=" * 60)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
    trust_remote_code=True
)

model = prepare_model_for_kbit_training(model)
print(f"Model loaded. GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ============================================================
# STEP 5 — CONFIGURE LORA — FIXED
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: CONFIGURING LoRA")
print("=" * 60)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============================================================
# STEP 6 — FINE-TUNING — FIXED
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: FINE-TUNING")
print("=" * 60)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("results", exist_ok=True)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        bf16=True,
        fp16=False,
        logging_steps=10,
        save_strategy="epoch",
        warmup_steps=50,
        lr_scheduler_type="cosine",
        report_to="none",
        dataloader_num_workers=0,
        max_length=MAX_SEQ_LENGTH,
    ),
)

print(f"Training on {len(train_dataset)} examples")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
print(f"Learning rate: {LEARNING_RATE}")
print()

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nModel saved to {OUTPUT_DIR}")

# ============================================================
# STEP 7 — EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("STEP 7: EVALUATION ON TEST SET")
print("=" * 60)

model.eval()
predictions = []
ground_truth = []
errors = 0
unknowns = 0

print(f"Running inference on {len(test_df)} test entries...")

for i, (_, row) in enumerate(test_df.iterrows()):
    if i % 50 == 0:
        print(f"  Progress: {i}/{len(test_df)}")

    prompt = create_prompt(
        row.get('title', ''),
        row['description'],
        label=None
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=15,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    ).strip()

    pred_label = None
    for label in VALID_LABELS:
        if label in generated:
            pred_label = label
            break

    if pred_label:
        predictions.append(pred_label)
        ground_truth.append(row['label'])
    else:
        unknowns += 1
        print(f"  Unknown output at {i}: '{generated[:50]}'")

print(f"\nInference complete. Unknowns: {unknowns}")

# ============================================================
# STEP 8 — RESULTS
# ============================================================

acc = accuracy_score(ground_truth, predictions)
f1  = f1_score(ground_truth, predictions, average='macro', zero_division=0)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Accuracy:  {acc:.4f} ({acc:.1%})")
print(f"Macro F1:  {f1:.4f}")
print(f"Unknowns:  {unknowns}/{len(test_df)}")

print(f"\nDetailed Report:")
print(classification_report(
    ground_truth, predictions,
    target_names=sorted(VALID_LABELS),
    zero_division=0
))

print("Confusion Matrix:")
cm = confusion_matrix(ground_truth, predictions, labels=sorted(VALID_LABELS))
print(pd.DataFrame(
    cm,
    index=sorted(VALID_LABELS),
    columns=sorted(VALID_LABELS)
).to_string())

# ============================================================
# STEP 9 — COMPARISON TABLE
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
    ("Llama 3.1 8B v1 (naive finetune)", 0.665, 0.665),
    ("Llama 3.1 8B v2 (fixed)",          acc,   f1),
    ("GPT-4 Taxonomy-Aligned (Ours)",    0.970, 0.970),
]

print(f"\n{'Method':<45} {'Accuracy':>10} {'Macro F1':>10}")
print("-" * 67)
for name, a, f in methods:
    marker = " <-- NEW" if "v2" in name else ""
    marker = " <-- OUR METHOD" if "GPT-4" in name else marker
    print(f"{name:<45} {a:>10.1%} {f:>10.1%}{marker}")

# Save results
results = {
    "timestamp": datetime.now().isoformat(),
    "model": MODEL_NAME,
    "version": "v2_fixed",
    "accuracy": acc,
    "f1_macro": f1,
    "unknowns": unknowns,
    "test_size": len(test_df),
    "train_size": len(train_df),
    "hyperparameters": {
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "grad_accumulation": GRAD_ACCUM,
    }
}

with open(RESULTS_FILE, 'w') as f_out:
    json.dump(results, f_out, indent=2)

print(f"\nResults saved to {RESULTS_FILE}")
print("\n" + "=" * 60)
print("EXPERIMENT COMPLETE")
print("=" * 60)
print(f"Llama 3.1 8B v2 accuracy:            {acc:.1%}")
print(f"vs v1 naive finetune:                 66.5%")
print(f"vs Mistral 7B zero-shot:              65.7%")
print(f"vs TF-IDF best baseline:              82.3%")
print(f"vs GPT-4 Taxonomy-Aligned:            97.0%")
print(f"\nImprovement over naive finetune:     +{(acc-0.665)*100:.1f} pp")
print(f"Gap to GPT-4:                        {(0.970-acc)*100:.1f} pp")
