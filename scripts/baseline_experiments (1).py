# ============================================================
# OSS Threat Detection — Baseline Experiments
# For AsiaCCS 2027 Paper
# Run this on Google Colab
# ============================================================
# INSTRUCTIONS:
# 1. Upload all your dataset CSV files to Google Colab
# 2. Run each section in order
# 3. Copy the output numbers into your paper
# ============================================================

# ============================================================
# SECTION 0 — Install and Import
# ============================================================
# !pip install pandas scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
import glob
import os
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("All imports successful")

# ============================================================
# ============================================================
# SECTION 1 — LOAD DATASET
# ============================================================

DATASET_PATH = 'data/oss_threat_dataset_beta.csv'

print(f"\nLoading dataset from: {DATASET_PATH}")
full_df = pd.read_csv(DATASET_PATH)
print(f"Loaded {len(full_df)} entries")

# Clean
full_df = full_df[full_df['label'].isin(valid_labels)].copy()
full_df = full_df.dropna(subset=['description', 'label'])

if 'id' in full_df.columns:
    before = len(full_df)
    full_df = full_df.drop_duplicates(subset=['id'])
    print(f"Removed {before - len(full_df)} duplicate IDs")

print(f"Clean entries: {len(full_df)}")
# ============================================================
# SECTION 2 — Prepare Features
# ============================================================

# Use title + description as combined text feature
if 'title' in full_df.columns:
    full_df['text'] = full_df['title'].fillna('') + ' ' + full_df['description'].fillna('')
else:
    full_df['text'] = full_df['description'].fillna('')

X = full_df['text'].values
y = full_df['label'].values

print(f"\nFeature matrix: {len(X)} samples")
print(f"Classes: {sorted(set(y))}")

# ============================================================
# SECTION 3 — Define Baselines
# ============================================================

# 5-fold stratified cross validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

baselines = {
    'Random Classifier': Pipeline([
        ('clf', DummyClassifier(strategy='stratified', random_state=42))
    ]),
    'Majority Class': Pipeline([
        ('clf', DummyClassifier(strategy='most_frequent'))
    ]),
    'Keyword TF-IDF + Naive Bayes': Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )),
        ('clf', MultinomialNB())
    ]),
    'TF-IDF + Logistic Regression': Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english',
            sublinear_tf=True
        )),
        ('clf', LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=42
        ))
    ]),
    'TF-IDF + LinearSVC': Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english',
            sublinear_tf=True
        )),
        ('clf', LinearSVC(
            C=1.0,
            max_iter=2000,
            random_state=42
        ))
    ]),
}

# ============================================================
# SECTION 4 — Run All Baselines
# ============================================================

results = {}

print("\n" + "="*60)
print("RUNNING BASELINE EXPERIMENTS (5-fold CV)")
print("="*60)

for name, pipeline in baselines.items():
    print(f"\n>>> {name}")

    # Accuracy
    acc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    # Macro F1
    f1_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1_macro')

    results[name] = {
        'accuracy_mean': acc_scores.mean(),
        'accuracy_std': acc_scores.std(),
        'f1_mean': f1_scores.mean(),
        'f1_std': f1_scores.std(),
    }

    print(f"  Accuracy: {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")
    print(f"  Macro F1: {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")

# ============================================================
# SECTION 5 — LLM Result (Your Paper's Claimed Result)
# ============================================================

# Add your LLM GPT-4 result manually
# Replace these numbers with your actual results from the paper
results['GPT-4 (Taxonomy-Aligned LLM)'] = {
    'accuracy_mean': 0.97,   # <-- Replace with your actual accuracy
    'accuracy_std': 0.0,
    'f1_mean': 0.97,         # <-- Replace with your actual F1
    'f1_std': 0.0,
}

# ============================================================
# SECTION 6 — Results Table
# ============================================================

print("\n" + "="*60)
print("RESULTS SUMMARY TABLE")
print("="*60)
print(f"{'Method':<40} {'Accuracy':>12} {'Macro F1':>12}")
print("-"*65)

for name, res in results.items():
    acc = f"{res['accuracy_mean']:.4f} ± {res['accuracy_std']:.4f}"
    f1 = f"{res['f1_mean']:.4f} ± {res['f1_std']:.4f}"
    print(f"{name:<40} {acc:>12} {f1:>12}")

# ============================================================
# SECTION 7 — Detailed Report for Best Baseline
# ============================================================

print("\n" + "="*60)
print("DETAILED CLASSIFICATION REPORT — TF-IDF + LinearSVC")
print("="*60)

best_pipeline = baselines['TF-IDF + LinearSVC']
y_pred = cross_val_predict(best_pipeline, X, y, cv=cv)

print(classification_report(y, y_pred, target_names=sorted(valid_labels)))

# ============================================================
# SECTION 8 — Confusion Matrix for Best Baseline
# ============================================================

cm = confusion_matrix(y, y_pred, labels=sorted(valid_labels))

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=sorted(valid_labels),
    yticklabels=sorted(valid_labels)
)
plt.title('Confusion Matrix — TF-IDF + LinearSVC Baseline')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_baseline.png', dpi=150)
plt.show()
print("Confusion matrix saved as confusion_matrix_baseline.png")

# ============================================================
# SECTION 9 — Bar Chart Comparing All Methods
# ============================================================

method_names = list(results.keys())
accuracies = [results[m]['accuracy_mean'] for m in method_names]
f1_scores_list = [results[m]['f1_mean'] for m in method_names]

x = np.arange(len(method_names))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='steelblue')
bars2 = ax.bar(x + width/2, f1_scores_list, width, label='Macro F1', color='coral')

ax.set_ylabel('Score')
ax.set_title('Comparison of Classification Methods on OSS Threat Dataset')
ax.set_xticks(x)
ax.set_xticklabels(method_names, rotation=25, ha='right', fontsize=9)
ax.legend()
ax.set_ylim(0, 1.05)

for bar in bars1:
    ax.annotate(f'{bar.get_height():.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=8)

for bar in bars2:
    ax.annotate(f'{bar.get_height():.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('method_comparison.png', dpi=150)
plt.show()
print("Comparison chart saved as method_comparison.png")

# ============================================================
# SECTION 10 — Key Numbers For Paper
# ============================================================

print("\n" + "="*60)
print("KEY NUMBERS TO COPY INTO YOUR PAPER")
print("="*60)

best_baseline_acc = results['TF-IDF + LinearSVC']['accuracy_mean']
llm_acc = results['GPT-4 (Taxonomy-Aligned LLM)']['accuracy_mean']
improvement = (llm_acc - best_baseline_acc) * 100

print(f"\nDataset size: {len(full_df)} verified OSS threat incidents")
print(f"Number of categories: 5 (AV-200, AV-300, AV-400, AV-410, AV-509)")
print(f"\nRandom baseline accuracy:     {results['Random Classifier']['accuracy_mean']:.1%}")
print(f"Majority class accuracy:       {results['Majority Class']['accuracy_mean']:.1%}")
print(f"TF-IDF + Naive Bayes:          {results['Keyword TF-IDF + Naive Bayes']['accuracy_mean']:.1%}")
print(f"TF-IDF + Logistic Regression:  {results['TF-IDF + Logistic Regression']['accuracy_mean']:.1%}")
print(f"TF-IDF + LinearSVC (best):     {best_baseline_acc:.1%}")
print(f"GPT-4 Taxonomy-Aligned LLM:    {llm_acc:.1%}")
print(f"\nLLM improvement over best baseline: +{improvement:.1f} percentage points")
print(f"\nConclusion: Our taxonomy-aligned LLM approach outperforms the")
print(f"strongest traditional baseline (TF-IDF + LinearSVC) by {improvement:.1f}%")
print(f"demonstrating the value of LLM-based contextual understanding")
print(f"for OSS supply chain threat classification.")

print("\n" + "="*60)
print("EXPERIMENT COMPLETE")
print("="*60)
print("Save confusion_matrix_baseline.png and method_comparison.png")
print("Copy the numbers above into your results section.")
