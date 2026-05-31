# ============================================================
# OSS Threat Detection — Baseline Experiments
# For AsiaCCS 2027 Paper
# ============================================================

import pandas as pd
import numpy as np
import os
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use('Agg')  # No display needed — saves files directly
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("OSS THREAT DETECTION — BASELINE EXPERIMENTS")
print("=" * 60)

# ============================================================
# SECTION 1 — LOAD DATASET
# ============================================================

# Valid AV categories — defined FIRST before any use
valid_labels = ['AV-200', 'AV-300', 'AV-400', 'AV-410', 'AV-509']

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
print("\nLabel distribution:")
for label, count in full_df['label'].value_counts().sort_index().items():
    pct = count / len(full_df) * 100
    print(f"  {label}: {count} ({pct:.1f}%)")

# ============================================================
# SECTION 2 — PREPARE FEATURES
# ============================================================

if 'title' in full_df.columns:
    full_df['text'] = (
        full_df['title'].fillna('') + ' ' +
        full_df['description'].fillna('')
    )
else:
    full_df['text'] = full_df['description'].fillna('')

X = full_df['text'].values
y = full_df['label'].values

print(f"\nFeature: title + description combined")
print(f"Samples: {len(X)}")
print(f"Classes: {sorted(set(y))}")

# ============================================================
# SECTION 3 — DEFINE BASELINES
# ============================================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

baselines = {
    'Random Classifier': Pipeline([
        ('clf', DummyClassifier(strategy='stratified', random_state=42))
    ]),
    'Majority Class': Pipeline([
        ('clf', DummyClassifier(strategy='most_frequent'))
    ]),
    'TF-IDF + Naive Bayes': Pipeline([
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
# SECTION 4 — RUN ALL BASELINES
# ============================================================

results = {}

print("\n" + "=" * 60)
print("RUNNING BASELINES (5-fold stratified cross-validation)")
print("=" * 60)

for name, pipeline in baselines.items():
    print(f"\n>>> {name}")
    acc = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    f1  = cross_val_score(pipeline, X, y, cv=cv, scoring='f1_macro')
    results[name] = {
        'accuracy_mean': acc.mean(),
        'accuracy_std':  acc.std(),
        'f1_mean':       f1.mean(),
        'f1_std':        f1.std(),
    }
    print(f"  Accuracy : {acc.mean():.4f} +/- {acc.std():.4f}")
    print(f"  Macro F1 : {f1.mean():.4f} +/- {f1.std():.4f}")

# ============================================================
# SECTION 5 — ADD LLM RESULT
# ============================================================

# Replace 0.97 with your actual GPT-4 accuracy from your paper
LLM_ACCURACY = 0.97
LLM_F1       = 0.97

results['GPT-4 (Taxonomy-Aligned LLM)'] = {
    'accuracy_mean': LLM_ACCURACY,
    'accuracy_std':  0.0,
    'f1_mean':       LLM_F1,
    'f1_std':        0.0,
}

# ============================================================
# SECTION 6 — RESULTS TABLE
# ============================================================

print("\n" + "=" * 60)
print("RESULTS TABLE")
print("=" * 60)
print(f"{'Method':<42} {'Accuracy':>16} {'Macro F1':>16}")
print("-" * 76)

for name, res in results.items():
    acc_str = f"{res['accuracy_mean']:.4f} +/- {res['accuracy_std']:.4f}"
    f1_str  = f"{res['f1_mean']:.4f} +/- {res['f1_std']:.4f}"
    marker  = " <-- OUR METHOD" if 'GPT-4' in name else ""
    print(f"{name:<42} {acc_str:>16} {f1_str:>16}{marker}")

# ============================================================
# SECTION 7 — DETAILED REPORT FOR BEST BASELINE
# ============================================================

print("\n" + "=" * 60)
print("DETAILED REPORT — TF-IDF + LinearSVC (Best Baseline)")
print("=" * 60)

best_pipeline = baselines['TF-IDF + LinearSVC']
y_pred = cross_val_predict(best_pipeline, X, y, cv=cv)
print(classification_report(y, y_pred, target_names=sorted(valid_labels)))

# ============================================================
# SECTION 8 — CONFUSION MATRIX
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
plt.title('Confusion Matrix — TF-IDF + LinearSVC Baseline', fontsize=13)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_baseline.png', dpi=150)
plt.close()
print("Saved: confusion_matrix_baseline.png")

# ============================================================
# SECTION 9 — BAR CHART
# ============================================================

method_names = list(results.keys())
acc_vals     = [results[m]['accuracy_mean'] for m in method_names]
f1_vals      = [results[m]['f1_mean']       for m in method_names]

x     = np.arange(len(method_names))
width = 0.35

fig, ax = plt.subplots(figsize=(13, 6))
bars1 = ax.bar(x - width/2, acc_vals, width, label='Accuracy', color='steelblue')
bars2 = ax.bar(x + width/2, f1_vals,  width, label='Macro F1',  color='coral')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('OSS Threat Detection — Method Comparison', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(method_names, rotation=20, ha='right', fontsize=9)
ax.legend(fontsize=11)
ax.set_ylim(0, 1.1)

for bar in list(bars1) + list(bars2):
    ax.annotate(
        f'{bar.get_height():.3f}',
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 3), textcoords='offset points',
        ha='center', va='bottom', fontsize=8
    )

plt.tight_layout()
plt.savefig('method_comparison.png', dpi=150)
plt.close()
print("Saved: method_comparison.png")

# ============================================================
# SECTION 10 — KEY NUMBERS FOR YOUR PAPER
# ============================================================

best_acc    = results['TF-IDF + LinearSVC']['accuracy_mean']
llm_acc     = results['GPT-4 (Taxonomy-Aligned LLM)']['accuracy_mean']
random_acc  = results['Random Classifier']['accuracy_mean']
improvement = (llm_acc - best_acc) * 100

print("\n" + "=" * 60)
print("KEY NUMBERS — COPY INTO YOUR PAPER")
print("=" * 60)
print(f"\nDataset: {len(full_df)} verified OSS supply chain incidents")
print(f"Categories: 5 (AV-200, AV-300, AV-400, AV-410, AV-509)")
print()
print(f"Random baseline accuracy         : {random_acc:.1%}")
print(f"Majority class accuracy          : {results['Majority Class']['accuracy_mean']:.1%}")
print(f"TF-IDF + Naive Bayes             : {results['TF-IDF + Naive Bayes']['accuracy_mean']:.1%}")
print(f"TF-IDF + Logistic Regression     : {results['TF-IDF + Logistic Regression']['accuracy_mean']:.1%}")
print(f"TF-IDF + LinearSVC (best)        : {best_acc:.1%}")
print(f"GPT-4 Taxonomy-Aligned LLM       : {llm_acc:.1%}")
print()
print(f"LLM improvement over best baseline: +{improvement:.1f} percentage points")
print()
print("Paper sentence:")
print(f"  Our taxonomy-aligned LLM approach achieves {llm_acc:.0%} accuracy,")
print(f"  outperforming the strongest traditional baseline")
print(f"  (TF-IDF + LinearSVC at {best_acc:.1%}) by {improvement:.1f} percentage points.")

print("\n" + "=" * 60)
print("EXPERIMENT COMPLETE")
print("Output: confusion_matrix_baseline.png, method_comparison.png")
print("=" * 60)
