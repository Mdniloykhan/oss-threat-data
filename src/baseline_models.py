import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==============================
# 1. Load Dataset
# ==============================
df = pd.read_csv("data/oss_threat_dataset_beta")

# Combine title + description
df["text"] = df["title"] + " " + df["description"]

# Ground truth labels
X = df["text"]
y = df["label"]

# ==============================
# 2. Train/Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 3. TF-IDF
# ==============================
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ==============================
# 4. Models
# ==============================
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC(),
    "NaiveBayes": MultinomialNB()
}

results = []

# Create results folder
os.makedirs("results", exist_ok=True)

# ==============================
# 5. Train & Evaluate
# ==============================
for name, model in models.items():
    print(f"\n===== {name} =====")

    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Accuracy:", acc)
    print(report)

    results.append({
        "model": name,
        "accuracy": acc
    })

    # Save predictions
    pred_df = pd.DataFrame({
        "text": X_test,
        "true_label": y_test,
        "predicted_label": y_pred
    })

    pred_df.to_csv(f"results/{name}_predictions.csv", index=False)

# ==============================
# 6. Keyword Baseline
# ==============================
def keyword_classifier(text):
    text = text.lower()

    if "typo" in text or "similar name" in text:
        return "AV-200"
    elif "unicode" in text or "obfuscation" in text:
        return "AV-300"
    elif "pipeline" in text or "ci" in text:
        return "AV-410"
    elif "dependency" in text or "confusion" in text:
        return "AV-509"
    else:
        return "AV-400"

print("\n===== Keyword Baseline =====")

y_pred_kw = X_test.apply(keyword_classifier)

acc_kw = accuracy_score(y_test, y_pred_kw)
print("Accuracy:", acc_kw)
print(classification_report(y_test, y_pred_kw))

results.append({
    "model": "Keyword",
    "accuracy": acc_kw
})

# Save keyword predictions
pd.DataFrame({
    "text": X_test,
    "true_label": y_test,
    "predicted_label": y_pred_kw
}).to_csv("results/keyword_predictions.csv", index=False)

# ==============================
# 7. Confusion Matrix (SVM)
# ==============================
y_pred_svm = models["SVM"].predict(X_test_tfidf)

cm = confusion_matrix(y_test, y_pred_svm, labels=sorted(y.unique()))

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=sorted(y.unique()),
            yticklabels=sorted(y.unique()))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (SVM)")
plt.savefig("results/confusion_matrix.png")

# ==============================
# 8. Save Results Summary
# ==============================
results_df = pd.DataFrame(results)
results_df.to_csv("results/metrics.csv", index=False)

print("\nAll results saved in /results/")
