# ==============================
# 1. Install & Import
# ==============================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ==============================
# 2. Load Dataset
# ==============================
# Your CSV should have columns: id, title, description, label

df = pd.read_csv("data/dataset.csv")
# Combine text fields
df["text"] = df["title"] + " " + df["description"]

# ==============================
# 3. Train/Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# ==============================
# 4. TF-IDF Vectorization
# ==============================
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ==============================
# 5. Models
# ==============================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC(),
    "Naive Bayes": MultinomialNB()
}

results = {}

# ==============================
# 6. Train & Evaluate ML Models
# ==============================
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    results[name] = {
        "accuracy": acc,
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"]
    }

    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred))

# ==============================
# 7. Keyword-Based Classifier
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

y_pred_kw = X_test.apply(keyword_classifier)

acc_kw = accuracy_score(y_test, y_pred_kw)
report_kw = classification_report(y_test, y_pred_kw, output_dict=True)

results["Keyword"] = {
    "accuracy": acc_kw,
    "precision": report_kw["weighted avg"]["precision"],
    "recall": report_kw["weighted avg"]["recall"],
    "f1": report_kw["weighted avg"]["f1-score"]
}

print("\n=== Keyword Classifier ===")
print(classification_report(y_test, y_pred_kw))

# ==============================
# 8. Confusion Matrix (SVM Example)
# ==============================

y_pred_svm = models["SVM"].predict(X_test_tfidf)

cm = confusion_matrix(y_test, y_pred_svm, labels=df["label"].unique())

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=df["label"].unique(),
            yticklabels=df["label"].unique())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (SVM)")
plt.show()

# ==============================
# 9. Results Table
# ==============================

results_df = pd.DataFrame(results).T
print("\n=== Final Results ===")
print(results_df)
