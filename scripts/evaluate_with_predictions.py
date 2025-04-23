# scripts/evaluate.py

import pandas as pd

def main():
    df = pd.read_csv("data/oss_threat_dataset.csv")

    print("\n📊 Dataset Summary:")
    print(f"Total samples: {len(df)}")
    print("\n🧾 Label distribution:")
    print(df['label'].value_counts())

print("\n🔮 Evaluating predictions...")

acc = accuracy_score(df['label'], df['predicted_label'])
print(f"\n🎯 Accuracy: {acc:.2%}\n")

print("📊 Classification Report:")
print(classification_report(df['label'], df['predicted_label']))

print("📉 Confusion Matrix:")
print(confusion_matrix(df['label'], df['predicted_label']))


if __name__ == "__main__":
    main()
