import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    df = pd.read_csv("data/oss_threat_dataset.csv")

    if 'predicted_label' not in df.columns:
        print("⚠️ No 'predicted_label' column found. Skipping accuracy evaluation.")
        return

    print("\n✅ Dataset loaded with predictions!")
    print(f"🔢 Total samples: {len(df)}")
    print("\n🧾 Ground truth label distribution:")
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
