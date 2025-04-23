# scripts/evaluate_with_predictions.py

import pandas as pd
import argparse
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

def analyze_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise SystemExit(f"🚨 Error: File not found at {file_path}")

    # Validate required columns
    required_cols = ['label', 'predicted_label']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise SystemExit(f"🚨 Missing required columns: {', '.join(missing_cols)}")

    print("\n📊 Dataset Summary:")
    print(f"• Total samples: {len(df):,}")
    print(f"• Features: {len(df.columns)} columns")
    print(f"• Columns: {', '.join(df.columns)}")
    
    print("\n🔍 Data Types:")
    print(df.dtypes.to_string())
    
    print("\n❓ Missing Values:")
    missing = df.isnull().sum()
    print(missing[missing > 0].to_string() or "No missing values found")

    print("\n📈 True Label Distribution:")
    label_counts = df['label'].value_counts()
    label_percents = df['label'].value_counts(normalize=True).mul(100).round(2)
    
    print("\nCount:")
    print(label_counts.to_string())
    print("\nPercentage:")
    print(label_percents.astype(str) + "%")

    return df

def evaluate_predictions(df):
    print("\n🔮 Evaluating predictions...")
    
    # Calculate metrics
    acc = accuracy_score(df['label'], df['predicted_label'])
    clf_report = classification_report(df['label'], df['predicted_label'])
    cm = confusion_matrix(df['label'], df['predicted_label'])

    # Print results
    print(f"\n🎯 Accuracy: {acc:.2%}")
    print("\n📊 Classification Report:")
    print(clf_report)
    print("📉 Confusion Matrix:")
    print(cm)

def main():
    parser = argparse.ArgumentParser(description="Evaluate OSS Threat Predictions")
    parser.add_argument("--input", type=str, default="data/oss_threat_dataset.csv",
                      help="Path to input CSV file with predictions")
    args = parser.parse_args()

    input_path = Path(args.input)
    print(f"🔎 Analyzing dataset at: {input_path.resolve()}")

    try:
        df = analyze_dataset(input_path)
        evaluate_predictions(df)
    except Exception as e:
        raise SystemExit(f"🚨 Analysis failed: {str(e)}")

    print("\n✅ Evaluation complete!")

if __name__ == "__main__":
    main()
