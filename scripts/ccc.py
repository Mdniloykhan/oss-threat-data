# scripts/evaluate.py

import pandas as pd

def main():
    df = pd.read_csv("data/oss_threat_dataset_v3.csv")

    print("\n📊 Dataset Summary:")
    print(f"Total samples: {len(df)}")
    print("\n🧾 Label distribution:")
    print(df['label'].value_counts())

if __name__ == "__main__":
    main()
