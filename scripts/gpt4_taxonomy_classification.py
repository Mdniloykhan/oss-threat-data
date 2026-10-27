# gpt4_taxonomy_classification.py

import openai
import pandas as pd
import json
from sklearn.metrics import accuracy_score, f1_score

TAXONOMY_PROMPT = """You are an expert in OSS supply chain security.
Classify the following incident into exactly one category:
AV-200 (Typosquatting)
AV-300 (Trojan Source)
AV-400 (Malicious Builds)
AV-410 (Pipeline Poisoning)
AV-509 (Dependency Confusion)
Respond with ONLY the category label."""

def classify_incident(client, title, description):
    text = f"Title: {title}\n\nDescription: {description}"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": TAXONOMY_PROMPT},
            {"role": "user", "content": text}
        ],
        temperature=0,
        max_tokens=10
    )
    return response.choices[0].message.content.strip()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/oss_threat_dataset_beta.csv')
    parser.add_argument('--output', default='results/gpt4_results.json')
    parser.add_argument('--api-key', required=True)
    args = parser.parse_args()

    client = openai.OpenAI(api_key=args.api_key)
    df = pd.read_csv(args.input)

    VALID = ['AV-200','AV-300','AV-400','AV-410','AV-509']
    df = df[df['label'].isin(VALID)].reset_index(drop=True)

    predictions = []
    errors = 0

    for i, row in df.iterrows():
        try:
            pred = classify_incident(
                client,
                str(row.get('title','')),
                str(row.get('description',''))
            )
            if pred not in VALID:
                pred = 'AV-400'
            predictions.append(pred)
        except Exception as e:
            print(f"Error on row {i}: {e}")
            predictions.append('AV-400')
            errors += 1

    df['predicted_label'] = predictions
    acc = accuracy_score(df['label'], predictions)
    f1 = f1_score(df['label'], predictions, average='macro')

    results = {
        'model': 'gpt-4',
        'accuracy': round(acc, 3),
        'f1_macro': round(f1, 3),
        'n': len(df),
        'errors': errors,
        'prompt': TAXONOMY_PROMPT
    }

    print(f"Accuracy: {acc:.3f}")
    print(f"Macro F1: {f1:.3f}")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    df.to_csv(args.output.replace('.json','_predictions.csv'), index=False)
    print(f"Results saved to {args.output}")

if __name__ == '__main__':
    main()