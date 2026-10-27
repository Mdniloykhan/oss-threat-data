#!/usr/bin/env python3
# ============================================================
# GPT-4 Taxonomy-Aligned Classification Script
# Autonomous OSS Threat Detection via Taxonomy-Aligned LLMs
# ============================================================
# Reproduces the GPT-4 taxonomy-aligned results reported in:
# "Autonomous OSS Threat Detection via Taxonomy-Aligned LLMs"
# arXiv: https://arxiv.org/abs/2610.01263
# Dataset: https://github.com/Mdniloykhan/oss-threat-data
# ============================================================
# Usage:
#   pip install openai pandas scikit-learn tqdm
#   python gpt4_taxonomy_classification.py \
#       --api-key YOUR_OPENAI_KEY \
#       --input data/oss_threat_dataset_beta.csv \
#       --output results/gpt4_results.json
# ============================================================
# Expected results (as reported in paper):
#   Accuracy: 97.0%
#   Macro F1: 97.0%
#   Dataset:  999 verified OSS supply chain incidents
# ============================================================

import openai
import pandas as pd
import numpy as np
import json
import time
import argparse
import os
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from tqdm import tqdm

# ============================================================
# TAXONOMY-ALIGNED SYSTEM PROMPT
# This exact prompt was used to generate the 97.0% results
# reported in the paper. Do not modify for reproduction.
# ============================================================

TAXONOMY_PROMPT = """You are an expert in open source software supply chain security. \
Classify the following security incident into exactly one of these categories:

AV-200 (Typosquatting): Malicious packages that mimic legitimate package names \
through character substitution, transposition, or namespace impersonation.

AV-300 (Trojan Source): Attacks exploiting Unicode bidirectional control characters \
or homoglyphs to create source code that executes differently from its visual representation.

AV-400 (Malicious Builds): Packages containing malicious code executed during \
installation, build, or import processes through lifecycle hooks such as postinstall \
and setup.py.

AV-410 (Pipeline Poisoning): Attacks that compromise CI/CD pipeline execution including \
GitHub Actions workflow injection, poisoned pipeline execution (PPE), and build system compromise.

AV-509 (Dependency Confusion): Attacks exploiting package manager resolution priority \
to cause build systems to install malicious public packages instead of intended private \
internal dependencies.

Respond with ONLY the category label (AV-200, AV-300, AV-400, AV-410, or AV-509). \
No explanation. No punctuation. Just the label."""

VALID_LABELS = ["AV-200", "AV-300", "AV-400", "AV-410", "AV-509"]

# ============================================================
# CLASSIFICATION FUNCTION
# ============================================================

def classify_incident(client, title, description, model="gpt-4",
                       temperature=0, max_retries=3):
    """
    Classify a single OSS supply chain incident using
    taxonomy-aligned GPT-4 prompting.

    Args:
        client: OpenAI client instance
        title: Incident title string
        description: Incident description string
        model: OpenAI model name (default: gpt-4)
        temperature: Sampling temperature (0 for deterministic)
        max_retries: Number of retries on API error

    Returns:
        str: Predicted AV-xxx label
    """
    text = f"Title: {str(title).strip()}\n\nDescription: {str(description).strip()}"

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": TAXONOMY_PROMPT},
                    {"role": "user", "content": text}
                ],
                temperature=temperature,
                max_tokens=15
            )
            label = response.choices[0].message.content.strip()

            # Validate label
            if label in VALID_LABELS:
                return label

            # Try to extract valid label from response
            for valid in VALID_LABELS:
                if valid in label:
                    return valid

            # Default to AV-400 if no valid label found
            print(f"  Warning: Invalid label '{label}' — defaulting to AV-400")
            return "AV-400"

        except openai.RateLimitError:
            wait_time = (attempt + 1) * 30
            print(f"  Rate limit hit. Waiting {wait_time}s...")
            time.sleep(wait_time)

        except openai.APIError as e:
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                raise e

    return "AV-400"  # Final fallback


# ============================================================
# EVALUATION FUNCTIONS
# ============================================================

def evaluate(y_true, y_pred, labels=VALID_LABELS):
    """Compute accuracy, macro F1, and per-class metrics."""
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro',
                        zero_division=0)
    report = classification_report(
        y_true, y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return acc, f1_macro, report, cm


def print_results(acc, f1_macro, report, cm, labels=VALID_LABELS):
    """Print formatted evaluation results."""
    print("\n" + "=" * 55)
    print("EVALUATION RESULTS")
    print("=" * 55)
    print(f"Accuracy:   {acc:.4f} ({acc:.1%})")
    print(f"Macro F1:   {f1_macro:.4f} ({f1_macro:.1%})")
    print()
    print("Per-class results:")
    print(f"{'Category':<30} {'P':>6} {'R':>6} {'F1':>6} {'N':>6}")
    print("-" * 55)
    for label in labels:
        if label in report:
            r = report[label]
            print(f"{label:<30} "
                  f"{r['precision']:>6.3f} "
                  f"{r['recall']:>6.3f} "
                  f"{r['f1-score']:>6.3f} "
                  f"{int(r['support']):>6}")
    print("-" * 55)
    print(f"{'Macro Average':<30} "
          f"{report.get('macro avg', {}).get('precision', 0):>6.3f} "
          f"{report.get('macro avg', {}).get('recall', 0):>6.3f} "
          f"{f1_macro:>6.3f} "
          f"{sum(int(report[l]['support']) for l in labels if l in report):>6}")
    print()
    print("Confusion Matrix:")
    print(f"{'':>12}", end="")
    for label in labels:
        print(f"{label:>10}", end="")
    print()
    for i, true_label in enumerate(labels):
        print(f"{true_label:>12}", end="")
        for j in range(len(labels)):
            print(f"{cm[i][j]:>10}", end="")
        print()
    print("=" * 55)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="GPT-4 Taxonomy-Aligned OSS Threat Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full dataset classification
  python gpt4_taxonomy_classification.py \\
      --api-key sk-... \\
      --input data/oss_threat_dataset_beta.csv \\
      --output results/gpt4_results.json

  # Test on sample of 50 entries
  python gpt4_taxonomy_classification.py \\
      --api-key sk-... \\
      --input data/oss_threat_dataset_beta.csv \\
      --output results/gpt4_sample_results.json \\
      --sample 50

  # Resume interrupted run
  python gpt4_taxonomy_classification.py \\
      --api-key sk-... \\
      --input data/oss_threat_dataset_beta.csv \\
      --output results/gpt4_results.json \\
      --resume results/gpt4_predictions_partial.csv
        """
    )

    parser.add_argument('--api-key', type=str, default='',
                        help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--input', type=str,
                        default='data/oss_threat_dataset_beta.csv',
                        help='Input dataset CSV path')
    parser.add_argument('--output', type=str,
                        default='results/gpt4_results.json',
                        help='Output results JSON path')
    parser.add_argument('--model', type=str, default='gpt-4',
                        help='OpenAI model (default: gpt-4)')
    parser.add_argument('--sample', type=int, default=None,
                        help='Evaluate on random sample of N entries')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from partial predictions CSV')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Delay between API calls in seconds (default: 0.5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling (default: 42)')

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("Error: OpenAI API key required.")
        print("Provide via --api-key or set OPENAI_API_KEY environment variable.")
        return

    print("=" * 55)
    print("GPT-4 TAXONOMY-ALIGNED OSS THREAT CLASSIFICATION")
    print("=" * 55)
    print(f"Model:  {args.model}")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")

    # Initialize OpenAI client
    client = openai.OpenAI(api_key=api_key)

    # Load dataset
    try:
        df = pd.read_csv(args.input)
        print(f"\nLoaded {len(df)} entries from {args.input}")
    except FileNotFoundError:
        print(f"Error: {args.input} not found.")
        print("Download dataset from: https://github.com/Mdniloykhan/oss-threat-data")
        return

    # Validate required columns
    required_cols = ['title', 'description', 'label']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Error: Missing columns: {missing}")
        return

    # Filter to valid labels only
    df = df[df['label'].isin(VALID_LABELS)].reset_index(drop=True)
    print(f"Valid labeled entries: {len(df)}")

    # Label distribution
    print("\nGround truth distribution:")
    for label in VALID_LABELS:
        count = (df['label'] == label).sum()
        print(f"  {label}: {count} ({count/len(df):.1%})")

    # Sample if requested
    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=args.seed)
        df = df.reset_index(drop=True)
        print(f"\nSampled {args.sample} entries (seed={args.seed})")

    # Handle resume
    already_predicted = {}
    if args.resume and os.path.exists(args.resume):
        df_resume = pd.read_csv(args.resume)
        for _, row in df_resume.iterrows():
            if pd.notna(row.get('predicted_label')):
                already_predicted[str(row.get('id', ''))] = row['predicted_label']
        print(f"\nResuming: {len(already_predicted)} predictions already done")

    # Create output directory
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)

    # Classification loop
    predictions = []
    errors = 0
    skipped = 0
    partial_path = args.output.replace('.json', '_predictions_partial.csv')

    print(f"\nClassifying {len(df)} entries with {args.model}...")
    estimated_cost = len(df) * 0.002
    print(f"Estimated API cost: ${estimated_cost:.2f}")
    print()

    for i, row in enumerate(tqdm(df.itertuples(), total=len(df),
                                  desc="Classifying")):
        row_id = str(getattr(row, 'id', i))

        # Check if already predicted (resume mode)
        if row_id in already_predicted:
            predictions.append(already_predicted[row_id])
            skipped += 1
            continue

        try:
            pred = classify_incident(
                client=client,
                title=getattr(row, 'title', ''),
                description=getattr(row, 'description', ''),
                model=args.model
            )
            predictions.append(pred)

        except Exception as e:
            print(f"\n  Error on row {i} (id={row_id}): {e}")
            predictions.append("AV-400")
            errors += 1

        # Save partial results every 50 entries
        if (i + 1) % 50 == 0:
            df_partial = df.iloc[:len(predictions)].copy()
            df_partial['predicted_label'] = predictions
            df_partial.to_csv(partial_path, index=False)

        time.sleep(args.delay)

    # Final evaluation
    df['predicted_label'] = predictions
    y_true = df['label'].tolist()
    y_pred = predictions

    acc, f1_macro, report, cm = evaluate(y_true, y_pred)
    print_results(acc, f1_macro, report, cm)

    # Save results
    results = {
        "model": args.model,
        "prompt": TAXONOMY_PROMPT,
        "dataset": args.input,
        "n_total": len(df),
        "n_errors": errors,
        "n_skipped_resumed": skipped,
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "per_class": {
            label: {
                "precision": round(report[label]['precision'], 4),
                "recall": round(report[label]['recall'], 4),
                "f1": round(report[label]['f1-score'], 4),
                "support": int(report[label]['support'])
            }
            for label in VALID_LABELS if label in report
        },
        "confusion_matrix": cm.tolist(),
        "labels": VALID_LABELS,
        "paper": "Autonomous OSS Threat Detection via Taxonomy-Aligned LLMs",
        "arxiv": "https://arxiv.org/abs/2610.01263",
        "dataset_url": "https://github.com/Mdniloykhan/oss-threat-data"
    }

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Save predictions CSV
    pred_path = args.output.replace('.json', '_predictions.csv')
    df.to_csv(pred_path, index=False)
    print(f"Predictions saved to {pred_path}")

    # Clean up partial file
    if os.path.exists(partial_path):
        os.remove(partial_path)

    # Summary
    print(f"\n{'='*55}")
    print("SUMMARY")
    print(f"{'='*55}")
    print(f"Entries classified: {len(df)}")
    print(f"Errors:             {errors}")
    print(f"Accuracy:           {acc:.1%}")
    print(f"Macro F1:           {f1_macro:.1%}")

    # Compare to paper results
    paper_acc = 0.970
    paper_f1  = 0.970
    print(f"\nComparison to paper results:")
    print(f"  Accuracy:  {acc:.1%} (paper: {paper_acc:.1%}, diff: {abs(acc-paper_acc):.1%})")
    print(f"  Macro F1:  {f1_macro:.1%} (paper: {paper_f1:.1%}, diff: {abs(f1_macro-paper_f1):.1%})")

    if abs(acc - paper_acc) <= 0.02:
        print("\n✅ Results within 2% of paper — reproduction successful")
    else:
        print("\n⚠️  Results differ >2% from paper")
        print("   Note: GPT-4 model versions may cause minor variations")
        print("   Paper used gpt-4 with temperature=0 in October 2026")

    print("\nDone.")


if __name__ == "__main__":
    main()
