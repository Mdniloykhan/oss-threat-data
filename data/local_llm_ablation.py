#!/usr/bin/env python3
"""
Local LLM prompt ablation for OSS supply-chain threat classification.

Free: runs entirely on your own machine through Ollama (https://ollama.com).
Uses ONE model for every condition, so differences come from the prompt only:

  labels_only  - only the five codes (AV-200 ... AV-509); no names, no definitions
  names        - codes with short names (the prompt printed in the paper)
  definitions  - codes, names AND one-sentence definitions

Every raw model answer is saved, so every reported number can be audited later.
Ground truth is the `label` column only. The `predicted_label` column is never read.
Unparseable answers are counted as WRONG (nothing is silently defaulted to a class).

Quick test (5 incidents per class):
    python local_llm_ablation.py --model llama3.1:8b --per-class 5
Full run on all rows:
    python local_llm_ablation.py --model llama3.1:8b

Use the ORIGINAL 999-entry dataset file, not the expanded/merged ones.
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from tqdm import tqdm

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
LABELS = ["AV-200", "AV-300", "AV-400", "AV-410", "AV-509"]
LABEL_RE = re.compile(r"AV-(200|300|400|410|509)")

_INTRO = "You are an expert in open source software supply chain security. "

PROMPTS = {
    "labels_only": (
        _INTRO
        + "Classify the following security incident into exactly one of these "
          "categories: AV-200, AV-300, AV-400, AV-410, AV-509. "
          "Respond with only the category label."
    ),
    # Exactly as printed in the paper's Methodology section.
    "names": (
        _INTRO
        + "Classify the following security incident into exactly one of these "
          "categories: AV-200 (Typosquatting), AV-300 (Trojan Source), "
          "AV-400 (Malicious Builds), AV-410 (Pipeline Poisoning), "
          "AV-509 (Dependency Confusion). Respond with only the category label."
    ),
    "definitions": (
        _INTRO
        + "Classify the following security incident into exactly one of these "
          "categories:\n\n"
          "AV-200 (Typosquatting): Malicious packages that mimic legitimate package "
          "names through character substitution, transposition, or namespace "
          "impersonation.\n"
          "AV-300 (Trojan Source): Attacks that use Unicode bidirectional control "
          "characters or homoglyphs so that source code executes differently from "
          "how it appears.\n"
          "AV-400 (Malicious Builds): Packages whose malicious code runs during "
          "installation, build, or import through lifecycle hooks such as "
          "postinstall or setup.py.\n"
          "AV-410 (Pipeline Poisoning): Attacks that compromise CI/CD pipeline "
          "execution, including GitHub Actions workflow injection, poisoned "
          "pipeline execution, and build system compromise.\n"
          "AV-509 (Dependency Confusion): Attacks that exploit package manager "
          "resolution priority so build systems install a malicious public package "
          "instead of the intended private internal dependency.\n\n"
          "Respond with only the category label."
    ),
}
CONDITIONS = list(PROMPTS)


# ---------------------------------------------------------------- Ollama I/O
def check_ollama(model):
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        r.raise_for_status()
    except Exception as e:
        sys.exit(f"Cannot reach Ollama at {OLLAMA_URL}.\n"
                 "Install it from https://ollama.com and make sure it is running.\n"
                 f"(details: {e})")
    names = [m.get("name", "") for m in r.json().get("models", [])]
    ok = any(n == model or n.split(":")[0] == model for n in names)
    if not ok:
        sys.exit(f"Model '{model}' is not installed.\n"
                 f"Run:  ollama pull {model}\nInstalled models: {names or 'none'}")


def ollama_version():
    try:
        return requests.get(f"{OLLAMA_URL}/api/version", timeout=10).json().get(
            "version", "unknown")
    except Exception:
        return "unknown"


def ask(model, system_prompt, user_text, seed, retries=3):
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "options": {"temperature": 0, "seed": seed, "num_predict": 24},
    }
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=300)
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Ollama request failed after {retries} tries: {last}")


def parse(raw):
    m = LABEL_RE.search(raw or "")
    return f"AV-{m.group(1)}" if m else "INVALID"


# ---------------------------------------------------------------------- data
def load_data(path, per_class, seed):
    if not os.path.exists(path):
        alt = os.path.basename(path)
        if os.path.exists(alt):
            path = alt
        else:
            sys.exit(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    for c in ("title", "description", "label"):
        if c not in df.columns:
            sys.exit(f"Missing column '{c}' in {path}")
    df = df[df["label"].isin(LABELS)].copy()
    df["title"] = df["title"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    df = df[(df["title"].str.strip() != "") | (df["description"].str.strip() != "")]
    if "id" not in df.columns:
        df["id"] = [f"row{i}" for i in range(len(df))]
    df["id"] = df["id"].astype(str)
    df = df.drop_duplicates(subset="id")
    if per_class:
        n = min(per_class, int(df["label"].value_counts().min()))
        df = df.groupby("label", group_keys=False).sample(n=n, random_state=seed)
    df = df.reset_index(drop=True)
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    return df, path, sha


# ----------------------------------------------------------------- running
def run_condition(df, model, cond, outdir, seed):
    safe = model.replace(":", "-").replace("/", "-")
    path = os.path.join(outdir, f"{safe}__{cond}__predictions.csv")
    done = set()
    new_file = not os.path.exists(path)
    if not new_file:
        done = set(pd.read_csv(path, dtype=str)["id"].tolist())

    todo = df[~df["id"].isin(done)]
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["id", "label", "raw", "pred", "seconds"])
        for row in tqdm(todo.itertuples(index=False), total=len(todo),
                        desc=f"{cond:<12}"):
            text = f"Title: {row.title.strip()}\n\nDescription: {row.description.strip()[:2000]}"
            t0 = time.time()
            raw = ask(model, PROMPTS[cond], text, seed)
            w.writerow([row.id, row.label, raw.replace("\n", " "),
                        parse(raw), round(time.time() - t0, 2)])
            f.flush()

    out = pd.read_csv(path, dtype=str).fillna("")
    out = out[out["id"].isin(set(df["id"]))].drop_duplicates(subset="id")
    return out, path


def evaluate(pred_df, seed):
    y_true = pred_df["label"].tolist()
    y_pred = pred_df["pred"].tolist()
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    correct = np.array([a == b for a, b in zip(y_true, y_pred)], dtype=float)
    rng = np.random.default_rng(seed)
    boots = [rng.choice(correct, size=len(correct), replace=True).mean()
             for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "n": len(y_true),
        "accuracy": round(float(acc), 4),
        "accuracy_ci95": [round(float(lo), 4), round(float(hi), 4)],
        "macro_f1": round(float(f1), 4),
        "invalid_rate": round(float(np.mean([p == "INVALID" for p in y_pred])), 4),
        "per_class": classification_report(y_true, y_pred, labels=LABELS,
                                           output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Ollama model, e.g. llama3.1:8b")
    ap.add_argument("--input", default="data/oss_threat_dataset_beta.csv")
    ap.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    ap.add_argument("--per-class", type=int, default=0,
                    help="Stratified sample size per class (0 = use all rows)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="results_local_llm")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df, path, sha = load_data(args.input, args.per_class, args.seed)
    check_ollama(args.model)

    print(f"Model: {args.model}   Dataset: {path}   Rows used: {len(df)}")
    print("Rows per class:", df["label"].value_counts().sort_index().to_dict())

    results = {}
    for cond in args.conditions:
        pred_df, pred_path = run_condition(df, args.model, cond, args.outdir, args.seed)
        results[cond] = evaluate(pred_df, args.seed)
        results[cond]["predictions_file"] = pred_path

    print("\n" + "=" * 72)
    print(f"RESULTS for {args.model}  (this is NOT GPT-4)")
    print("=" * 72)
    print(f"{'condition':<13}{'n':>6}   {'accuracy [95% CI]':<26}{'macro-F1':>9}{'invalid':>9}")
    for cond, r in results.items():
        lo, hi = r["accuracy_ci95"]
        print(f"{cond:<13}{r['n']:>6}   "
              f"{r['accuracy']:.3f} [{lo:.3f}, {hi:.3f}]".ljust(36)
              + f"{r['macro_f1']:>9.3f}{r['invalid_rate']:>9.1%}")

    summary = {
        "model": args.model,
        "ollama_version": ollama_version(),
        "run_date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset_file": path,
        "dataset_sha256": sha,
        "rows_used": len(df),
        "per_class_sample": args.per_class or "all",
        "seed": args.seed,
        "temperature": 0,
        "prompts": {c: PROMPTS[c] for c in args.conditions},
        "results": results,
    }
    safe = args.model.replace(":", "-").replace("/", "-")
    out = os.path.join(args.outdir, f"summary_{safe}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out}")
    print("Send me this JSON file plus the *_predictions.csv files.")


if __name__ == "__main__":
    main()
