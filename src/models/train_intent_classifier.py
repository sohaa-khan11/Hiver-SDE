"""
train_intent_classifier.py
--------------------------
Baseline Intent Classifier for the Hiver Customer Support Assistant.

Pipeline:
  TF-IDF Vectorizer + Multinomial Logistic Regression

Workflow:
  1. Load weakly-labeled training data (Dataset/processed/training_data.csv).
  2. Split into 80% train and 20% validation using stratified sampling (seed=42).
  3. Fit TF-IDF and Logistic Regression on the training split ONLY.
  4. Evaluate on the internal validation split (Accuracy, Macro F1, Per-class report).
  5. Evaluate on the completely untouched Golden Set (Dataset/processed/golden_set.csv).
  6. Print classification reports, confusion matrix, and save Golden Set predictions.

Data Governance Rule:
  golden_set.csv is held out strictly for evaluation. It is never used to fit TF-IDF,
  tune hyperparameters, or train the model.

Usage:
  python src/models/train_intent_classifier.py
"""

import os
import csv
from typing import List, Dict
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load a CSV dataset with UTF-8 encoding."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path, encoding="utf-8-sig")


def print_section(title: str):
    """Utility to print clean visual section dividers."""
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def print_confusion_matrix(cm, labels: List[str]):
    """Print an aligned, human-readable confusion matrix."""
    # Shorten labels for clean column formatting
    short_labels = [label[:10] for label in labels]
    title_col = "Actual \\ Pred"
    header = f"{title_col:<22} | " + " | ".join(f"{sl:>10}" for sl in short_labels)
    print(header)
    print("-" * len(header))
    for i, actual in enumerate(labels):
        row_counts = " | ".join(f"{cm[i][j]:>10}" for j in range(len(labels)))
        print(f"{actual:<22} | {row_counts}")


def main():
    train_path = os.path.join("Dataset", "processed", "training_data.csv")
    golden_path = os.path.join("Dataset", "processed", "golden_set.csv")
    preds_output_path = os.path.join("Dataset", "processed", "golden_set_predictions.csv")

    print_section("Step 1: Loading Training Data")
    train_df = load_dataset(train_path)
    print(f"Loaded training data: {len(train_df)} rows from {train_path}")

    X = train_df["customer_message"].astype(str)
    y = train_df["intent"].astype(str)

    # 80/20 Stratified Train/Validation Split
    # Ensures all 8 intent categories are proportionately represented in both splits
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training split size:   {len(X_train)} samples")
    print(f"Validation split size: {len(X_val)} samples")

    print_section("Step 2: Training Pipeline (TF-IDF + Logistic Regression)")
    # Model Choices & Rationale:
    # - ngram_range=(1, 2): Captures both single keywords ('battery', 'keyboard')
    #   and compound phrases ('spinning wheel', 'battery drain', 'sign in').
    # - min_df=2: Discards rare typos and one-off artifacts that appear only once.
    # - max_features=5000: Bounds feature dimension to prevent sparsity overfitting.
    # - sublinear_tf=True: Replaces raw term count with 1 + log(tf), damping repeated words.
    # - LogisticRegression(C=1.0, class_weight='balanced'): L2 regularization prevents
    #   overfitting on weak labels, while balanced weights adjust for minor class skew.
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=5000,
            sublinear_tf=True,
            stop_words="english"
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    print("Fitting pipeline on training split...")
    pipeline.fit(X_train, y_train)
    vocab_size = len(pipeline.named_steps["tfidf"].vocabulary_)
    print(f"TF-IDF Vocabulary Size: {vocab_size} n-grams")

    # -------------------------------------------------------------
    # Step 3: Evaluate on Internal Validation Split
    # -------------------------------------------------------------
    print_section("Step 3: Validation Split Results (20% Holdout from Training Data)")
    y_val_pred = pipeline.predict(X_val)

    val_acc = accuracy_score(y_val, y_val_pred)
    val_macro_f1 = f1_score(y_val, y_val_pred, average="macro", zero_division=0)

    print(f"Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"Validation Macro F1: {val_macro_f1 * 100:.2f}%\n")
    print("Validation Classification Report:")
    print(classification_report(y_val, y_val_pred, digits=3, zero_division=0))

    # -------------------------------------------------------------
    # Step 4: Evaluate on Untouched Golden Set
    # -------------------------------------------------------------
    print_section("Step 4: Golden Set Evaluation (Completely Untouched Ground Truth)")
    golden_df = load_dataset(golden_path)
    print(f"Loaded Golden Set: {len(golden_df)} rows from {golden_path}")

    X_gold = golden_df["customer_message"].astype(str)
    y_gold = golden_df["intent"].astype(str)

    y_gold_pred = pipeline.predict(X_gold)

    gold_acc = accuracy_score(y_gold, y_gold_pred)
    gold_macro_f1 = f1_score(y_gold, y_gold_pred, average="macro", zero_division=0)

    print(f"\nGolden Set Accuracy: {gold_acc * 100:.2f}%")
    print(f"Golden Set Macro F1: {gold_macro_f1 * 100:.2f}%\n")
    print("Golden Set Classification Report:")
    labels = sorted(list(y_gold.unique()))
    print(classification_report(y_gold, y_gold_pred, digits=3, labels=labels, zero_division=0))

    # Confusion Matrix
    print_section("Step 5: Golden Set Confusion Matrix")
    cm = confusion_matrix(y_gold, y_gold_pred, labels=labels)
    print_confusion_matrix(cm, labels)

    # -------------------------------------------------------------
    # Step 6: Save Golden Set Predictions
    # -------------------------------------------------------------
    golden_df_results = golden_df[["conversation_id", "customer_message", "intent"]].copy()
    golden_df_results.rename(columns={"intent": "actual_intent"}, inplace=True)
    golden_df_results["predicted_intent"] = y_gold_pred
    golden_df_results["is_correct"] = golden_df_results["actual_intent"] == golden_df_results["predicted_intent"]

    golden_df_results.to_csv(preds_output_path, index=False, encoding="utf-8-sig")
    print_section("Step 6: Output Saved")
    print(f"Saved Golden Set predictions to: {preds_output_path}")
    print(f"Correct predictions: {golden_df_results['is_correct'].sum()} / {len(golden_df_results)}")
    print(f"Error count:         {(~golden_df_results['is_correct']).sum()} / {len(golden_df_results)}")
    print("=" * 75)


if __name__ == "__main__":
    main()
