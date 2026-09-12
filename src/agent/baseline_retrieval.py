"""
baseline_retrieval.py
---------------------
Baseline 2: Historical Retrieval Only Reply Generation.

For each of the 214 Golden Set customer messages:
1. Retrieves the top-1 most similar historical conversation from apple_cleaned.csv
   using TF-IDF and Cosine Similarity (strictly excluding all Golden Set IDs).
2. Uses the historical human AppleSupport response directly as the generated reply.
3. Does NOT use an LLM to rewrite or adapt the response.

Usage:
  python src/agent/baseline_retrieval.py
"""

import os
import sys
import csv
from typing import List, Dict
import pandas as pd
import numpy as np

# Ensure project root is on sys.path for clean imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.retrieval.retrieve_similar import (
    load_excluded_golden_ids,
    load_historical_conversations,
    build_retrieval_index,
    retrieve_similar,
)

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def compute_token_jaccard(text1: str, text2: str) -> float:
    """Compute token-level Jaccard similarity between two texts."""
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union)


def main():
    cleaned_path = os.path.join("Dataset", "processed", "apple_cleaned.csv")
    golden_path = os.path.join("Dataset", "processed", "golden_set.csv")
    output_path = os.path.join("Dataset", "processed", "baseline_retrieval_predictions.csv")

    if not os.path.exists(golden_path):
        raise FileNotFoundError(f"Golden set not found at: {golden_path}")
    if not os.path.exists(cleaned_path):
        raise FileNotFoundError(f"Cleaned dataset not found at: {cleaned_path}")

    print("=" * 80)
    print("Baseline 2: Historical Retrieval Only Reply Generation")
    print("=" * 80)

    # 1. Load Golden Set IDs to ensure zero data leakage
    exclude_ids = load_excluded_golden_ids(golden_path)
    print(f"Excluded {len(exclude_ids)} Golden Set conversation IDs from retrieval corpus.")

    # 2. Index historical conversations from apple_cleaned.csv
    corpus = load_historical_conversations(cleaned_path, exclude_ids)
    print(f"Indexed {len(corpus):,} historical AppleSupport conversations.")

    # 3. Build TF-IDF retrieval index
    vectorizer, tfidf_matrix = build_retrieval_index(corpus)
    print(f"Fitted TF-IDF index with {len(vectorizer.vocabulary_):,} vocabulary features.")

    # 4. Load Golden Set examples
    golden_df = pd.read_csv(golden_path, encoding="utf-8-sig")
    total_examples = len(golden_df)
    print(f"Evaluating on all {total_examples} Golden Set examples...")

    results: List[Dict] = []
    similarities: List[float] = []
    jaccards: List[float] = []

    for _, row in golden_df.iterrows():
        cid = str(row["conversation_id"])
        msg = str(row["customer_message"])
        actual_intent = str(row["intent"])

        # Retrieve top-1 match
        top_matches = retrieve_similar(
            query=msg,
            vectorizer=vectorizer,
            tfidf_matrix=tfidf_matrix,
            corpus=corpus,
            top_k=1
        )

        if top_matches:
            top_res = top_matches[0]
            retrieved_cid = top_res["conversation_id"]
            sim_score = top_res["similarity_score"]
            retrieved_msg = top_res["customer_message"]
            retrieved_reply = top_res["first_reply"]
        else:
            retrieved_cid = ""
            sim_score = 0.0
            retrieved_msg = ""
            retrieved_reply = "We'd like to help with your issue. Please reach out to Apple Support for assistance."

        words = len(retrieved_reply.split())
        chars = len(retrieved_reply)
        jaccard = compute_token_jaccard(msg, retrieved_reply)

        similarities.append(sim_score)
        jaccards.append(jaccard)

        results.append({
            "conversation_id": cid,
            "customer_message": msg,
            "actual_intent": actual_intent,
            "retrieved_cid": retrieved_cid,
            "similarity_score": round(sim_score, 4),
            "retrieved_customer_message": retrieved_msg,
            "retrieved_reply": retrieved_reply,
            "reply_word_count": words,
            "reply_char_count": chars,
            "jaccard_similarity": round(jaccard, 4)
        })

    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved retrieval baseline predictions to: {output_path}")

    # Summary Statistics
    word_counts = out_df["reply_word_count"]
    char_counts = out_df["reply_char_count"]
    sim_scores = out_df["similarity_score"]
    has_match = (sim_scores > 0.0).sum()

    print("\n--- Evaluation Summary ---")
    print(f"Examples Evaluated:             {total_examples}")
    print(f"Retrieval Success Rate (Sim>0): {has_match}/{total_examples} ({(has_match / total_examples) * 100:.2f}%)")
    print(f"Mean Similarity Score:          {sim_scores.mean():.4f} (min: {sim_scores.min():.4f}, max: {sim_scores.max():.4f}, median: {sim_scores.median():.4f})")
    print(f"Mean Word Count:                {word_counts.mean():.1f} words (min: {word_counts.min()}, max: {word_counts.max()})")
    print(f"Mean Char Count:                {char_counts.mean():.1f} chars (min: {char_counts.min()}, max: {char_counts.max()})")
    print(f"Mean Token Jaccard Overlap:     {np.mean(jaccards):.4f}")

    print("\n--- Representative Examples ---")
    sample_indices = [0, 1, 3, 5]
    for idx in sample_indices:
        r = results[idx]
        print(f"\n[Example {idx + 1}] Golden CID: {r['conversation_id']} | Actual Intent: [{r['actual_intent']}]")
        print(f"  Golden Customer:    \"{r['customer_message']}\"")
        print(f"  Retrieved Match:    (CID: {r['retrieved_cid']} | Sim: {r['similarity_score']:.4f})")
        print(f"  Retrieved Customer: \"{r['retrieved_customer_message']}\"")
        print(f"  Retrieved Reply:    \"{r['retrieved_reply']}\"")

    print("\n" + "=" * 80)
    print("Baseline 2 Execution Complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
