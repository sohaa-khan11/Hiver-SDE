"""
retrieve_similar.py
-------------------
Historical Reply Retrieval for the Hiver Customer Support Assistant.

Goal:
  Given a new customer message, find the top 5 most similar historical AppleSupport
  customer messages from Dataset/processed/apple_cleaned.csv using TF-IDF vectorization
  and Cosine Similarity. Return the historical customer inquiries along with the
  actual AppleSupport replies from those conversations as evidence.

Data Governance Rule:
  golden_set.csv is held out strictly for evaluation. All Golden Set conversation IDs
  are excluded from the retrieval corpus to ensure zero data leakage.

Usage:
  python src/retrieval/retrieve_similar.py
"""

import os
import sys
import csv
from typing import List, Dict, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure UTF-8 output on Windows consoles to cleanly display emojis in customer tweets
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_excluded_golden_ids(golden_path: str) -> Set[str]:
    """
    Load conversation IDs from the Golden Set to ensure complete data separation.
    The Golden Set is strictly for evaluation and must not be part of the retrieval corpus.
    """
    if not os.path.exists(golden_path):
        return set()
    df = pd.read_csv(golden_path, encoding="utf-8-sig")
    return set(df["conversation_id"].astype(str).unique())


def load_historical_conversations(cleaned_path: str, exclude_ids: Set[str]) -> List[Dict]:
    """
    Load historical conversations from apple_cleaned.csv.
    Groups tweets by conversation_id to pair each initial customer inquiry
    with its corresponding AppleSupport reply/replies.
    """
    if not os.path.exists(cleaned_path):
        raise FileNotFoundError(f"Cleaned dataset not found at: {cleaned_path}")

    # Read rows in chronological order
    with open(cleaned_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    # Group by conversation_id
    threads: Dict[str, Dict] = {}
    for row in all_rows:
        cid = row.get("conversation_id", "").strip()
        if not cid or cid in exclude_ids:
            continue

        if cid not in threads:
            threads[cid] = {
                "conversation_id": cid,
                "customer_message": None,
                "support_replies": []
            }

        turn_index = row.get("turn_index", "").strip()
        is_inbound = row.get("inbound", "").strip().lower() == "true"
        text_clean = row.get("text_clean", "").strip()

        # Turn 0 inbound is the root customer inquiry
        if turn_index == "0" and is_inbound and not threads[cid]["customer_message"]:
            threads[cid]["customer_message"] = text_clean
        elif not is_inbound:
            # Outbound tweet from AppleSupport
            if text_clean:
                threads[cid]["support_replies"].append(text_clean)

    # Filter for complete threads that have both customer inquiry and at least one support reply
    corpus = [
        t for t in threads.values()
        if t["customer_message"] and len(t["support_replies"]) > 0
    ]

    return corpus


def build_retrieval_index(corpus: List[Dict]) -> Tuple[TfidfVectorizer, np.ndarray]:
    """
    Fit a TF-IDF vectorizer over the customer inquiry messages.
    - ngram_range=(1, 2): Captures both keywords ('airpods', 'battery') and phrases ('keep disconnecting', 'draining fast').
    - min_df=2: Removes rare typos and noise.
    - sublinear_tf=True: Dampens repetitive words using 1 + log(tf).
    - stop_words='english': Strips conversational filler so technical terms drive similarity.
    """
    customer_texts = [item["customer_message"] for item in corpus]
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        stop_words="english"
    )
    tfidf_matrix = vectorizer.fit_transform(customer_texts)
    return vectorizer, tfidf_matrix


def retrieve_similar(
    query: str,
    vectorizer: TfidfVectorizer,
    tfidf_matrix: np.ndarray,
    corpus: List[Dict],
    top_k: int = 5
) -> List[Dict]:
    """
    Given a new customer message, calculate cosine similarity against all historical
    customer messages and return the top_k most similar cases with their support replies.
    """
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix)[0]

    # Rank by descending similarity
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = float(similarities[idx])
        item = corpus[idx]
        results.append({
            "conversation_id": item["conversation_id"],
            "customer_message": item["customer_message"],
            "support_replies": item["support_replies"],
            "first_reply": item["support_replies"][0],
            "similarity_score": round(score, 4)
        })

    return results


def print_results(query: str, results: List[Dict]):
    """Print retrieved results in a clear, human-readable format."""
    print("\n" + "=" * 80)
    print(f"QUERY: \"{query}\"")
    print("=" * 80)

    for rank, res in enumerate(results, start=1):
        print(f"\n--- Match #{rank} (Similarity Score: {res['similarity_score']:.4f} | Conversation ID: {res['conversation_id']}) ---")
        print(f"Customer Inquiry: \"{res['customer_message']}\"")
        if len(res["support_replies"]) == 1:
            print(f"AppleSupport Reply: \"{res['first_reply']}\"")
        else:
            print(f"AppleSupport Reply (Turn 1): \"{res['first_reply']}\"")
            for reply_idx, extra_reply in enumerate(res["support_replies"][1:], start=2):
                print(f"AppleSupport Reply (Turn {reply_idx}): \"{extra_reply}\"")


def main():
    cleaned_path = os.path.join("Dataset", "processed", "apple_cleaned.csv")
    golden_path = os.path.join("Dataset", "processed", "golden_set.csv")

    print("=" * 80)
    print("Historical Reply Retrieval: AppleSupport Corpus Indexing")
    print("=" * 80)

    # 1. Load Golden Set IDs to exclude from retrieval index (Zero Data Leakage)
    exclude_ids = load_excluded_golden_ids(golden_path)
    print(f"Excluded {len(exclude_ids)} Golden Set conversation IDs to prevent leakage.")

    # 2. Load historical conversation corpus
    corpus = load_historical_conversations(cleaned_path, exclude_ids)
    print(f"Indexed {len(corpus):,} historical AppleSupport conversations.")

    # 3. Build TF-IDF representation
    vectorizer, tfidf_matrix = build_retrieval_index(corpus)
    vocab_size = len(vectorizer.vocabulary_)
    print(f"Fitted TF-IDF index with {vocab_size:,} vocabulary features.")
    print(f"Index matrix shape: {tfidf_matrix.shape}")

    # 4. Test with realistic customer inquiries
    test_queries = [
        "My AirPods keep disconnecting from Bluetooth",
        "My iPhone battery is draining really fast",
        "I forgot my Apple ID password"
    ]

    print("\nRunning similarity retrieval on test inquiries...")
    for query in test_queries:
        results = retrieve_similar(query, vectorizer, tfidf_matrix, corpus, top_k=5)
        print_results(query, results)

    print("\n" + "=" * 80)
    print("Retrieval Phase Verification Complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
