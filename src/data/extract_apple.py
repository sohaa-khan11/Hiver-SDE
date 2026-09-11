"""
extract_apple.py
----------------
Extracts a reproducible, coherent working dataset of AppleSupport conversation
threads from the raw Twitter Customer Support (TWCS) dataset.

Pipeline:
  1. Load raw twitter_support.csv.
  2. Identify all AppleSupport outbound tweets.
  3. Trace conversations upward to identify true customer root tweets.
  4. Trace downward to collect all replies and follow-ups in each thread.
  5. Sample complete conversation threads using a fixed random seed.
  6. Order tweets chronologically within each conversation and assign turn_index.
  7. Run rigorous validation checks on conversation integrity.
  8. Save the extracted dataset to a processed CSV.

Usage:
  python src/data/extract_apple.py
  python src/data/extract_apple.py --sample-size 9000 --seed 42
"""

import os
import argparse
from collections import defaultdict
import pandas as pd
import numpy as np


def parse_arguments():
    """Parse command-line arguments for configurable, reproducible extraction."""
    parser = argparse.ArgumentParser(
        description="Extract a reproducible brand conversation dataset from TWCS."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=os.path.join("Dataset", "twcs", "twitter_support.csv"),
        help="Path to raw twitter_support.csv dataset.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=os.path.join("Dataset", "processed", "apple_conversations.csv"),
        help="Destination path for extracted conversation dataset.",
    )
    parser.add_argument(
        "--brand",
        type=str,
        default="AppleSupport",
        help="Brand account handle to extract (default: AppleSupport).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=9000,
        help="Number of conversation threads to sample (default: 9000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible conversation sampling (default: 42).",
    )
    return parser.parse_args()


def load_raw_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load raw TWCS dataset with explicit data types.
    Avoids dtype inference warnings and ensures tweet IDs remain 64-bit integers.
    """
    print(f"[Step 1/7] Loading raw dataset from: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    # Explicit string types for references because they may contain NaN or commas
    dtypes = {
        "tweet_id": "int64",
        "author_id": "str",
        "inbound": "bool",
        "created_at": "str",
        "text": "str",
        "response_tweet_id": "str",
        "in_response_to_tweet_id": "str",
    }

    df = pd.read_csv(csv_path, dtype=dtypes, low_memory=False)
    print(f"  Loaded {len(df):,} total tweets from raw dataset.")
    return df


def build_fast_lookups(df: pd.DataFrame):
    """
    Build dictionary lookups for fast graph traversal.
    Dictionary lookups in Python run in O(1) time, avoiding slow DataFrame filtering.
    """
    print("[Step 2/7] Building graph index for fast parent/child traversal...")
    tweet_author = dict(zip(df["tweet_id"], df["author_id"]))
    tweet_inbound = dict(zip(df["tweet_id"], df["inbound"]))
    tweet_parent = dict(zip(df["tweet_id"], df["in_response_to_tweet_id"]))
    tweet_responses = dict(zip(df["tweet_id"], df["response_tweet_id"]))
    return tweet_author, tweet_inbound, tweet_parent, tweet_responses


def reconstruct_conversations(df: pd.DataFrame, target_brand: str, tweet_author, tweet_inbound, tweet_parent, tweet_responses):
    """
    Reconstruct full conversation threads for the target brand:
      - Start from all outbound tweets sent by target_brand.
      - Traverse upward along parent links (in_response_to_tweet_id) to locate the root customer tweet.
      - Traverse downward along response links (response_tweet_id) to collect follow-ups.
      - Ensure every conversation starts with a customer root (inbound=True) and contains brand reply.
    """
    print(f"[Step 3/7] Reconstructing conversation threads for brand '{target_brand}'...")
    brand_tweets = df[df["author_id"] == target_brand]
    brand_tweet_ids = set(brand_tweets["tweet_id"])
    print(f"  Found {len(brand_tweet_ids):,} outbound tweets authored by {target_brand}.")

    root_to_thread = defaultdict(set)
    unresolved_chains = 0

    # Upward traversal: find root tweet for every brand reply
    for tid in brand_tweet_ids:
        curr = tid
        path = [curr]
        visited = {curr}

        while True:
            parent_raw = tweet_parent.get(curr)
            # Stop if reached a root (no parent)
            if pd.isna(parent_raw) or parent_raw is None or str(parent_raw).strip() in ("", "nan"):
                root_id = curr
                break

            try:
                pid = int(str(parent_raw).split(".")[0])
            except (ValueError, TypeError):
                root_id = None
                break

            if pid in visited:
                # Cycle detected
                root_id = None
                break

            if pid not in tweet_author:
                # Parent missing from dataset (dangling reference)
                root_id = None
                break

            visited.add(pid)
            path.append(pid)
            curr = pid

        if root_id is not None:
            # A valid conversation root must be from an inbound customer
            if tweet_inbound.get(root_id, False) is True:
                for p in path:
                    root_to_thread[root_id].add(p)
            else:
                unresolved_chains += 1
        else:
            unresolved_chains += 1

    print(f"  Located {len(root_to_thread):,} valid customer root conversations.")
    print(f"  Unresolved/dangling chains skipped: {unresolved_chains:,}")

    # Downward traversal: collect customer follow-ups and additional brand replies
    print("  Traversing downward to capture follow-up turns...")
    for root_id in list(root_to_thread.keys()):
        frontier = list(root_to_thread[root_id])
        visited = set(frontier)

        while frontier:
            curr = frontier.pop()
            resp_val = tweet_responses.get(curr)
            if pd.notna(resp_val) and resp_val:
                for child_str in str(resp_val).split(","):
                    try:
                        cid = int(child_str.strip().split(".")[0])
                    except (ValueError, TypeError):
                        continue

                    if cid in tweet_author and cid not in visited:
                        # Collect tweets that belong to the customer or the brand
                        c_auth = tweet_author[cid]
                        if c_auth == target_brand or tweet_inbound.get(cid, False) is True:
                            visited.add(cid)
                            root_to_thread[root_id].add(cid)
                            frontier.append(cid)

    # Filter threads: must have >= 2 tweets and include the brand
    valid_conversations = {}
    for root_id, thread_ids in root_to_thread.items():
        if len(thread_ids) >= 2:
            has_brand = any(tweet_author.get(tid) == target_brand for tid in thread_ids)
            if has_brand:
                valid_conversations[root_id] = thread_ids

    print(f"  Total verified, complete {target_brand} conversations: {len(valid_conversations):,}")
    return valid_conversations


def sample_conversations(valid_conversations: dict, sample_size: int, random_seed: int) -> dict:
    """
    Select a reproducible random sample of conversation threads.
    Sampling is performed at the conversation level (not tweet level) to preserve thread integrity.
    """
    print(f"[Step 4/7] Sampling conversations (target: {sample_size:,}, seed: {random_seed})...")
    all_root_ids = sorted(list(valid_conversations.keys()))

    if len(all_root_ids) <= sample_size:
        print(f"  Available conversations ({len(all_root_ids):,}) <= sample size. Using all available.")
        return valid_conversations

    # Use a fixed NumPy random generator for strict reproducibility across platforms
    rng = np.random.default_rng(seed=random_seed)
    selected_root_indices = rng.choice(len(all_root_ids), size=sample_size, replace=False)
    selected_root_ids = [all_root_ids[i] for i in selected_root_indices]

    sampled_conversations = {rid: valid_conversations[rid] for rid in selected_root_ids}
    print(f"  Successfully sampled {len(sampled_conversations):,} distinct conversation threads.")
    return sampled_conversations


def build_conversation_dataframe(df: pd.DataFrame, sampled_conversations: dict) -> pd.DataFrame:
    """
    Assemble the final working DataFrame:
      - Assign deterministic conversation_id (= root tweet_id).
      - Sort tweets chronologically by created_at within each conversation.
      - Assign turn_index (0 for root customer tweet, 1, 2, ... for replies).
      - Retain all original tweet-level attributes untouched.
    """
    print("[Step 5/7] Assembling conversation DataFrame and computing turn indices...")
    # Gather all selected tweet IDs
    all_selected_tids = set()
    for t_set in sampled_conversations.values():
        all_selected_tids.update(t_set)

    # Subset the original DataFrame
    sub_df = df[df["tweet_id"].isin(all_selected_tids)].copy()

    # Parse timestamps for chronological ordering
    # Twitter format: "Tue Oct 31 22:27:49 +0000 2017"
    sub_df["parsed_timestamp"] = pd.to_datetime(
        sub_df["created_at"],
        format="%a %b %d %H:%M:%S +0000 %Y",
        errors="coerce",
    )

    # Map each tweet_id to its conversation_id (root_id)
    tweet_to_conv_id = {}
    for root_id, t_set in sampled_conversations.items():
        for tid in t_set:
            tweet_to_conv_id[tid] = root_id

    sub_df["conversation_id"] = sub_df["tweet_id"].map(tweet_to_conv_id)

    # Sort by conversation_id, then chronologically by timestamp, then tweet_id
    sub_df = sub_df.sort_values(
        by=["conversation_id", "parsed_timestamp", "tweet_id"],
        ascending=[True, True, True]
    ).reset_index(drop=True)

    # Calculate turn_index within each conversation
    sub_df["turn_index"] = sub_df.groupby("conversation_id").cumcount()

    # Drop temporary parsed_timestamp column
    sub_df = sub_df.drop(columns=["parsed_timestamp"])

    # Define clean column order
    ordered_columns = [
        "conversation_id",
        "turn_index",
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id",
    ]
    sub_df = sub_df[ordered_columns]

    print(f"  Final working dataset contains {len(sub_df):,} tweets across {sub_df['conversation_id'].nunique():,} conversations.")
    return sub_df


def validate_extracted_dataset(df_extracted: pd.DataFrame, df_raw: pd.DataFrame):
    """
    Run data quality validation checks.
    Fails with an AssertionError if any structural invariant is violated.
    """
    print("[Step 6/7] Running data quality validation checks...")

    # 1. No duplicate tweet IDs
    dup_tweets = df_extracted["tweet_id"].duplicated().sum()
    assert dup_tweets == 0, f"Validation Failed: Found {dup_tweets} duplicate tweet IDs!"
    print("  [PASS] All tweet_id values are strictly unique.")

    # 2. No duplicate (conversation_id, turn_index) pairs
    dup_turns = df_extracted.duplicated(subset=["conversation_id", "turn_index"]).sum()
    assert dup_turns == 0, f"Validation Failed: Found {dup_turns} duplicate (conversation_id, turn_index) pairs!"
    print("  [PASS] All (conversation_id, turn_index) pairs are unique.")

    # 3. Turn indices start at 0 and are consecutive
    roots = df_extracted[df_extracted["turn_index"] == 0]
    total_convs = df_extracted["conversation_id"].nunique()
    assert len(roots) == total_convs, f"Validation Failed: Expected {total_convs} roots, found {len(roots)}!"
    print(f"  [PASS] Every conversation has exactly one root at turn_index=0 ({total_convs:,} roots).")

    # 4. Roots must be inbound (customer) and have conversation_id == tweet_id
    assert roots["inbound"].all(), "Validation Failed: Found root tweets that are not inbound customer tweets!"
    assert (roots["conversation_id"] == roots["tweet_id"]).all(), "Validation Failed: Root conversation_id != tweet_id!"
    print("  [PASS] All root tweets are customer messages (inbound=True) and match conversation_id.")

    # 5. Verify text was not modified (check random sample against raw data)
    sample_check = df_extracted.sample(n=min(500, len(df_extracted)), random_state=42)
    raw_lookup = dict(zip(df_raw["tweet_id"], df_raw["text"]))
    mismatches = 0
    for _, row in sample_check.iterrows():
        if row["text"] != raw_lookup[row["tweet_id"]]:
            mismatches += 1
    assert mismatches == 0, f"Validation Failed: {mismatches} text modifications detected!"
    print("  [PASS] Text content is 100% identical to the raw dataset (zero modifications).")

    # 6. Conversation length summary
    conv_lengths = df_extracted.groupby("conversation_id").size()
    assert (conv_lengths >= 2).all(), "Validation Failed: Found single-turn conversations!"
    print(f"  [PASS] All conversations are complete multi-turn threads (min length = {conv_lengths.min()}).")
    print(f"  Summary Statistics: Mean length = {conv_lengths.mean():.2f} turns, Median = {conv_lengths.median():.0f}, Max = {conv_lengths.max()} turns.")


def save_dataset(df: pd.DataFrame, output_path: str):
    """Save processed dataset to CSV."""
    print(f"[Step 7/7] Saving processed dataset to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Successfully wrote {len(df):,} rows ({file_size_mb:.2f} MB) to {output_path}.")


def main():
    args = parse_arguments()

    print("=" * 70)
    print("  Hiver SDE Assignment: AppleSupport Conversation Dataset Extraction")
    print("=" * 70)

    # 1. Load raw data
    df_raw = load_raw_dataset(args.input_path)

    # 2. Build graph indices
    tweet_author, tweet_inbound, tweet_parent, tweet_responses = build_fast_lookups(df_raw)

    # 3. Reconstruct threads
    valid_conversations = reconstruct_conversations(
        df_raw,
        target_brand=args.brand,
        tweet_author=tweet_author,
        tweet_inbound=tweet_inbound,
        tweet_parent=tweet_parent,
        tweet_responses=tweet_responses,
    )

    # 4. Sample threads
    sampled_conversations = sample_conversations(
        valid_conversations,
        sample_size=args.sample_size,
        random_seed=args.seed,
    )

    # 5. Build DataFrame
    df_extracted = build_conversation_dataframe(df_raw, sampled_conversations)

    # 6. Validate
    validate_extracted_dataset(df_extracted, df_raw)

    # 7. Save
    save_dataset(df_extracted, args.output_path)

    print("=" * 70)
    print("  Extraction Completed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
