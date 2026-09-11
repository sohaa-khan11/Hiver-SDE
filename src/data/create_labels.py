"""
create_labels.py
----------------
Samples a balanced, human-reviewable candidate annotation dataset (~250-300 conversations)
from Dataset/processed/apple_cleaned.csv for Golden Set labeling.

Sampling Methodology:
  1. Filters conversation roots (turn_index == 0) to ensure conversation-level sampling.
  2. Stratified Candidate Pools:
     - 7 core technical problem areas (~30 conversations each = 210):
       Battery, Phone Performance, Keyboard, Apple ID, Sound & Bluetooth, Screen & Camera, Apps & Storage.
     - Multi-category / Borderline cases (~35 conversations):
       Conversations exhibiting symptoms across multiple domains (e.g., battery drain after update, screen freeze).
     - Ambiguous / General Complaint / 'Other' cases (~35 conversations):
       Non-keyword tweets, emotional rants, and vague queries to test the 'Other' boundary.
  3. Total target: 280 unique conversations.
  4. Formats early conversation context (up to first 3 turns) to aid human annotators without dumping long threads.
  5. The 'intent' column is left intentionally blank for manual human annotation.

Usage:
  python src/data/create_labels.py
"""

import os
import re
import argparse
import pandas as pd


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Sample candidate conversations for human intent annotation."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=os.path.join("Dataset", "processed", "apple_cleaned.csv"),
        help="Path to cleaned conversations CSV.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=os.path.join("Dataset", "processed", "label_candidates.csv"),
        help="Path to save candidate annotation CSV.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


def build_early_context(df_conv: pd.DataFrame, max_turns: int = 3) -> str:
    """
    Constructs a readable, compact summary of early conversation turns (up to max_turns).
    Gives annotators enough context to understand the problem without dumping 20+ turns.
    """
    sorted_turns = df_conv.sort_values("turn_index").head(max_turns)
    lines = []
    for _, row in sorted_turns.iterrows():
        speaker = "Customer" if row["inbound"] else "AppleSupport"
        text = str(row["text_clean"]).strip()
        lines.append(f"[{speaker}]: {text}")
    return "\n".join(lines)


def create_label_candidates(
    df: pd.DataFrame,
    target_per_category: int = 30,
    borderline_target: int = 35,
    other_target: int = 35,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Selects a balanced, diverse pool of candidate conversations across all 8 support categories,
    including edge cases and ambiguous queries.
    """
    # Filter to initial customer requests (roots)
    roots = df[(df["turn_index"] == 0) & (df["inbound"] == True)].copy()

    # Drop duplicate customer messages to ensure variety
    roots = roots.drop_duplicates(subset=["text_clean"]).copy()

    # Broad candidate filters based on problem area vocabulary
    patterns = {
        "Battery": re.compile(
            r"\b(?:battery|drain|draining|charge|charging|charger|percentage|dies|dying)\b", re.I
        ),
        "Phone Performance": re.compile(
            r"\b(?:freeze|freezing|frozen|crashed|crashing|crash|reboot|rebooting|restart|boot\s*loop|lag|lagging|slow|sluggish|stuck)\b", re.I
        ),
        "Keyboard": re.compile(
            r"\b(?:keyboard|autocorrect|auto\s*correct|typing|letter\s*i|predictive|boxes|question\s*mark|text\s*replacement)\b", re.I
        ),
        "Apple ID": re.compile(
            r"\b(?:apple\s*id|icloud|account|password|passcode|locked|unlock|verification|sign\s*in|login|log\s*in)\b", re.I
        ),
        "Sound & Bluetooth": re.compile(
            r"\b(?:airpod|airpods|headphone|headphones|earphones|bluetooth|sound|audio|speaker|microphone|mic|volume)\b", re.I
        ),
        "Screen & Camera": re.compile(
            r"\b(?:screen|display|touch|unresponsive|camera|lens|home\s*button|cracked|glass|flash)\b", re.I
        ),
        "Apps & Storage": re.compile(
            r"\b(?:app\s*store|itunes|storage|space|full|download.*apps?|deleting.*apps?|podcast|safari)\b", re.I
        ),
    }

    # Evaluate matches per category
    match_mask = {}
    for cat, pat in patterns.items():
        match_mask[cat] = roots["text_clean"].str.contains(pat, na=False)

    match_df = pd.DataFrame(match_mask, index=roots.index)
    match_count = match_df.sum(axis=1)

    selected_cids = []
    category_estimates = {}

    # 1. Sample primary technical categories (pure or dominant matches)
    for cat in patterns.keys():
        # Prefer candidates that match this category
        cat_candidates = roots[match_df[cat] & (match_count == 1)]
        if len(cat_candidates) < target_per_category:
            # Fall back to any match if pure pool is smaller
            cat_candidates = roots[match_df[cat]]

        sample_n = min(target_per_category, len(cat_candidates))
        sampled = cat_candidates.sample(n=sample_n, random_state=seed)
        selected_cids.extend(sampled["conversation_id"].tolist())
        category_estimates[cat] = sample_n

    # 2. Sample Borderline / Multi-Category Overlap candidates
    overlap_candidates = roots[match_count >= 2]
    overlap_candidates = overlap_candidates[~overlap_candidates["conversation_id"].isin(selected_cids)]
    sample_overlap_n = min(borderline_target, len(overlap_candidates))
    sampled_overlap = overlap_candidates.sample(n=sample_overlap_n, random_state=seed)
    selected_cids.extend(sampled_overlap["conversation_id"].tolist())
    category_estimates["Multi-Category / Borderline"] = sample_overlap_n

    # 3. Sample Ambiguous / General / 'Other' candidates
    other_candidates = roots[match_count == 0]
    other_candidates = other_candidates[~other_candidates["conversation_id"].isin(selected_cids)]
    sample_other_n = min(other_target, len(other_candidates))
    sampled_other = other_candidates.sample(n=sample_other_n, random_state=seed)
    selected_cids.extend(sampled_other["conversation_id"].tolist())
    category_estimates["Other / Ambiguous"] = sample_other_n

    # Ensure no duplicates
    selected_cids = list(dict.fromkeys(selected_cids))

    # Build final candidate dataset
    selected_roots = roots[roots["conversation_id"].isin(selected_cids)].copy()
    # Preserve sampling order or shuffle deterministically
    selected_roots = selected_roots.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    records = []
    for _, row in selected_roots.iterrows():
        cid = row["conversation_id"]
        cust_msg = row["text_clean"]

        # Gather conversation turns for context
        df_conv = df[df["conversation_id"] == cid]
        context_str = build_early_context(df_conv, max_turns=3)

        records.append({
            "conversation_id": cid,
            "customer_message": cust_msg,
            "context": context_str,
            "intent": "",  # Intentionally blank for human manual annotation
        })

    candidate_df = pd.DataFrame(records)
    return candidate_df, category_estimates


def main():
    args = parse_arguments()

    print("=" * 70)
    print("  Sampling Candidate Conversations for Golden Set Annotation")
    print("=" * 70)

    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input dataset not found at: {args.input_path}")

    print(f"Reading cleaned dataset from: {args.input_path}")
    df = pd.read_csv(args.input_path, dtype={
        "conversation_id": "int64",
        "turn_index": "int64",
        "tweet_id": "int64",
        "author_id": "str",
        "inbound": "bool",
        "text": "str",
        "text_clean": "str",
    })

    candidate_df, category_estimates = create_label_candidates(
        df=df,
        target_per_category=30,
        borderline_target=35,
        other_target=35,
        seed=args.seed,
    )

    # Save to CSV with utf-8-sig for Excel / human review compatibility
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    candidate_df.to_csv(args.output_path, index=False, encoding="utf-8-sig")

    print(f"\nSuccessfully sampled {len(candidate_df)} candidate conversations.")
    print(f"Saved candidate annotation file to: {args.output_path}")

    print("\n--- Estimated Distribution in Candidate Pool ---")
    for group, count in category_estimates.items():
        print(f"  {group:<35} : {count:>3} candidates")
    print(f"  {'Total Candidate Set':<35} : {len(candidate_df):>3} candidates")
    print("  'intent' column is left blank for human review and manual labeling.")
    print("=" * 70)


if __name__ == "__main__":
    main()
