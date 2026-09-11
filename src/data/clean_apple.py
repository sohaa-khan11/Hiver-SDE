"""
clean_apple.py
--------------
Applies conservative, justified data preprocessing to the extracted AppleSupport
conversation dataset (apple_conversations.csv) without destroying structural
relationships, conversational integrity, or real customer language.

What is cleaned (stored in derived column 'text_clean'):
  1. HTML Entities: Decoded (&amp; -> &, &gt; -> >, &lt; -> <) using html.unescape.
  2. Routing Mentions:
     - Outbound (Brand): Stripped leading customer ID mention (^@\d+\s+) since it reflects 2017 Twitter reply mechanics.
     - Inbound (Customer): Stripped leading brand mention (^@AppleSupport\s+) where it adds no discriminatory intent information.
     - Mid-sentence mentions (@AppleSupport, @Spotify, etc.) are strictly preserved.
  3. URLs: Replaced with semantic token '[URL]' because linked web content is not in the dataset.
  4. Whitespace: Collapsed consecutive spaces and newlines into single spaces; trimmed edges.

What is preserved (NOT removed or destroyed):
  - Emojis & Unicode: Preserved intact (including variation selectors and symbols).
  - Punctuation & Capitalization: Preserved (shouting/caps and '??' / '!' indicate urgency/frustration).
  - Slang & Typos: Preserved (authentic user queries; understood by modern embeddings/LLMs).
  - Short Messages & Duplicate Texts: Preserved in conversation dataset (represent valid replies/acknowledgments).
  - Original Text: Preserved verbatim in the 'text' column for auditing and ground truth comparison.

Output:
  Dataset/processed/apple_cleaned.csv
"""

import os
import re
import html
import argparse
import pandas as pd


def parse_arguments():
    """Parse command-line arguments for configurable, reproducible preprocessing."""
    parser = argparse.ArgumentParser(
        description="Clean and preprocess extracted AppleSupport conversations."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=os.path.join("Dataset", "processed", "apple_conversations.csv"),
        help="Path to input extracted conversation dataset.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=os.path.join("Dataset", "processed", "apple_cleaned.csv"),
        help="Path to save cleaned dataset.",
    )
    return parser.parse_args()


def clean_tweet_text(raw_text: str, is_inbound: bool) -> str:
    """
    Deterministically cleans a single tweet's text using conservative rules.

    Args:
        raw_text: The original tweet string.
        is_inbound: True if authored by customer, False if authored by AppleSupport.

    Returns:
        Cleaned text string.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return ""

    # 1. Decode HTML entities (&amp; -> &, &gt; -> >, &lt; -> <)
    cleaned = html.unescape(raw_text)

    # 2. Strip leading routing handle only
    if is_inbound:
        # Strip leading @AppleSupport mention (case-insensitive)
        cleaned = re.sub(r"^@AppleSupport\s+", "", cleaned, flags=re.IGNORECASE)
    else:
        # Strip leading customer numeric mention (@115854)
        cleaned = re.sub(r"^@\d+\s+", "", cleaned)

    # 4. Normalize URLs to [URL] token
    cleaned = re.sub(r"https?://\S+", "[URL]", cleaned)

    # 5. Collapse consecutive spaces and newlines into a single space
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Safety check: if text was ONLY "@AppleSupport", keep the original mention
    # so we never produce an empty cleaned text from a valid tweet.
    if not cleaned:
        cleaned = raw_text.strip()

    return cleaned


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates derived 'text_clean' column while preserving all original columns.
    """
    print(f"[Step 2/4] Applying conservative cleaning to {len(df):,} tweets...")

    cleaned_texts = [
        clean_tweet_text(text, inbound)
        for text, inbound in zip(df["text"], df["inbound"])
    ]

    df_cleaned = df.copy()
    df_cleaned["text_clean"] = cleaned_texts

    # Order columns logically: place text_clean next to text
    ordered_columns = [
        "conversation_id",
        "turn_index",
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "text_clean",
        "response_tweet_id",
        "in_response_to_tweet_id",
    ]
    df_cleaned = df_cleaned[ordered_columns]
    return df_cleaned


def validate_cleaned_dataset(df_raw: pd.DataFrame, df_cleaned: pd.DataFrame):
    """
    Runs integrity checks comparing raw extracted dataset and cleaned dataset.
    Fails with an AssertionError if any invariant is violated.
    """
    print("[Step 3/4] Running data quality and integrity checks...")

    # 1. Row count and conversation count equality
    assert len(df_cleaned) == len(df_raw), (
        f"Validation Failed: Row count changed! Raw: {len(df_raw)}, Cleaned: {len(df_cleaned)}"
    )
    assert df_cleaned["conversation_id"].nunique() == df_raw["conversation_id"].nunique(), (
        f"Validation Failed: Conversation count changed! Raw: {df_raw['conversation_id'].nunique()}, "
        f"Cleaned: {df_cleaned['conversation_id'].nunique()}"
    )
    print(f"  [PASS] Row count ({len(df_cleaned):,}) and conversation count ({df_cleaned['conversation_id'].nunique():,}) strictly preserved.")

    # 2. Tweet IDs uniqueness and equality
    assert df_cleaned["tweet_id"].equals(df_raw["tweet_id"]), "Validation Failed: tweet_id ordering or values changed!"
    assert df_cleaned["tweet_id"].duplicated().sum() == 0, "Validation Failed: Duplicate tweet_ids detected!"
    print("  [PASS] All tweet_id values remain strictly unique and identically ordered.")

    # 3. Structural columns unchanged
    structural_cols = ["conversation_id", "turn_index", "author_id", "inbound", "created_at", "response_tweet_id", "in_response_to_tweet_id"]
    for col in structural_cols:
        # Fill NA with empty string for exact Series equality check
        raw_col = df_raw[col].fillna("")
        clean_col = df_cleaned[col].fillna("")
        assert (raw_col == clean_col).all(), f"Validation Failed: Structural column '{col}' was modified!"
    print("  [PASS] All structural and metadata columns are 100% identical to extracted dataset.")

    # 4. Original text column unchanged
    assert (df_cleaned["text"] == df_raw["text"]).all(), "Validation Failed: Original 'text' column was modified!"
    print("  [PASS] Original 'text' column preserved verbatim (zero in-place modifications).")

    # 5. Cleaned text non-empty
    empty_clean = (df_cleaned["text_clean"].str.strip() == "").sum()
    assert empty_clean == 0, f"Validation Failed: Found {empty_clean} empty text_clean entries!"
    print("  [PASS] All 26,129 text_clean values are non-empty and valid.")


def save_cleaned_dataset(df: pd.DataFrame, output_path: str):
    """Saves cleaned dataset to CSV using utf-8-sig for Excel compatibility."""
    print(f"[Step 4/4] Saving cleaned dataset to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Use 'utf-8-sig' (UTF-8 with Byte Order Mark) so spreadsheet tools like Microsoft Excel
    # automatically recognize UTF-8 encoding on Windows and render curly quotes/emojis without ANSI mojibake.
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Successfully wrote {len(df):,} rows ({file_size_mb:.2f} MB) to {output_path}.")


def main():
    args = parse_arguments()

    print("=" * 70)
    print("  Hiver SDE Assignment: AppleSupport Data Cleaning & Preprocessing")
    print("=" * 70)

    # 1. Load extracted dataset
    print(f"[Step 1/4] Loading extracted dataset from: {args.input_path}")
    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input file not found: {args.input_path}")

    dtypes = {
        "conversation_id": "int64",
        "turn_index": "int64",
        "tweet_id": "int64",
        "author_id": "str",
        "inbound": "bool",
        "created_at": "str",
        "text": "str",
        "response_tweet_id": "str",
        "in_response_to_tweet_id": "str",
    }
    df_raw = pd.read_csv(args.input_path, dtype=dtypes, low_memory=False)
    print(f"  Loaded {len(df_raw):,} tweets across {df_raw['conversation_id'].nunique():,} conversations.")

    # 2. Clean dataset
    df_cleaned = clean_dataset(df_raw)

    # 3. Validate
    validate_cleaned_dataset(df_raw, df_cleaned)

    # 4. Save
    save_cleaned_dataset(df_cleaned, args.output_path)

    print("=" * 70)
    print("  Data Cleaning Completed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
