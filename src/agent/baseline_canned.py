"""
baseline_canned.py
------------------
Baseline 1: Intent + Generic Canned Reply Generation.

Generates responses for the 214 Golden Set customer messages using only the predicted
intent from the baseline classifier, mapping each intent to a fixed, generic canned template.
Does NOT use historical retrieval.
Does NOT use an LLM.

Usage:
  python src/agent/baseline_canned.py
"""

import os
import sys
import csv
from typing import Dict, List
import pandas as pd
import numpy as np

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


CANNED_RESPONSES: Dict[str, str] = {
    "Battery": (
        "Thanks for reaching out. For battery performance issues, please check Settings > Battery "
        "to review app battery usage, ensure your device is running the latest iOS update, "
        "and consider enabling Low Power Mode."
    ),
    "Phone Performance": (
        "We'd like to help with your device performance. Please try restarting your device, "
        "closing unused background applications, and ensuring you have sufficient available storage."
    ),
    "Keyboard": (
        "Thanks for contacting Apple Support. If you are experiencing keyboard or typing issues, "
        "please check Settings > General > Keyboard and try resetting your keyboard dictionary."
    ),
    "Apple ID": (
        "Thanks for reaching out regarding your account. For Apple ID and password assistance, "
        "please visit appleid.apple.com to verify your account security settings or initiate recovery."
    ),
    "Sound & Bluetooth": (
        "We're here to help with your audio and connectivity concerns. Please try toggling Bluetooth off and on, "
        "unpairing and repairing your accessory, or restarting your device."
    ),
    "Screen & Camera": (
        "Thanks for reaching out. For screen or camera issues, please restart your device, clean the display/lens, "
        "and verify your display settings under Settings > Display & Brightness."
    ),
    "Apps & Storage": (
        "Thanks for contacting us. For app or storage issues, please check Settings > General > iPhone Storage, "
        "make sure your apps are updated in the App Store, and restart your device."
    ),
    "Other": (
        "Thanks for reaching out to Apple Support. We'd like to learn more about what you're experiencing. "
        "Please provide additional details so we can assist you."
    ),
}


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
    preds_path = os.path.join("Dataset", "processed", "golden_set_predictions.csv")
    output_path = os.path.join("Dataset", "processed", "baseline_canned_predictions.csv")

    if not os.path.exists(preds_path):
        raise FileNotFoundError(f"Predictions file not found at: {preds_path}")

    print("=" * 80)
    print("Baseline 1: Intent + Generic Canned Reply Generation")
    print("=" * 80)

    df = pd.read_csv(preds_path, encoding="utf-8-sig")
    total_examples = len(df)
    print(f"Loaded {total_examples} Golden Set examples from: {preds_path}")

    results: List[Dict] = []
    jaccard_scores: List[float] = []

    for _, row in df.iterrows():
        cid = str(row["conversation_id"])
        msg = str(row["customer_message"])
        actual_intent = str(row["actual_intent"])
        predicted_intent = str(row["predicted_intent"])

        # Select canned reply based strictly on predicted intent
        reply = CANNED_RESPONSES.get(
            predicted_intent,
            CANNED_RESPONSES["Other"]
        )

        words = len(reply.split())
        chars = len(reply)
        jaccard = compute_token_jaccard(msg, reply)
        jaccard_scores.append(jaccard)

        results.append({
            "conversation_id": cid,
            "customer_message": msg,
            "actual_intent": actual_intent,
            "predicted_intent": predicted_intent,
            "intent_is_correct": actual_intent == predicted_intent,
            "generated_reply": reply,
            "reply_word_count": words,
            "reply_char_count": chars,
            "jaccard_similarity": round(jaccard, 4)
        })

    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved canned baseline predictions to: {output_path}")

    # Summary Statistics
    word_counts = out_df["reply_word_count"]
    char_counts = out_df["reply_char_count"]

    print("\n--- Evaluation Summary ---")
    print(f"Examples Evaluated:             {total_examples}")
    print(f"Response Available Rate:        100.0% ({total_examples}/{total_examples})")
    print(f"Mean Word Count:                {word_counts.mean():.1f} words (min: {word_counts.min()}, max: {word_counts.max()})")
    print(f"Mean Char Count:                {char_counts.mean():.1f} chars (min: {char_counts.min()}, max: {char_counts.max()})")
    print(f"Mean Token Jaccard Overlap:     {np.mean(jaccard_scores):.4f}")
    print(f"Underlying Intent Accuracy:     {(out_df['intent_is_correct'].sum() / total_examples) * 100:.2f}%")

    print("\n--- Representative Examples ---")
    sample_indices = [0, 1, 3, 5]
    for idx in sample_indices:
        r = results[idx]
        print(f"\n[Example {idx + 1}] Conversation ID: {r['conversation_id']}")
        print(f"  Customer Message:  \"{r['customer_message']}\"")
        print(f"  Actual Intent:     [{r['actual_intent']}]")
        print(f"  Predicted Intent:  [{r['predicted_intent']}] (Correct: {r['intent_is_correct']})")
        print(f"  Canned Response:   \"{r['generated_reply']}\"")

    print("\n" + "=" * 80)
    print("Baseline 1 Execution Complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
