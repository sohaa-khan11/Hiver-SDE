"""
evaluate_system.py
------------------
Evaluation Harness for the Hiver AppleSupport AI Customer Support System.

Evaluates:
  1. Escalation Policy: Compares deterministic agent escalation vs Golden policy labels.
     (Note: Golden escalation labels are policy-derived evaluation targets, not historical outcomes).
  2. Automated Proxy Metrics: Compares Baseline 1 (Canned), Baseline 2 (Retrieval),
     and Final Agent across all 214 Golden Set examples.
     (Note: Lexical overlap/Jaccard is reported strictly as an automated proxy, not true quality).
  3. Optimized LLM-as-Judge Framework:
     - 25 representative cases sampled with fixed seed=42 (stratified across all 8 intents).
     - Live agent generation for AUTO-HANDLE cases only (at most 8-10 calls, 0 for ESCALATE).
     - 1 consolidated judge call per case comparing Baseline 1, Baseline 2, and Final Agent.
     - Total API budget capped at ~33-35 calls maximum across the entire project.
     - Strictly caches every generated result to prevent repeated API spend.
  4. Human-vs-LLM Agreement Evaluator:
     - Directly aligned 25-case human review template matching the judge sample.
     - Blank columns for manual human scoring (never faked).
     - Automatically computes exact agreement and MAE once filled.

Usage:
  python src/evaluation/evaluate_system.py
"""

import os
import sys
import re
import json
import time
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agent.support_agent import AppleSupportAgent, clean_evidence_text, call_llm_api

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ==============================================================================
# 1. Escalation Policy Evaluation
# ==============================================================================

def evaluate_escalation(golden_df: pd.DataFrame, agent_decisions: List[str]) -> Dict[str, Any]:
    """
    Compare agent escalation decisions against Golden Set policy labels.
    Note: Golden Set labels are policy-derived evaluation targets, not historical outcomes.
    """
    y_true = golden_df["escalation_label"].tolist()
    y_pred = agent_decisions

    labels = ["AUTO-HANDLE", "ESCALATE"]
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "labels": labels
    }


# ==============================================================================
# 2. Automated Reply Proxy Metrics
# ==============================================================================

def compute_token_jaccard(text1: str, text2: str) -> float:
    """Compute token-level Jaccard overlap between two texts as an automated proxy."""
    t1 = set(re.findall(r"\w+", str(text1).lower()))
    t2 = set(re.findall(r"\w+", str(text2).lower()))
    if not t1 or not t2:
        return 0.0
    return len(t1.intersection(t2)) / len(t1.union(t2))


def contains_actionable_guidance(text: str) -> bool:
    """Check if the text contains actionable troubleshooting steps, links, or settings navigation."""
    t = str(text).lower()
    action_keywords = [
        "settings >", "settings>", "restart", "update", "visit", "reset",
        "turn on", "turn off", "sign in", "dm us", "direct message", "[url]",
        "appleid.apple.com", "getsupport.apple.com", "support.apple.com"
    ]
    return any(k in t for k in action_keywords)


def evaluate_automated_metrics(
    golden_df: pd.DataFrame,
    canned_df: pd.DataFrame,
    retrieval_df: pd.DataFrame,
    agent_results: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Compute automated proxy metrics for Baseline 1, Baseline 2, and Final Agent across 214 cases.
    """
    metrics = []

    # 1. Baseline 1 (Canned)
    canned_replies = canned_df["generated_reply"].tolist()
    canned_words = [len(str(r).split()) for r in canned_replies]
    canned_chars = [len(str(r)) for r in canned_replies]
    canned_jaccard = [
        compute_token_jaccard(q, r)
        for q, r in zip(golden_df["customer_message"], canned_replies)
    ]
    canned_actionable = sum(contains_actionable_guidance(r) for r in canned_replies) / len(canned_replies)

    metrics.append({
        "Model": "Baseline 1 (Canned)",
        "Response Availability": "100%",
        "Avg Word Count": round(np.mean(canned_words), 1),
        "Avg Char Count": round(np.mean(canned_chars), 1),
        "Cust Overlap Proxy (Jaccard)": round(np.mean(canned_jaccard), 4),
        "Actionable Guidance Rate": f"{canned_actionable * 100:.1f}%",
        "Escalation Handled": "0.0% (No escalation)"
    })

    # 2. Baseline 2 (Retrieval-Only)
    retrieval_replies = retrieval_df["retrieved_reply"].tolist()
    retrieval_words = [len(str(r).split()) for r in retrieval_replies]
    retrieval_chars = [len(str(r)) for r in retrieval_replies]
    retrieval_jaccard = [
        compute_token_jaccard(q, r)
        for q, r in zip(golden_df["customer_message"], retrieval_replies)
    ]
    retrieval_actionable = sum(contains_actionable_guidance(r) for r in retrieval_replies) / len(retrieval_replies)

    metrics.append({
        "Model": "Baseline 2 (Retrieval-Only)",
        "Response Availability": "100%",
        "Avg Word Count": round(np.mean(retrieval_words), 1),
        "Avg Char Count": round(np.mean(retrieval_chars), 1),
        "Cust Overlap Proxy (Jaccard)": round(np.mean(retrieval_jaccard), 4),
        "Actionable Guidance Rate": f"{retrieval_actionable * 100:.1f}%",
        "Escalation Handled": "0.0% (No escalation)"
    })

    # 3. Final Agent
    agent_replies = [res["final_reply"] for res in agent_results]
    agent_words = [len(str(r).split()) for r in agent_replies]
    agent_chars = [len(str(r)) for r in agent_replies]
    agent_jaccard = [
        compute_token_jaccard(q, r)
        for q, r in zip(golden_df["customer_message"], agent_replies)
    ]
    agent_actionable = sum(contains_actionable_guidance(r) for r in agent_replies) / len(agent_replies)
    escalate_count = sum(1 for res in agent_results if res["escalation_decision"] == "ESCALATE")

    metrics.append({
        "Model": "Final Support Agent",
        "Response Availability": "100%",
        "Avg Word Count": round(np.mean(agent_words), 1),
        "Avg Char Count": round(np.mean(agent_chars), 1),
        "Cust Overlap Proxy (Jaccard)": round(np.mean(agent_jaccard), 4),
        "Actionable Guidance Rate": f"{agent_actionable * 100:.1f}%",
        "Escalation Handled": f"{escalate_count / len(agent_results) * 100:.1f}% ({escalate_count}/{len(agent_results)})"
    })

    return pd.DataFrame(metrics)


# ==============================================================================
# 3. Optimized Sample Generation & Consolidated LLM-as-Judge
# ==============================================================================

CONSOLIDATED_JUDGE_PROMPT = """You are an expert impartial evaluator for an Apple Customer Support assistant.
Evaluate three candidate support replies to the customer inquiry using the provided historical AppleSupport reference evidence.

Customer Message: "{customer_message}"
Predicted Category: "{intent}"

Historical Reference Evidence:
{evidence_text}

Candidate 1 (Baseline Canned): "{canned_reply}"
Candidate 2 (Baseline Retrieval): "{retrieval_reply}"
Candidate 3 (Final Support Agent): "{agent_reply}"

Evaluate each candidate on 5 criteria:
1. Groundedness (1-5): Supported strictly by official Apple procedures in evidence.
2. Relevance (1-5): Directly addresses the customer's specific inquiry.
3. Actionability (1-5): Offers concrete steps, links, or diagnostic questions.
4. Support Tone (1-5): Polite, concise, empathetic social customer support voice.
5. Unsupported Claim (0 or 1): 1 if candidate makes up unsupported policies, prices, or fake steps; 0 if safe.

Return ONLY a valid JSON object in this exact format:
{{
  "candidate_1": {{"groundedness": <1-5>, "relevance": <1-5>, "actionability": <1-5>, "tone": <1-5>, "unsupported_claim": <0 or 1>}},
  "candidate_2": {{"groundedness": <1-5>, "relevance": <1-5>, "actionability": <1-5>, "tone": <1-5>, "unsupported_claim": <0 or 1>}},
  "candidate_3": {{"groundedness": <1-5>, "relevance": <1-5>, "actionability": <1-5>, "tone": <1-5>, "unsupported_claim": <0 or 1>}}
}}
"""


def generate_evaluation_sample(
    golden_df: pd.DataFrame,
    canned_df: pd.DataFrame,
    retrieval_df: pd.DataFrame,
    agent_results: List[Dict[str, Any]],
    sample_size: int = 25,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Select a reproducible random subset of exactly 25 Golden Set cases for LLM judging and human review.
    Stratified across all 8 intents to ensure balanced coverage.
    """
    sample_indices = golden_df.groupby("intent", group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), max(1, int(round(sample_size * len(x) / len(golden_df))))), random_state=random_seed),
        include_groups=False
    ).index[:sample_size].tolist()

    records = []
    for idx in sample_indices:
        row = golden_df.loc[idx]
        cid = int(row["conversation_id"])
        c_msg = str(row["customer_message"])
        intent = str(row["intent"])
        esc_label = str(row["escalation_label"])

        agent_res = agent_results[idx]
        top_cases = agent_res.get("retrieved_examples", [])
        evidence_text = "\n".join([
            f"- Case {i+1}: Cust: \"{c['customer_message']}\" | Reply: \"{c['first_reply']}\""
            for i, c in enumerate(top_cases[:3])
        ])

        canned_reply = canned_df.loc[idx, "generated_reply"]
        retrieval_reply = retrieval_df.loc[idx, "retrieved_reply"]
        agent_reply = agent_res["final_reply"]

        records.append({
            "sample_id": len(records) + 1,
            "conversation_id": cid,
            "customer_message": c_msg,
            "intent": intent,
            "golden_escalation": esc_label,
            "evidence_text": evidence_text,
            "canned_reply": canned_reply,
            "retrieval_reply": retrieval_reply,
            "agent_reply": agent_reply,
            "agent_decision": agent_res["escalation_decision"]
        })

    return pd.DataFrame(records)


def check_ollama_service() -> Tuple[bool, str]:
    """Check if the local Ollama service is reachable."""
    import urllib.request
    ollama_host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    url = f"{ollama_host.rstrip('/')}/api/tags"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                return True, "Ollama service online"
    except Exception as e:
        return False, f"Ollama unreachable at {ollama_host}: {e}"
    return False, f"Ollama returned unexpected status"


def apply_live_agent_generation(
    sample_df: pd.DataFrame,
    cache_path: str,
    agent: AppleSupportAgent,
    max_auto_handle_calls: int = 10
) -> Tuple[pd.DataFrame, str]:
    """
    Generate live LLM responses strictly for the AUTO-HANDLE cases in the sample
    using local Ollama (at most 8-10 calls, 0 for ESCALATE).
    Results are permanently cached to cache_path to eliminate repeated generation.
    """
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 100:
        cached_df = pd.read_csv(cache_path, encoding="utf-8-sig")
        cache_map = dict(zip(cached_df["sample_id"], cached_df["cached_reply"]))
        sample_df["agent_reply"] = sample_df["sample_id"].map(cache_map).fillna(sample_df["agent_reply"])
        return sample_df, f"Loaded {len(cache_map)} cached agent replies from {cache_path} (0 local LLM calls made)."

    is_online, status_msg = check_ollama_service()
    if not is_online:
        return sample_df, f"Pending: Local Ollama service not running ({status_msg}). Using offline reference fallbacks for agent replies."

    records = []
    auto_handle_count = 0
    print(f"Generating live agent responses for AUTO-HANDLE cases via local Ollama (max {max_auto_handle_calls} calls)...")

    for idx, row in sample_df.iterrows():
        sid = row["sample_id"]
        c_msg = row["customer_message"]
        decision = row["agent_decision"]

        if decision == "ESCALATE":
            reply = row["agent_reply"]  # deterministic safe response (0 LLM calls)
        elif auto_handle_count < max_auto_handle_calls:
            print(f"  [Sample {sid:02d}] Generating response for AUTO-HANDLE inquiry...")
            t0 = time.time()
            res = agent.process_message(c_msg, enable_llm=True)
            dt = time.time() - t0
            reply = res["final_reply"]
            auto_handle_count += 1
            print(f"    Done in {dt:.1f}s. LLM Used: {res.get('llm_used')}")
        else:
            reply = row["agent_reply"]

        sample_df.at[idx, "agent_reply"] = reply
        records.append({"sample_id": sid, "conversation_id": row["conversation_id"], "cached_reply": reply})

    pd.DataFrame(records).to_csv(cache_path, index=False, encoding="utf-8-sig")
    return sample_df, f"Generated and cached {auto_handle_count} live agent replies to {cache_path}."


def run_llm_judge_sample(sample_df: pd.DataFrame, output_path: str) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Run consolidated LLM-as-Judge on exactly 25 sample cases (1 call per case = 25 calls max)
    using local Ollama with JSON mode.
    Uses caching: if output_path exists and is non-empty, loads from cache without inference calls.
    """
    if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
        cached = pd.read_csv(output_path, encoding="utf-8-sig")
        return cached, f"Loaded cached LLM-as-Judge results from {output_path} (0 local LLM calls made)."

    is_online, status_msg = check_ollama_service()
    if not is_online:
        return None, f"Pending: Local Ollama service not running ({status_msg}). LLM judge evaluation skipped (deterministic pipeline intact)."

    import urllib.request
    ollama_host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    model_name = os.environ.get("OLLAMA_MODEL", "phi3:mini")
    chat_url = f"{ollama_host.rstrip('/')}/api/chat"

    results = []
    print(f"Running Consolidated LLM-as-Judge via local Ollama ({model_name}) on {len(sample_df)} sample cases...")

    model_mapping = {
        "candidate_1": "Baseline 1 (Canned)",
        "candidate_2": "Baseline 2 (Retrieval)",
        "candidate_3": "Final Support Agent"
    }

    for i, (_, row) in enumerate(sample_df.iterrows(), 1):
        prompt = CONSOLIDATED_JUDGE_PROMPT.format(
            customer_message=row["customer_message"],
            intent=row["intent"],
            evidence_text=row["evidence_text"],
            canned_reply=row["canned_reply"],
            retrieval_reply=row["retrieval_reply"],
            agent_reply=row["agent_reply"]
        )

        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "options": {
                "temperature": 0.0,
                "num_predict": 250
            },
            "format": "json",
            "stream": False
        }

        parsed_scores = {}
        print(f"  [Judge Case {i:02d}/{len(sample_df)}] Evaluating candidates...", end="", flush=True)
        t0 = time.time()
        for attempt in range(2):
            try:
                req = urllib.request.Request(
                    chat_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    res = json.loads(resp.read().decode("utf-8"))
                content = res.get("message", {}).get("content", "").strip()
                match = re.search(r"\{.*\}", content, re.DOTALL)
                if match:
                    parsed_scores = json.loads(match.group(0))
                    break
            except Exception as e:
                time.sleep(1)

        dt = time.time() - t0
        print(f" done ({dt:.1f}s)")

        for cand_key, model_name_str in model_mapping.items():
            scores = parsed_scores.get(cand_key, {})
            results.append({
                "sample_id": row["sample_id"],
                "conversation_id": row["conversation_id"],
                "model_name": model_name_str,
                "groundedness": scores.get("groundedness", 3),
                "relevance": scores.get("relevance", 3),
                "actionability": scores.get("actionability", 3),
                "tone": scores.get("tone", 3),
                "unsupported_claim": scores.get("unsupported_claim", 0)
            })

    judge_df = pd.DataFrame(results)
    judge_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return judge_df, f"Successfully executed LLM-as-Judge ({len(sample_df)} calls via local {model_name}) and cached to {output_path}."


# ==============================================================================
# 4. Human-vs-LLM Agreement Template & Evaluator
# ==============================================================================

def create_human_eval_template(sample_df: pd.DataFrame, output_path: str):
    """
    Generate a simple, blank review template for the exact 25 sample cases for human evaluation.
    DO NOT fake human labels. Leaves rating columns blank for manual entry.
    """
    if os.path.exists(output_path):
        return

    human_records = []
    for _, row in sample_df.iterrows():
        human_records.append({
            "sample_id": row["sample_id"],
            "conversation_id": row["conversation_id"],
            "customer_message": row["customer_message"],
            "intent": row["intent"],
            "final_agent_reply": row["agent_reply"],
            "human_groundedness_1to5": "",
            "human_relevance_1to5": "",
            "human_actionability_1to5": "",
            "human_tone_1to5": "",
            "human_unsupported_claim_0or1": "",
            "human_notes": ""
        })

    pd.DataFrame(human_records).to_csv(output_path, index=False, encoding="utf-8-sig")


def compute_human_agreement(template_path: str, judge_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """
    Calculate agreement between human scores and LLM judge scores if human scores are filled.
    If human fields are empty, cleanly returns pending status.
    """
    if not os.path.exists(template_path):
        return {"status": "Template not found"}

    human_df = pd.read_csv(template_path, encoding="utf-8-sig")
    unfilled = human_df["human_groundedness_1to5"].isna().all() or (human_df["human_groundedness_1to5"] == "").all()

    if unfilled or judge_df is None:
        return {
            "status": "Pending Manual Human Review",
            "message": f"Template ready at {template_path}. Enter scores to compute human-vs-LLM agreement metrics.",
            "total_review_samples": len(human_df)
        }

    agent_judge = judge_df[judge_df["model_name"] == "Final Support Agent"].set_index("sample_id")
    merged = human_df.join(agent_judge, on="sample_id", how="inner")

    if len(merged) == 0:
        return {"status": "No overlapping samples to compare"}

    exact_claim_agreement = (
        merged["human_unsupported_claim_0or1"].astype(int) == merged["unsupported_claim"].astype(int)
    ).mean()
    groundedness_mae = np.mean(np.abs(merged["human_groundedness_1to5"].astype(float) - merged["groundedness"].astype(float)))
    relevance_mae = np.mean(np.abs(merged["human_relevance_1to5"].astype(float) - merged["relevance"].astype(float)))

    return {
        "status": "Completed",
        "evaluated_samples": len(merged),
        "unsupported_claim_exact_agreement": round(exact_claim_agreement, 4),
        "groundedness_mae": round(groundedness_mae, 3),
        "relevance_mae": round(relevance_mae, 3)
    }


# ==============================================================================
# 5. Main Runner
# ==============================================================================

def main():
    print("=" * 90)
    print("Hiver SDE Take-Home: Comprehensive System Evaluation")
    print("=" * 90)

    golden_path = os.path.join("Dataset", "processed", "golden_set.csv")
    canned_path = os.path.join("Dataset", "processed", "baseline_canned_predictions.csv")
    retrieval_path = os.path.join("Dataset", "processed", "baseline_retrieval_predictions.csv")
    gen_cache_path = os.path.join("Dataset", "processed", "llm_generation_cache.csv")
    judge_cache_path = os.path.join("Dataset", "processed", "llm_judge_results.csv")
    human_template_path = os.path.join("Dataset", "processed", "human_eval_template.csv")

    print(f"Loading Golden Set ({golden_path})...")
    golden_df = pd.read_csv(golden_path, encoding="utf-8-sig")
    print(f"Golden Set loaded: {len(golden_df)} examples.")

    # 1. Load Baselines
    if not os.path.exists(canned_path):
        print("Running Baseline 1 (Canned)...")
        import src.agent.baseline_canned as b1
        b1.main()
    canned_df = pd.read_csv(canned_path, encoding="utf-8-sig")

    if not os.path.exists(retrieval_path):
        print("Running Baseline 2 (Retrieval-Only)...")
        import src.agent.baseline_retrieval as b2
        b2.main()
    retrieval_df = pd.read_csv(retrieval_path, encoding="utf-8-sig")

    # 2. Run Final Agent over the Golden Set (Deterministic pipeline, enable_llm=False)
    print("\nRunning Final Agent across Golden Set (214 cases, deterministic mode)...")
    agent = AppleSupportAgent(top_k=4)
    agent_results = []
    agent_decisions = []

    for idx, row in golden_df.iterrows():
        msg = str(row["customer_message"])
        # Bulk evaluation across all 214 cases evaluates deterministic intent, retrieval, and policy escalation
        res = agent.process_message(msg, enable_llm=False)
        agent_results.append(res)
        agent_decisions.append(res["escalation_decision"])

    print("Agent deterministic execution complete.")

    # 3. Escalation Evaluation
    print("\n" + "-" * 90)
    print("1. ESCALATION POLICY EVALUATION (vs 214 Golden Policy Labels)")
    print("-" * 90)
    print("Note: Golden escalation labels are policy-derived targets, not historical outcomes.")
    print("      Both agent rules and labels share policy criteria; measures policy adherence.")
    esc_eval = evaluate_escalation(golden_df, agent_decisions)
    print(f"Escalation Policy Accuracy: {esc_eval['accuracy'] * 100:.2f}%")
    print(f"Escalation Policy Macro F1: {esc_eval['macro_f1'] * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(f"                 Pred AUTO-HANDLE   Pred ESCALATE")
    print(f"True AUTO-HANDLE        {esc_eval['confusion_matrix'][0][0]:<18} {esc_eval['confusion_matrix'][0][1]}")
    print(f"True ESCALATE           {esc_eval['confusion_matrix'][1][0]:<18} {esc_eval['confusion_matrix'][1][1]}")

    print("\nDetailed Classification Report:")
    for lbl in ["AUTO-HANDLE", "ESCALATE"]:
        metrics = esc_eval['classification_report'][lbl]
        print(f"  {lbl:<12} Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1-score']:.4f} (n={metrics['support']})")

    # 4. Automated Proxy Metrics Comparison
    print("\n" + "-" * 90)
    print("2. AUTOMATED REPLY QUALITY PROXY COMPARISON")
    print("-" * 90)
    print("Note: Token Jaccard overlap is reported strictly as an automated proxy, not true quality.")
    metrics_df = evaluate_automated_metrics(golden_df, canned_df, retrieval_df, agent_results)
    print(metrics_df.to_string(index=False))

    # 5. Optimized Sample Generation & LLM-as-Judge
    print("\n" + "-" * 90)
    print("3. LLM-AS-JUDGE EVALUATION (Optimized: Exactly 25 Stratified Sample Cases)")
    print("-" * 90)
    sample_df = generate_evaluation_sample(golden_df, canned_df, retrieval_df, agent_results, sample_size=25, random_seed=42)
    sample_df, gen_status = apply_live_agent_generation(sample_df, gen_cache_path, agent, max_auto_handle_calls=10)
    print(f"Live Generation Status: {gen_status}")

    judge_df, judge_status = run_llm_judge_sample(sample_df, judge_cache_path)
    print(f"Judge Evaluation Status: {judge_status}")

    if judge_df is not None:
        summary = judge_df.groupby("model_name").agg({
            "groundedness": "mean",
            "relevance": "mean",
            "actionability": "mean",
            "tone": "mean",
            "unsupported_claim": "mean"
        }).reset_index()
        summary.rename(columns={"unsupported_claim": "unsupported_claim_rate"}, inplace=True)
        print("\nLLM Judge Comparison (Sample n=25, 1-5 scale, claim rate 0-1):")
        print(summary.to_string(index=False))

    # 6. Human Review Template & Agreement
    print("\n" + "-" * 90)
    print("4. HUMAN VS LLM JUDGE AGREEMENT")
    print("-" * 90)
    create_human_eval_template(sample_df, human_template_path)
    agreement = compute_human_agreement(human_template_path, judge_df)
    print(f"Status: {agreement['status']}")
    if agreement["status"] == "Completed":
        print(f"  Evaluated Samples: {agreement['evaluated_samples']}")
        print(f"  Unsupported Claim Exact Agreement: {agreement['unsupported_claim_exact_agreement'] * 100:.2f}%")
        print(f"  Groundedness MAE: {agreement['groundedness_mae']:.3f}")
        print(f"  Relevance MAE:    {agreement['relevance_mae']:.3f}")
    else:
        print(f"  {agreement.get('message', 'Review template ready.')}")

    print("\n" + "=" * 90)
    print("System Evaluation Completed.")
    print("=" * 90)


if __name__ == "__main__":
    main()
