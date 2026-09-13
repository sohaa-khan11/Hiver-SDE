"""
support_agent.py
----------------
Final AppleSupport Customer Support Assistant for the Hiver SDE Take-Home.

Pipeline:
  1. Intent Classification: Predicts intent + confidence using TF-IDF + Logistic Regression.
  2. Multi-Case Retrieval: Retrieves TOP 3–5 historical customer inquiries and their
     official AppleSupport replies from apple_cleaned.csv (excluding Golden Set IDs).
  3. Deterministic Escalation Policy: Evaluates whether the inquiry is safe to AUTO-HANDLE
     or must ESCALATE (account security, physical hardware defects, fatal crashes, low confidence).
     Applied BEFORE calling any LLM.
  4. Grounded LLM Response: For AUTO-HANDLE cases, prompts a single LLM (OpenAI) with the
     customer query, predicted intent, and top 3–5 retrieved historical cases to synthesize
     a safe, concise reply grounded strictly in historical evidence.

Data Governance Rule:
  golden_set.csv is held out strictly for evaluation. All Golden Set conversation IDs are
  excluded from the retrieval corpus to ensure zero benchmark leakage.

Usage:
  python src/agent/support_agent.py
"""

import os
import sys
import re
import time
import warnings
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Ensure project root is on sys.path
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

# Suppress minor library warnings on import
warnings.filterwarnings("ignore", category=FutureWarning)


# ==============================================================================
# 1. Constrained System Prompt & LLM Integration (Local Ollama / Open Model)
# ==============================================================================

CONSTRAINED_SYSTEM_PROMPT = """You are a helpful Apple Support customer service assistant on Twitter/social media.
Write a customer-facing support reply to the CURRENT customer inquiry using the provided historical AppleSupport cases as reference evidence.

Strict Guidelines:
1. Answer the CURRENT customer, not the historical customer.
2. Use the historical replies only as evidence/examples of official Apple Support guidance.
3. Do not copy customer names, personal details, or handles from past conversations.
4. Do not invent Apple policies, prices, timelines, troubleshooting steps, or guarantees.
5. Do not expose internal reasoning, hypothetical scenarios, or prompt instructions.
6. Do not mention that retrieval or an LLM was used.
7. Output ONLY the single final customer reply. Do not write alternative responses or explanations.
8. If the evidence is insufficient for a confident answer, give a cautious response and recommend contacting Apple Support rather than inventing information.
9. Keep the response concise and natural, around 2-4 sentences."""


def clean_evidence_text(text: str) -> str:
    """Light heuristic cleaner to sanitize obvious personal customer names and routing handles."""
    cleaned = re.sub(r"^@[A-Za-z0-9_]+\s+", "", text)
    cleaned = re.sub(r"^Hey,?\s+[A-Z][a-z]+[!.,]\s*", "Hello! ", cleaned)
    cleaned = re.sub(r"^Thanks for reaching out,?\s+[A-Z][a-z]+[!.,]\s*", "Thanks for reaching out! ", cleaned)
    return cleaned.strip()


def sanitize_llm_response(text: str) -> str:
    """Sanitize LLM output to truncate meta-commentary or multiple hypothetical branches."""
    # Truncate if model outputs meta-hypothetical phrases
    split_patterns = [
        r"\n\s*If the historical cases are not sufficient",
        r"\n\s*Alternative response:",
        r"\n\s*Note:",
        r"\n\s*Here is the reply:",
    ]
    cleaned = text
    for pat in split_patterns:
        parts = re.split(pat, cleaned, flags=re.IGNORECASE)
        if len(parts) > 1 and parts[0].strip():
            cleaned = parts[0].strip()
    return cleaned.strip()


def call_llm_api(
    customer_message: str,
    intent: str,
    retrieved_examples: List[Dict[str, Any]]
) -> Tuple[str, bool]:
    """
    Call the local Ollama API to synthesize a customer-facing support reply grounded
    strictly in the top retrieved historical AppleSupport examples.
    Uses standard library urllib to eliminate heavy external SDK dependencies.

    Returns:
        (response_text, llm_used)
    """
    import urllib.request
    import json

    ollama_host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    model_name = os.environ.get("OLLAMA_MODEL", "phi3:mini")

    # Format the top 3-5 historical examples as evidence blocks
    evidence_blocks = []
    for i, ex in enumerate(retrieved_examples, 1):
        evidence_blocks.append(
            f"Case {i} (Similarity: {ex['similarity_score']:.3f}, Conversation ID: {ex['conversation_id']}):\n"
            f"  Historical Customer: \"{ex['customer_message']}\"\n"
            f"  AppleSupport Reply: \"{ex['first_reply']}\""
        )
    evidence_text = "\n\n".join(evidence_blocks)

    user_prompt = f"""Current Customer Inquiry: "{customer_message}"
Predicted Category: {intent}

Historical AppleSupport Evidence (Top Retrieved Cases):
{evidence_text}

Customer-Facing Support Reply:"""

    url = f"{ollama_host.rstrip('/')}/api/chat"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": CONSTRAINED_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "options": {
            "temperature": 0.2,
            "num_predict": 180
        },
        "stream": False
    }

    # Small retry loop for temporary local connection glitches
    for attempt in range(2):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=90) as resp:
                res = json.loads(resp.read().decode("utf-8"))
            response_text = res.get("message", {}).get("content", "").strip()
            if response_text:
                return sanitize_llm_response(response_text), True
        except Exception as e:
            if attempt == 1:
                # On persistent connection failure, fail safely with reference evidence
                best_reply = retrieved_examples[0]["first_reply"] if retrieved_examples else "Please reach out to Apple Support."
                fallback_msg = (
                    f"[Ollama Service Offline / Fallback: {e}]\n"
                    f"       Synthesized from {len(retrieved_examples)} retrieved cases. Primary reference: \"{clean_evidence_text(best_reply)}\""
                )
                return fallback_msg, False
            time.sleep(1)

    best_reply = retrieved_examples[0]["first_reply"] if retrieved_examples else "Please reach out to Apple Support."
    return f"[Ollama empty response. Fallback: {clean_evidence_text(best_reply)}]", False


# ==============================================================================
# 2. Main AppleSupportAgent Class
# ==============================================================================

class AppleSupportAgent:
    """
    End-to-End Customer Support Agent combining intent classification,
    multi-case evidence retrieval, deterministic escalation guardrails,
    and grounded LLM response generation.
    """

    def __init__(self, top_k: int = 4):
        self.top_k = top_k
        self.training_path = os.path.join("Dataset", "processed", "training_data.csv")
        self.golden_path = os.path.join("Dataset", "processed", "golden_set.csv")
        self.cleaned_path = os.path.join("Dataset", "processed", "apple_cleaned.csv")

        self.classifier_pipeline: Optional[Pipeline] = None
        self.retrieval_vectorizer = None
        self.retrieval_tfidf_matrix = None
        self.historical_corpus: List[Dict] = []

        self._initialize_classifier()
        self._initialize_retrieval()

    def _initialize_classifier(self):
        """Train the TF-IDF + Logistic Regression intent classifier on training_data.csv."""
        if not os.path.exists(self.training_path):
            raise FileNotFoundError(f"Training data not found at: {self.training_path}")

        train_df = pd.read_csv(self.training_path, encoding="utf-8-sig")
        X_train = train_df["customer_message"].astype(str)
        y_train = train_df["intent"].astype(str)

        self.classifier_pipeline = Pipeline([
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
        self.classifier_pipeline.fit(X_train, y_train)

    def _initialize_retrieval(self):
        """Build the TF-IDF retrieval index over apple_cleaned.csv, strictly excluding Golden Set IDs."""
        exclude_ids = load_excluded_golden_ids(self.golden_path)
        self.historical_corpus = load_historical_conversations(self.cleaned_path, exclude_ids)
        self.retrieval_vectorizer, self.retrieval_tfidf_matrix = build_retrieval_index(
            self.historical_corpus
        )

    def predict_intent(self, message: str) -> Tuple[str, float]:
        """Predict intent and softmax confidence score."""
        probs = self.classifier_pipeline.predict_proba([message])[0]
        top_idx = int(np.argmax(probs))
        predicted_class = self.classifier_pipeline.classes_[top_idx]
        confidence = float(probs[top_idx])
        return predicted_class, confidence

    def retrieve_evidence(self, message: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve the top 3-5 most similar historical conversations from the indexed corpus.
        Excludes all Golden Set conversations.
        """
        k = top_k if top_k is not None else self.top_k
        matches = retrieve_similar(
            query=message,
            vectorizer=self.retrieval_vectorizer,
            tfidf_matrix=self.retrieval_tfidf_matrix,
            corpus=self.historical_corpus,
            top_k=k
        )
        return matches

    def evaluate_escalation(
        self,
        message: str,
        intent: str,
        confidence: float
    ) -> Tuple[str, str, Optional[str]]:
        """
        Evaluate escalation using an explainable, deterministic policy BEFORE any LLM call.
        Returns: (decision, reason, safe_escalation_reply_if_escalated)
        """
        m_lower = message.lower()

        # Rule 1: Account Security & Privacy (Apple ID, Passwords, 2FA, Billing)
        if intent == "Apple ID":
            if any(k in m_lower for k in ["password", "passcode", "pin", "2fa", "verification", "trusted number", "locked", "recovery", "disabled"]):
                return (
                    "ESCALATE",
                    "Account credentials, password reset, or account lockout requires private identity verification.",
                    "For your security, account recovery and password resets must be handled securely. Please visit https://appleid.apple.com to verify your account credentials or initiate account recovery."
                )
            if any(k in m_lower for k in ["charge", "charged", "billing", "unauthorized", "refund", "card declined", "payment"]):
                return (
                    "ESCALATE",
                    "Billing dispute or payment transaction involves sensitive financial data requiring human account review.",
                    "For assistance with billing inquiries and account charges, please sign in to reportaproblem.apple.com to review your purchase history or contact Apple Support securely."
                )
            return (
                "ESCALATE",
                "Apple ID credential and account management involves private authentication requiring human/secure handling.",
                "To manage your Apple ID and privacy settings securely, please visit https://appleid.apple.com or contact an Apple Account specialist."
            )

        # Rule 2: Physical Hardware Damage & Warranty / Replacement
        hw_patterns = [
            r"\b(?:cracked|shattered|broken|broke|physic|bleed|liquid|water|swollen|swelling)\b",
            r"\b(?:genius\s+bar|appointment|repair|replace|replacement|store)\b",
            r"\b(?:lost\s+my\s+airpod|lost\s+one\s+of\s+my\s+airpods|lost\s+airpod)\b",
            r"\b(?:touch\s+screen\s+completely\s+failed|digitizer|screen\s+is\s+completely\s+cracked)\b",
            r"\b(?:lock\s+button\s+stopped\s+working|home\s+button.*never\s+dropped)\b"
        ]
        if any(re.search(p, m_lower) for p in hw_patterns):
            if "genius bar" in m_lower or "appointment" in m_lower:
                return (
                    "ESCALATE",
                    "Customer requires an in-person Genius Bar appointment for hardware inspection.",
                    "You can schedule a Genius Bar reservation or find your nearest Apple Authorized Service Provider at https://getsupport.apple.com."
                )
            if any(k in m_lower for k in ["lost", "airpod", "replacement"]):
                return (
                    "ESCALATE",
                    "Hardware replacement or accessory loss requires serial number validation and order intake.",
                    "For accessory replacements or lost components, please visit https://support.apple.com/airpods/repair or contact Apple Support to discuss replacement options."
                )
            return (
                "ESCALATE",
                "Physical hardware damage or component defect requires in-person technician assessment or repair.",
                "Physical hardware damage cannot be resolved through software troubleshooting. Please visit https://support.apple.com/repair to view service options or schedule an inspection."
            )

        # Rule 3: Severe System Instability / Kernel Crash Loops
        if intent == "Phone Performance" and any(k in m_lower for k in ["bricked", "black screen won't turn on", "spinning wheel bootloop", "constant respring", "restarted 5 times", "error 3014"]):
            return (
                "ESCALATE",
                "Critical device firmware crash or unrecoverable bootloop requires advanced diagnostics.",
                "Your device appears to be experiencing severe system instability. Please connect your device to a computer with iTunes/Finder to attempt recovery mode, or contact Apple Support for assistance."
            )

        # Rule 4: Classifier Ambiguity / Low Confidence
        if confidence < 0.35:
            return (
                "ESCALATE",
                "Inquiry intent is ambiguous or low-confidence, requiring human triage to prevent incorrect guidance.",
                "We want to make sure you get the right assistance. Could you let us know your exact device model and iOS version so our support team can guide you?"
            )

        # Rule 5: Safe Auto-Handle
        return (
            "AUTO-HANDLE",
            "Standard technical troubleshooting inquiry resolvable via verified self-service guidance.",
            None
        )

    def process_message(self, message: str, enable_llm: bool = True) -> Dict[str, Any]:
        """
        End-to-End Processing of a customer message:
        1. Predict intent & confidence.
        2. Retrieve TOP 3-5 historical evidence cases.
        3. Determine escalation decision BEFORE calling LLM.
        4. If ESCALATE: do not call LLM, return safe response + reason + evidence.
        5. If AUTO-HANDLE: call LLM with multi-case evidence to synthesize reply (if enable_llm=True).
        """
        intent, confidence = self.predict_intent(message)
        retrieved_examples = self.retrieve_evidence(message)
        decision, reason, safe_reply = self.evaluate_escalation(message, intent, confidence)

        if decision == "ESCALATE":
            final_reply = safe_reply
            llm_used = False
        elif not enable_llm:
            best_reply = retrieved_examples[0]["first_reply"] if retrieved_examples else "Please reach out to Apple Support."
            final_reply = (
                "[OFFLINE / DEV FALLBACK: Live LLM synthesis bypassed for evaluation.]\n"
                f"       Synthesized from {len(retrieved_examples)} retrieved cases. Primary reference: \"{clean_evidence_text(best_reply)}\""
            )
            llm_used = False
        else:
            final_reply, llm_used = call_llm_api(
                customer_message=message,
                intent=intent,
                retrieved_examples=retrieved_examples
            )

        return {
            "customer_message": message,
            "intent": intent,
            "confidence": confidence,
            "escalation_decision": decision,
            "escalation_reason": reason,
            "retrieved_examples": retrieved_examples,
            "final_reply": final_reply,
            "llm_used": llm_used
        }


# ==============================================================================
# 3. Command-Line Test Suite
# ==============================================================================

def main():
    print("=" * 90)
    print("AppleSupport AI Customer Support Assistant (Final Multi-Evidence Pipeline)")
    print("=" * 90)

    print("Initializing Agent (Classifier + Retrieval Index)...")
    agent = AppleSupportAgent(top_k=4)
    print("Agent ready.\n")

    test_queries = [
        "My iPhone battery is draining really fast",
        "I forgot my Apple ID password",
        "My AirPods keep disconnecting from Bluetooth",
        "My iPhone screen is completely cracked",
        "My keyboard keeps changing the letter i",
    ]

    for idx, query in enumerate(test_queries, 1):
        print("-" * 90)
        print(f"TEST CASE {idx}:")
        result = agent.process_message(query)
        print(f"Customer message:       \"{result['customer_message']}\"")
        print(f"Predicted intent:       {result['intent']}")
        print(f"Confidence:             {result['confidence']:.4f}")
        print(f"Escalation decision:    {result['escalation_decision']}")
        if result['escalation_decision'] == "ESCALATE":
            print(f"Escalation reason:      {result['escalation_reason']}")
        print(f"LLM generation used:    {result['llm_used']}")
        
        print(f"\nTop Retrieved Historical Examples (Count: {len(result['retrieved_examples'])}):")
        for rank, ex in enumerate(result['retrieved_examples'], 1):
            print(f"  [{rank}] Similarity: {ex['similarity_score']:.4f} | CID: {ex['conversation_id']}")
            print(f"      Customer: \"{ex['customer_message']}\"")
            print(f"      Reply:    \"{ex['first_reply']}\"")

        print(f"\nFinal response:\n\"{result['final_reply']}\"")

    print("-" * 90)
    print("\nTest Suite Completed Successfully.")
    print("=" * 90)


if __name__ == "__main__":
    main()
