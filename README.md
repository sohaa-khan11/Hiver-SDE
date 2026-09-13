# AppleSupport AI Customer Support Assistant

An end-to-end AI customer support pipeline built from the Kaggle [Customer Support on Twitter (TWCS)](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) dataset for the **Hiver SDE Take-Home Assignment**.

The system classifies incoming customer inquiries, retrieves grounded historical resolutions from official AppleSupport conversations, applies an explainable deterministic escalation policy, and uses an LLM to synthesize concise, hallucination-free support responses.

---

## 1. Problem Framing & Architecture

### Pipeline Flow

```text
Incoming Customer Message
           │
           ▼
[Step 1] Intent Classifier (TF-IDF + Logistic Regression)
           │
           ├── Intent & Calibrated Confidence Score
           │
           ▼
[Step 2] Historical Evidence Retrieval (TF-IDF + Cosine Similarity)
           │
           ├── Top 3–5 Historical Cases (Excluding Golden Set IDs)
           │
           ▼
[Step 3] Deterministic Escalation Policy (Rule-Based Guardrails)
           │
     ┌─────┴──────────────────────┐
     │                            │
[ESCALATE]                   [AUTO-HANDLE]
     │                            │
     ▼                            ▼
Safe Human-Routing Reply     [Step 4] Grounded LLM Rewriter
(Account Security, Repairs,  (Synthesizes evidence, strips noise,
 Severe System Crashes)       strictly zero hallucinated facts)
     │                            │
     └─────────────┬──────────────┘
                   │
                   ▼
       Final Customer Response
```

### Core Design Philosophy
1. **Zero Hallucination via Constrained Synthesis**: The LLM is **never** asked to generate technical troubleshooting procedures from memory. It is constrained to adapt the verified historical AppleSupport replies retrieved for that specific problem.
2. **Deterministic Safety Guardrail**: Escalation decisions (account security, physical hardware damage, severe boot loops) are handled **before** the LLM call using explainable rules. LLMs are never allowed to make escalation decisions.
3. **Reproducibility with Zero Leakage**: The final 214-example evaluation set (`Dataset/processed/golden_set.csv`) was sampled at the conversation level, initially labeled with an automated/AI-assisted process, then manually reviewed and corrected to produce the final benchmark. All Golden Set conversation IDs are programmatically excluded from the historical retrieval index.

---

## 2. Why AppleSupport Was Selected

Out of the 108 brands in TWCS, we compared the three largest candidate brands: `AppleSupport`, `SpotifyCares`, and `Delta`.

| Evaluation Metric | AppleSupport | SpotifyCares | Delta |
| :--- | :---: | :---: | :---: |
| **Brand Outbound Tweets** | **106,860** | 43,265 | 42,253 |
| **Valid Root Conversations** | **80,250** | 26,940 | 28,485 |
| **Actionable Troubleshooting Rate** | **22.25%** | 18.14% | 4.99% |
| **Agent Signature Noise** | **0.00%** (Clean) | 74.68% (`/LS`) | 82.30% (`*QB`) |
| **Direct Message (DM) Routing** | 46.92% | 31.35% | 15.17% |

* **Troubleshooting Richness**: 22.25% of AppleSupport replies contain concrete technical steps (settings navigation, iOS version checks, hardware resets) compared to 4.99% for Delta (mostly flight booking lookups).
* **Zero Signature Noise**: Apple maintains a unified brand voice with 0% employee initials, avoiding contaminated training or retrieval targets.
* **Explainable Support Boundaries**: Clear technical demarcations between self-service software workarounds and physical hardware repair needs.

---

## 3. Intent Taxonomy

Analyzing opening customer inquiries across 9,000 reconstructed threads yielded an 8-category support taxonomy:

1. **`Battery` (11.2%)**: Fast drain, sudden shutdowns, charging cable/port failures.
2. **`Phone Performance` (8.4%)**: System freezing, UI stutter, random reboots, boot loops.
3. **`Keyboard` (8.6%)**: iOS 11 `"I"` autocorrect glitch (`"A [?]"`), predictive text bugs.
4. **`Apple ID` (4.6%)**: Forgotten passwords, account lockouts, 2FA codes, billing charges.
5. **`Sound & Bluetooth` (3.4%)**: AirPods disconnects, Bluetooth pairing, speaker/mic audio.
6. **`Screen & Camera` (5.8%)**: Cracked screens, black camera viewfinders, touch digitizer failure.
7. **`Apps & Storage` (4.6%)**: App Store download loops, storage full alerts, app crashes.
8. **`Other` (59.0% raw / 2.3% Golden)**: General rants, social chatter, out-of-scope requests.

---

## 4. Data Provenance & Governance

To maintain strict scientific and engineering rigor, the project clearly separates four tiers of data:

1. **Historical Retrieval Corpus (`Dataset/processed/apple_cleaned.csv`)**:
   - 8,786 authentic historical AppleSupport conversations extracted from TWCS and conservatively cleaned.
   - Strictly filtered to exclude all benchmark conversation IDs to guarantee zero retrieval leakage.
2. **Weak-Labeled Training Set (`Dataset/processed/training_data.csv`, 894 rows)**:
   - Generated using keyword and regex heuristics over non-benchmark conversations.
   - An automated audit of 100 randomly sampled examples verified **91.0% precision** against taxonomy rules.
3. **Manually Reviewed Golden Benchmark (`Dataset/processed/golden_set.csv`, 214 rows)**:
   - Sampled at the conversation level, initially labeled with an automated/AI-assisted process, then manually reviewed and corrected to produce the final evaluation benchmark.
   - Kept completely separate and frozen; never seen during classifier training or retrieval indexing.
4. **Policy-Derived Escalation Labels**:
   - The escalation ground truth in the Golden Set was created by applying defined safety policy rules (account security, physical damage, severe bootloops).
   - **Important**: These are policy-derived evaluation targets, not historical observed customer outcomes.

---

## 5. Historical Retrieval Engine

* **Corpus**: 8,786 historical AppleSupport conversation threads indexed from `Dataset/processed/apple_cleaned.csv` (excluding all 214 Golden Set conversation IDs).
* **Method**: Sparse TF-IDF vectorizer + Cosine dot-product.
* **Multi-Case Context**: Retrieves top 3–5 similar cases (`top_k=4` default). Technical support relies on exact tokens (`"iOS 11.0.2"`, `"AirPods"`, `"error 3014"`). Presenting multiple historical cases allows the downstream synthesizer to ignore irrelevant matches (e.g. a lost AirPod case) and extract relevant diagnostic questions (e.g. asking for device model and iOS version).

---

## 6. Deterministic Escalation Policy

Before invoking the LLM, the system applies explainable rule-based safety guardrails:

1. **Account Credentials & Financial Security (`ESCALATE`)**: Apple ID password resets, 2FA, account lockouts, billing charges. Bots must never request credentials in chat.
2. **Physical Hardware Defects & Replacements (`ESCALATE`)**: Cracked glass, liquid damage, battery swelling, hardware button failures, Genius Bar appointments. Software cannot repair cracked glass.
3. **Severe System Instability (`ESCALATE`)**: Unrecoverable boot loops, bricked recovery screens, iTunes restore error 3014.
4. **Classifier Ambiguity (`ESCALATE`)**: Inquiries with softmax confidence `< 0.35` are routed to human triage to prevent confident misdirection.
5. **Safe Inquiries (`AUTO-HANDLE`)**: Software troubleshooting, settings navigation, and verified workarounds.

---

## 7. Reply-Generation Baselines

To prove the necessity and value of the final LLM agent, we established two non-generative baselines across all 214 Golden Set conversations:

* **Baseline 1 (Intent + Generic Canned Reply)**: Maps predicted intent to a fixed corporate troubleshooting macro. Safe and predictable, but suffers from cascading misclassification errors and zero personalization.
* **Baseline 2 (Historical Retrieval Only)**: Returns the verbatim top-1 historical human agent reply. Rich in domain knowledge, but blindly copies previous customers' names (`"Hey Ryan!"`), dead links, and unhelpful one-line DM invites.

---

## 8. Comprehensive Evaluation Results

### A. Intent Classification Performance
* **Training Validation Split (20% holdout, 179 samples)**:
  - Accuracy: **91.06%** | Macro F1: **90.03%**
* **Golden Set Evaluation (Held-Out Benchmark, 214 samples)**:
  - Accuracy: **70.56%** (151 / 214 correct) | Macro F1: **61.98%**
  - Per-Intent F1: `Keyboard` (91.8%), `Battery` (83.0%), `Sound & Bluetooth` (75.3%), `Apps & Storage` (67.5%), `Phone Performance` (62.2%), `Apple ID` (58.8%), `Screen & Camera` (54.2%), `Other` (0.0%).

### B. Escalation Policy Evaluation (vs 214 Golden Policy Labels)
* **Accuracy**: **62.15%** | **Macro F1**: **59.04%**
* **Confusion Matrix**:
  - True `AUTO-HANDLE` (n=71): 37 Correct (52.1%), 34 False Escalations
  - True `ESCALATE` (n=143): 96 Correct (67.1%), 47 False Auto-Handles
* **Circularity & Cascading Errors**:
  The escalation ground-truth labels are policy-derived targets based on defined safety criteria, not real-world observed escalation outcomes. Furthermore, the agent's escalation policy rules use predicted intent. When the classifier misidentifies an intent (e.g. classifying an Apple ID password issue as Apps & Storage), it misses the credential escalation rule. Because both the labeling policy and agent rules share the same domain criteria, this evaluation tests rule alignment and pipeline adherence rather than unbiased real-world accuracy.

### C. Automated Reply Quality Comparison (214 Cases)
*Note: Lexical Jaccard overlap is reported strictly as an automated word-overlap proxy, not a true semantic quality measure.*

| Model | Response Availability | Avg Word Count | Avg Char Count | Cust Overlap Proxy (Jaccard) | Actionable Guidance Rate | Escalation Handled |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1 (Canned)** | 100% | 27.4 | 178.3 | 0.0498 | 100.0% | 0.0% (None) |
| **Baseline 2 (Retrieval-Only)** | 100% | 22.6 | 117.7 | 0.0548 | 82.7% | 0.0% (None) |
| **Final Support Agent** | 100% | 31.6 | 189.7 | 0.0408 | 46.3% | **60.7% (130/214)** |

---

## 9. LLM-as-Judge Evaluation Framework

To evaluate generation quality without relying on closed proprietary APIs or incurring cloud expenses, the evaluation harness uses a local open-weight instruction-tuned model (`phi3:mini`, 3.8B parameters via local Ollama).

The harness selects an optimized representative **25-case stratified sample** from the Golden Set (seed=42, covering all 8 intents) and evaluates Baseline 1, Baseline 2, and Final Agent across 5 criteria:

1. **Groundedness (1–5)**: Supported strictly by official Apple evidence; zero invented facts.
2. **Relevance (1–5)**: Addresses the current customer's specific problem.
3. **Actionability (1–5)**: Provides concrete next steps, links, or diagnostic questions.
4. **Support Tone (1–5)**: Polite, concise, empathetic social customer voice.
5. **Unsupported Claim Rate (0 or 1)**: Flags hallucinated prices, repair promises, or fake steps.

### Empirical LLM-as-Judge Results (Local `phi3:mini`, n=25 Sample):

| Model | Groundedness (1–5) | Relevance (1–5) | Actionability (1–5) | Tone (1–5) | Unsupported Claim Rate (0–1) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1 (Canned)** | 3.44 | 3.56 | 4.24 | 4.56 | **0.00** |
| **Baseline 2 (Retrieval)** | 3.88 | 4.28 | 3.96 | 4.56 | 0.04 |
| **Final Support Agent** | **4.32** | **4.40** | **4.48** | **4.84** | 0.04 |

### Key Findings & Self-Grading Bias:
* **Groundedness Gain (+0.88 vs Canned, +0.44 vs Retrieval)**: The Final Agent combines retrieved troubleshooting facts with customer-specific context, eliminating generic advice without blindly quoting raw past conversations.
* **Tone & Relevance**: The Final Agent achieved the highest tone (4.84/5) and relevance (4.40/5) by stripping previous customers' names and greeting the current user directly.
* **Important Self-Grading Bias Limitation**: Because `phi3:mini` served both as the response generator and the comparative judge, same-model preference bias is a recognized factor. While temperature was set to 0.0 and JSON schema constrained the evaluation, formal validation requires independent human ratings.
* **Execution & Persistent Caching**:
  - Live generated replies are cached to `Dataset/processed/llm_generation_cache.csv`.
  - Judge scores are cached to `Dataset/processed/llm_judge_results.csv`.
  - Re-running `python src/evaluation/evaluate_system.py` loads both caches instantly (0 repeated inference calls).

---

## 10. Human vs. LLM Judge Agreement

The assignment requires evaluating human-vs-LLM agreement to validate the judge's scoring reliability:
* The harness automatically maintains `Dataset/processed/human_eval_template.csv` containing the **exact same 25 sampled cases** used for the LLM judge.
* In accordance with academic honesty, **human scores are not fabricated or pre-filled**. Rating columns (`human_groundedness_1to5`, `human_relevance_1to5`, etc.) remain blank for authentic review.
* **Current Status**: **Pending Manual Human Review**.
* Once filled, running `python src/evaluation/evaluate_system.py` automatically computes exact agreement on binary unsupported claims and Mean Absolute Error (MAE) for the 1–5 metrics.

---

## 11. Top 5 Real Failures & Root-Cause Analysis

Analyzing actual errors from the 214 Golden Set predictions reveals key systemic failure patterns:

1. **CID 427841** (`Keyboard` $\rightarrow$ Predicted `Screen & Camera`)
   - *Message*: *"Hi I'm using ios 11.0.2 on iPhone 6, keyboard doesn't fill screen when device rotated on landscape? Please solve this..."*
   - *Cause*: Co-occurrence of `"keyboard"` and `"screen... landscape"`. Strong n-gram weights for `"screen"` dominated.
   - *Improvement*: Contextual dependency parsing or giving higher priority to input-method tokens.
2. **CID 811638** (`Apple ID` $\rightarrow$ Predicted `Apps & Storage`)
   - *Message*: *"Hi trying to download a game on to my phone but keeps telling payment failed, can't sign in to itunes too"*
   - *Cause*: Lexical overlap between downloading an app and iTunes Store authentication / payment failure.
   - *Improvement*: Elevating authentication/payment failure keywords to override generic app-download tokens.
3. **CID 2064248** (`Screen & Camera` $\rightarrow$ Predicted `Phone Performance`)
   - *Message*: *"updated 11.0.2 and now my phone is freezing up and screen rotation sticking in landscape mode!!!!"*
   - *Cause*: Multi-symptom complaint. Bag-of-words representation cannot determine which grievance is primary when both "freezing up" and "screen rotation" appear.
   - *Improvement*: Multi-label classification or hierarchical intent prediction.
4. **CID 1028445** (`Battery` $\rightarrow$ Predicted `Sound & Bluetooth`)
   - *Message*: *"my phone died and now ive plugged it in and its comepletly blackscreen and it wont even show the charging symbol HELP @AppleSupport"*
   - *Cause*: Vocabulary starvation on power failure verbs without the explicit word `"battery"`.
   - *Improvement*: Expanding weak-supervision regexes for charging hardware terms (`"plugged it in"`, `"charging symbol"`).
5. **AirPods Disconnection Retrieval Drift**
   - *Message*: *"My AirPods keep disconnecting from Bluetooth"*
   - *Cause*: In top-1 retrieval, the highest match was a generic complaint whose reply was a bare DM invitation link (`CID 1557363`).
   - *Improvement*: Solved in the final agent by retrieving top 4 cases, allowing the LLM to ignore the DM invite and synthesize diagnostic questions from the other cases.

---

## 12. "What Is Misleading About My Headline Number?"

A standard engineering pitfall is reporting validation accuracy as true system capability. Our system achieved:
* **Validation Accuracy (Weak Labels)**: **91.06%**
* **Golden Set Accuracy (Benchmark)**: **70.56%** (~20.5% drop)

### Why the 91.06% Number Is Misleading:
1. **Heuristic Rule Leakage**: The weak-labeled training data was generated using keyword patterns. The validation split was evaluated against those same rules, creating an artificially easy, closed-world test.
2. **Distribution Shift**: Real customer complaints in the Golden Set contain organic phrasing, emotional slang, typos, and multi-symptom rants not captured in clean heuristic patterns.
3. **Class Starvation**: Minority classes like `Other` (only 14 training examples) achieved 0.0% recall on the Golden Set, dragging down real-world performance while remaining invisible in the validation headline score.
4. **Cascading Pipeline Vulnerability**: A 70.56% classifier accuracy means ~29.4% of queries enter downstream retrieval with suboptimal intent context, causing cascading policy escalation errors.

---

## 13. What I Would Improve With One More Week

1. **Lightweight Semantic Hybrid Retrieval**: Combine sparse TF-IDF with small dense sentence embeddings (e.g. `all-MiniLM-L6-v2`) via reciprocal rank fusion to capture technical vocabulary without losing semantic synonyms.
2. **Confidence-Gated Escalation Tuning**: Optimize the softmax confidence threshold (currently `0.35`) using ROC analysis on a validation split to maximize recall on safety-critical escalation cases.
3. **Multi-Label Intent Support**: Allow the classifier to output secondary intent tags for compound customer complaints (e.g. iOS update + battery drain).
4. **Automated URL Verification**: Cross-reference retrieved Apple Support URLs against active documentation endpoints to prevent serving deprecated 2017 links.

---

## 14. Repository Structure

```text
Hiver/
├── .gitignore                          # Excludes raw data, caches, and generated prediction CSVs
├── requirements.txt                    # Minimal genuine dependencies (pandas, numpy, sklearn, openai)
├── README.md                           # Comprehensive report and reproduction documentation
├── docs/
│   └── decision_log.md                 # 15 structured architectural decisions (Decision, Why, Trade-off)
├── Dataset/
│   └── processed/
│       ├── golden_set.csv              # Tracked 214-example evaluation benchmark (manually reviewed)
│       └── training_data.csv           # Tracked 894-example weak-labeled training dataset
└── src/
    ├── data/
    │   ├── extract_apple.py            # Extracts AppleSupport conversations from TWCS
    │   ├── clean_apple.py              # Conservative cleaning, HTML entity decoding, URL normalization
    │   ├── create_labels.py            # Generates weak-label training data
    │   └── audit_labels.py             # Reproducible weak-label quality audit (91.0% precision)
    ├── models/
    │   └── train_intent_classifier.py  # Trains TF-IDF + Logistic Regression intent classifier
    ├── retrieval/
    │   └── retrieve_similar.py         # Multi-case historical retrieval excluding Golden Set
    ├── agent/
    │   ├── baseline_canned.py          # Baseline 1: Intent + generic canned response
    │   ├── baseline_retrieval.py       # Baseline 2: Top-1 verbatim historical retrieval
    │   └── support_agent.py            # Final Agent: Multi-case retrieval + escalation policy + LLM
    └── evaluation/
        └── evaluate_system.py          # Comprehensive evaluation harness (escalation, proxies, judge)
```

---

## 15. Reproduction Guide

### Environment Setup
```bash
git clone https://github.com/sohaa-khan11/Hiver-SDE.git
cd Hiver-SDE
pip install -r requirements.txt
```

### 1. Run the Entire Deterministic Pipeline (Zero Local LLM Inference Required)
All core components run completely self-contained on standard CPU:

```bash
# 1. Train and evaluate the intent classifier on the Golden Set
python src/models/train_intent_classifier.py

# 2. Test historical evidence retrieval (top matching cases)
python src/retrieval/retrieve_similar.py

# 3. Run the two reply baselines
python src/agent/baseline_canned.py
python src/agent/baseline_retrieval.py

# 4. Run the final support agent (deterministic escalation + fallback reply)
python src/agent/support_agent.py

# 5. Run the system evaluation harness on the Golden Set
python src/evaluation/evaluate_system.py
```

### 2. Live Local LLM Generation & Judge Evaluation via Ollama
The generative components use an open-weight local model (`phi3:mini`) served via [Ollama](https://ollama.com). No API keys, credentials, or cloud subscriptions are required:

1. **Install and launch Ollama**:
   Download and install from [ollama.com](https://ollama.com). Ensure the local server is running at `http://127.0.0.1:11434`.
2. **Pull the model**:
   ```bash
   ollama pull phi3:mini
   ```
3. **Execute live support agent testing**:
   ```bash
   python src/agent/support_agent.py
   ```
4. **Execute live LLM generation and LLM-as-Judge evaluation**:
   ```bash
   python src/evaluation/evaluate_system.py
   ```
   *Note: Responses are automatically cached to `Dataset/processed/llm_generation_cache.csv` and `Dataset/processed/llm_judge_results.csv`. Subsequent runs execute in seconds.*

### 3. Optional: Raw Dataset Extraction & Regeneration
If you wish to re-extract the full dataset from scratch:
1. Download `customer-support-on-twitter.zip` from Kaggle.
2. Place `twcs.csv` at `Dataset/twcs/twitter_support.csv` (note: ~493 MB, intentionally ignored by Git).
3. Execute the extraction and cleaning scripts:
   ```bash
   python src/data/extract_apple.py
   python src/data/clean_apple.py
   python src/data/create_labels.py
   python src/data/audit_labels.py
   ```
