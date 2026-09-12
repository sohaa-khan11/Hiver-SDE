# Engineering Decision Log: AppleSupport AI Customer Support Assistant

This document records the core architectural and methodological decisions made during the development of the AppleSupport AI Customer Support system for the Hiver SDE Take-Home assignment.

Each decision follows the structure:
- **Decision**: What choice was made.
- **Why**: Concrete empirical or domain justification.
- **Trade-off**: What downside, constraint, or limitation was accepted.

---

### Decision 1: Focus Exclusively on AppleSupport from the TWCS Dataset
- **Decision:** Selected `AppleSupport` as the single brand domain after comparing the three largest candidates in TWCS (`AppleSupport`, `SpotifyCares`, and `Delta`).
- **Why:** AppleSupport contains the highest instructional resolution rate (22.25% of tweets contain actionable troubleshooting workflows like settings paths, resets, and OS checks vs. 4.99% for Delta), and maintains 0.00% employee signature noise (`*QB`, `/LS`).
- **Trade-off:** High Direct Message (DM) routing rate (46.92%) where agents route account or serial-number inquiries to private channels, requiring explicit handling in escalation modeling.

---

### Decision 2: Reconstruct Full Conversational Chains via Graph Traversal
- **Decision:** Reconstructed full parent-child conversation threads back to the initial customer root tweet rather than modeling isolated tweet-reply pairs.
- **Why:** 36.82% of AppleSupport interactions are multi-turn. Opening tweets often lack crucial context (e.g., "it still didn't work") which only becomes clear when viewing the antecedent agent response.
- **Trade-off:** Requires graph traversal across 2.81 million rows during extraction, resulting in higher offline extraction time (~2 minutes).

---

### Decision 3: Define an 8-Category Natural Intent Taxonomy
- **Decision:** Established an 8-intent taxonomy (`Battery`, `Phone Performance`, `Keyboard`, `Apple ID`, `Sound & Bluetooth`, `Screen & Camera`, `Apps & Storage`, `Other`) grounded in real customer complaints.
- **Why:** Groups real customer technical grievances by resolution domain rather than artificial syntax (e.g. kept iOS 11 letter "I" autocorrect bug under `Keyboard` rather than generic `Phone Performance` because it requires a distinct Settings workaround).
- **Trade-off:** High lexical overlap between `Phone Performance` ("phone freezing") and `Apps & Storage` ("app crashing") creates boundary ambiguity for linear classifiers.

---

### Decision 4: Conservative Preprocessing (Preserve Emojis, Decode Entities, Normalize URLs)
- **Decision:** Decoded HTML entities (`&gt;` $\rightarrow$ `>`), stripped leading routing handles (`@AppleSupport`, `@115858`), normalized URLs to `[URL]`, but strictly preserved emojis, Unicode variation selectors (`\ufe0f`), and raw punctuation.
- **Why:** `Settings > General > About` navigation paths are crucial for troubleshooting. Emojis and punctuation (all-caps, `???`) provide vital sentiment and frustration signals for escalation detection.
- **Trade-off:** Preserving emojis requires careful UTF-8 console and file encoding handling (`utf-8-sig`) on Windows environments.

---

### Decision 5: Weak Supervision (894 Examples) with Automated Quality Audit
- **Decision:** Labeled 894 candidate training messages using keyword and regex heuristics, and audited 100 randomly sampled examples against canonical definitions.
- **Why:** Allowed rapid generation of training data without manual labeling fatigue, achieving an audited 91.0% precision across intents.
- **Trade-off:** Weak rules induce systematic blind spots (e.g. queries mentioning "iTunes payment" assigned to `Apps & Storage` rather than `Apple ID`).

---

### Decision 6: Dedicated Golden Benchmark (214 Examples) with Clear Provenance
- **Decision:** The final 214-example evaluation set (`Dataset/processed/golden_set.csv`) was sampled at the conversation level, initially labeled with an automated/AI-assisted process, then manually reviewed and corrected to produce the final benchmark. Escalation labels were assigned as policy-derived evaluation targets (not observed customer outcomes). All 214 conversation IDs are strictly excluded from classifier training and retrieval indexing.
- **Why:** Guarantees zero train-test and retrieval leakage, ensuring evaluation reflects true generalization on messy customer inquiries.
- **Trade-off:** Smaller benchmark size (214 rows) means minority classes like `Other` (5 examples) have high variance in recall.

---

### Decision 7: Intent Classification via TF-IDF + Multinomial Logistic Regression
- **Decision:** Built the intent classifier using sublinear TF-IDF n-grams (1-2) and balanced L2-regularized Logistic Regression rather than a fine-tuned transformer or LLM.
- **Why:** Sub-second training, zero GPU requirement, line-by-line interpretability of feature weights, and well-calibrated softmax confidence scores for escalation thresholding.
- **Trade-off:** Lower semantic abstraction than deep models; cannot easily resolve complex negation or syntactic rephrasing without explicit n-gram overlap.

---

### Decision 8: Zero-Leakage Historical Retrieval via TF-IDF and Cosine Dot-Product
- **Decision:** Implemented historical reply retrieval using sparse TF-IDF and cosine similarity over 8,786 historical conversations, explicitly filtering out Golden Set IDs.
- **Why:** Technical support queries hinge on exact lexical tokens (e.g. "iOS 11.0.2", "iPhone 7", "error 3014", "AirPods"). Dense vector embeddings often suffer from semantic drift (matching two unrelated complaints solely on shared negative sentiment).
- **Trade-off:** Inability to retrieve relevant cases that use completely disjoint vocabulary for the same underlying issue.

---

### Decision 9: Multi-Case Evidence Retrieval (Top 3–5) Over Single-Case Retrieval
- **Decision:** Expanded retrieval context from top-1 to top 3–5 historical conversations for the generative agent.
- **Why:** In inquiries like AirPods bluetooth disconnects, the top-1 match is often an unhelpful canned DM link or a lost AirPod case. Presenting 3–5 cases allows the LLM to ignore irrelevant examples and synthesize comprehensive troubleshooting questions (device model, iOS version, onset).
- **Trade-off:** Increases prompt token length by ~200 tokens per inference call.

---

### Decision 10: Deterministic Pre-LLM Escalation Guardrails & Policy Alignment
- **Decision:** Hardcoded escalation routing rules (Apple ID credentials, physical hardware damage, critical crash loops, classifier confidence < 0.35) executed before calling the LLM.
- **Why:** LLMs are prone to sycophancy, hallucinating troubleshooting steps for hardware damage, or requesting passwords in chat. Escalation decisions must be 100% predictable, safe, and zero-cost.
- **Trade-off:** Because the Golden escalation labels and the agent's rules share the same underlying policy principles, the 62.15% adherence score measures policy adherence and is subject to cascading classifier errors, rather than measuring unbiased real-world escalation accuracy.

---

### Decision 11: Implement Two Non-Generative Baselines
- **Decision:** Evaluated Baseline 1 (Intent + Canned response) and Baseline 2 (Top-1 Retrieval verbatim) before building the LLM agent.
- **Why:** Establishes rigorous lower bounds. Demonstrates empirically why static macros fail (cascading intent misclassifications) and why raw retrieval fails (leaking customer names and dead links).
- **Trade-off:** Requires maintaining and running two baseline pipelines across the benchmark.

---

### Decision 12: Constrained LLM Rewriter Over Freeform Answer Generation
- **Decision:** Prompted the LLM strictly to adapt and sanitize historical AppleSupport evidence, explicitly prohibiting it from inventing policies, prices, timelines, or DIY repairs.
- **Why:** Customer support agents operate under legal and brand liability. Constrained synthesis eliminates hallucinations while fixing the verbatim transfer flaws of pure retrieval.
- **Trade-off:** If retrieval returns poor or unhelpful evidence, the LLM is constrained to give a cautious referral to official support rather than creatively guessing.

---

### Decision 13: Reject Validation Accuracy (91.06%) as Headline Performance
- **Decision:** Explicitly reported the 70.56% Golden Set accuracy as true system performance and documented the 20.5% drop from validation (91.06%).
- **Why:** Validation data shared the same heuristic keyword distributions as the weak-labeling rules, creating an artificially easy test. The Golden Set represents true organic generalization.
- **Trade-off:** Requires defending a 70.56% headline number rather than presenting an inflated 91% score in take-home evaluations.

---

### Decision 14: Sampled LLM-as-Judge Evaluation (25 Cases, ~33–35 Calls Max) with Persistent Caching
- **Decision:** Sampled exactly 25 stratified cases for LLM-as-Judge scoring (covering all 8 intents), restricted live agent generation to AUTO-HANDLE cases (at most 8–10 calls), consolidated the 3 candidate model evaluations into 1 prompt per case (25 judge calls), and cached all outputs locally.
- **Why:** Calling an LLM judge across all 214 cases for 3 models separately would require 642 API calls, risking rate-limit exhaustion and high cost. Restricting live generation to 8–10 AUTO-HANDLE cases and consolidating the judge to 1 call per sample case caps total project spend to ~33–35 calls maximum.
- **Trade-off:** Smaller sample size increases confidence interval width on judge metrics.

---

### Decision 15: Single Isolated OpenAI Provider with Graceful Local Fallback
- **Decision:** Implemented a single provider integration using `openai` with environment variable authentication (`OPENAI_API_KEY`) and an automated local fallback.
- **Why:** Avoids heavy agent frameworks (LangChain/LlamaIndex) and avoids multiple redundant SDKs. The entire deterministic pipeline (classification, retrieval, escalation, baselines) runs without requiring an API key.
- **Trade-off:** Live LLM response generation requires setting an environment key.
