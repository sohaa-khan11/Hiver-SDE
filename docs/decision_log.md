# Decision Log: Brand Selection for Customer Support AI

## Context
For the Hiver SDE Intern take-home assignment, we are building an AI customer-support assistant using the Customer Support on Twitter (TWCS) dataset (`Dataset/twcs/twitter_support.csv`, 2.81 million tweets across 108 brands).

The assignment requires selecting **one primary brand** to focus on. We evaluated the three strongest candidate brands:
1. **AppleSupport**
2. **SpotifyCares**
3. **Delta**

---

## Empirical Comparison Summary

All metrics below were computed directly from `Dataset/twcs/twitter_support.csv`:

| Evaluation Metric | AppleSupport | SpotifyCares | Delta |
| :--- | :---: | :---: | :---: |
| **Total Brand Outbound Tweets** | **106,860** | 43,265 | 42,253 |
| **Customer Mentions / Queries** | **97,895** | 31,353 | 44,836 |
| **Valid Root Conversations** | **80,250** | 26,940 | 28,485 |
| **Parent Link Resolution Rate** | **99.93%** | 99.91% | 99.87% |
| **Multi-Turn Rate ($\ge 3$ turns)** | 36.82% | **39.31%** | 35.48% |
| **Troubleshooting Steps in Replies** | **22.25%** | 18.14% | 4.99% |
| **Agent Signature Noise** | **0.00%** (Clean) | 74.68% (`/LS`, `/MU`) | 82.30% (`*QB`, `*ALS`) |
| **Direct Message (DM) Routing** | 46.92% | 31.35% | 15.17% |
| **Customer Frustration Cues** | **14.93%** | 5.07% | 5.80% |
| **Language Simplicity** | > 99.9% English | > 99.9% English | > 99.9% English |

---

## Why AppleSupport Was Selected

1. **Rich Technical Resolution Evidence for RAG:**
   In 22.25% of replies, Apple Support agents provide actionable troubleshooting steps (navigating settings, checking OS versions, performing button-combination resets, toggling background notifications). In contrast, Delta provides instructional troubleshooting in only 4.99% of replies because airline support is largely operational (looking up PNRs and booking tickets).
2. **Cleanest Ground Truth Responses (Zero Agent Signatures):**
   Apple adheres to a strict unified brand voice with **0.00% agent initials or sign-offs**. Spotify has 74.68% employee slash signatures (`/LS`, `/BH`) and Delta has 82.30% asterisk signatures (`*QB`, `*ALS`). Using AppleSupport avoids contaminating response generation with human operator codes.
3. **Intuitive, Diverse Intent Taxonomy:**
   Customer queries span distinct, well-defined technical problem categories:
   - iOS Updates & Freezing / Boot Loops
   - Battery Health & Charging
   - Display, Touch Screen, & Hardware
   - Audio & Bluetooth (AirPods, Apple TV)
   - Apple ID, iCloud, & Account Security
   - Native Apps & Storage Management
4. **Natural Auto-Handle vs. Escalate Boundaries:**
   - **Auto-Handle:** Diagnostic advice, settings verification, software bug workarounds.
   - **Escalate:** Hardware defects, physical damage, persistent bootloops, forgotten Apple ID credentials, or queries requiring Genius Bar appointments or private DMs.
5. **Universal Explainability for Live Interview:**
   Every software engineer and interviewer understands iPhone issues and iOS updates. Explaining intent classification and RAG for Apple Support is clear, intuitive, and technically credible.

---

## Known Limitations and Mitigations

* **Masked Help Article URLs (`https://t.co/...`):**
  Apple replies frequently include links to Apple Support Knowledge Base articles. The actual page text is not in TWCS.
  * *Mitigation:* Normalize URLs to `[Apple Support Link]` so the retrieval model matches on the human agent's surrounding explanation.
* **High DM Routing (46.92%):**
  Apple routes queries to Direct Messages when personal account details or device serial numbers are needed.
  * *Mitigation:* We use DM requests as a key engineered signal for human escalation.

---

## Data Cleaning & Preprocessing Decisions

We follow a conservative, evidence-driven preprocessing philosophy on the extracted AppleSupport conversations.

### What Is Cleaned (Stored in `text_clean`):
1. **HTML Entities Decoded (`1,085` rows, 4.15%):**
   Decoded using standard library `html.unescape`. Apple replies heavily use `&gt;` for navigation menus (`Settings &gt; General &gt; About` $\rightarrow$ `Settings > General > About`) and `&amp;` for conjunctions.
2. **Leading Routing Handles Stripped:**
   - Outbound: 99.94% of brand replies begin with `@<customer_id>`. Stripped leading `@\d+\s+` to prevent AI response generation from hallucinating 2017 user IDs.
   - Inbound: 54.56% of customer tweets open with `@AppleSupport`. Stripped leading `@AppleSupport\s+` because in a single-brand setup, the target handle adds 0 intent signal.
   - *Mid-sentence mentions (`@Spotify`, `@Verizon`) are strictly preserved.*
3. **URL Normalization to `[URL]` (`10,858` rows, 41.56%):**
   Shortened `https://t.co/...` links are normalized to `[URL]`. 75% of brand replies and 14% of customer tweets include links. Replacing raw random hashes with `[URL]` prevents vocabulary pollution in RAG while preserving the signal that an external resource or screenshot was provided.
4. **Whitespace Collapsing (`1,465` rows, 5.61%):**
   Collapsed consecutive spaces and raw newlines into a single clean space.

### What Was Intentionally Preserved:
1. **Emojis & Unicode Characters (`U+FE0F` Variation Selectors):**
   Preserved intact. Following our engineering review, we explicitly removed the rule stripping `\ufe0f`. `\ufe0f` is part of standard multi-byte emoji sequences (`❤️`, `☹️`, `✌️`, `‼️`) as well as the authentic text of iOS 11 keyboard glitch reports. Removing it was unnecessary preprocessing; modern tokenizers and Python handle it natively.
2. **Capitalization & Punctuation:**
   Preserved intact. Shouting (all-caps) and excessive punctuation (`???`, `!!`) are key features for escalation detection.
3. **Typos & Informal Slang:**
   Preserved intact. Real customer language (`pls help`, `bricked`, `smh`, `sort your shit`) is naturally understood by modern embeddings and LLMs.
4. **Short Messages (33 rows $\le 15$ chars):**
   Preserved intact. All 33 are legitimate conversational replies (`turn_index > 0`, e.g. `11.1.1`, `Done`, `That fixed it`).
5. **Duplicate Text Strings (59 unique texts):**
   Preserved in conversation dataset because they represent real conversational acknowledgments and standard customer replies.
6. **Original Text Column:**
   The raw `text` column is preserved 100% untouched alongside `text_clean` for auditability and ground truth comparison.

### File Encoding & Excel Compatibility (`utf-8-sig`):
* **Investigation:** When opening `apple_cleaned.csv` in Excel on Windows, curly apostrophes (`’` / `U+2019`) were displayed as `â€™`. Python string inspection confirmed that our cleaning code did NOT corrupt the text—in Python memory and on disk, the text contained valid 3-byte UTF-8 (`0xE2 0x80 0x99`). Excel on Windows defaults to reading CSV files in ANSI/Windows-1252 unless a Byte Order Mark (BOM) is present.
* **Resolution:** The output CSV is saved with `encoding="utf-8-sig"`. This prepends the standard UTF-8 BOM (`0xEF 0xBB 0xBF`), instructing Excel to open the file in UTF-8 natively while remaining 100% transparent and compatible with Python `pd.read_csv()`.

---

## Customer Intent Discovery & Support Taxonomy

We analyzed the opening customer messages (`turn_index == 0`) across all 9,000 AppleSupport conversations in `Dataset/processed/apple_cleaned.csv` to establish natural, non-academic support categories grounded in actual customer complaints.

### Intent Taxonomy:
1. **Battery (~11.2%):** Battery drain, fast discharge, unexpected shutdowns, charging failures, cable/adapter issues.
2. **Phone Performance (~8.4%):** System freezing, UI lag/stutter, random reboots, boot loops, slow responsiveness.
3. **Keyboard (~8.6%):** iOS 11 `"I"` autocorrect bug (`"A [?]"`), predictive text glitches, symbol substitutions, stuck virtual keys.
4. **Apple ID (~4.6%):** Forgotten passwords, account lockouts, two-factor authentication, iCloud sign-in, credential verification.
5. **Sound & Bluetooth (~3.4%):** AirPods disconnects, Bluetooth pairing failures, low call volume, speaker/microphone distortion.
6. **Screen & Camera (~5.8%):** Black camera screen, flashlight failure, unresponsive touch digitizer, cracked screen.
7. **Apps & Storage (~4.6%):** App Store downloading/updating loops, iCloud/device storage full warnings, app crashes.
8. **Other (~59.0%):** Unactionable rants, vague complaints without symptoms, general product feedback, shipping/order inquiries.

### Key Boundary & Taxonomy Decisions:
* **Symptom First, Context Second:** Customers frequently mention `"since the update..."` alongside their symptoms. We classify by the actionable symptom (e.g., battery drain $\rightarrow$ Battery; freezing $\rightarrow$ Phone Performance) rather than creating an artificial "Update" bucket.
* **Keep Keyboard Distinct:** Although keyboard lag could technically relate to performance, the iOS 11 autocorrect bug represents 8.6% of the dataset and requires a unique resolution playbook (Settings > General > Keyboard > Text Replacement), so it is kept separate.
* **Combine Sound & Bluetooth:** Bluetooth complaints in this dataset are overwhelmingly about audio peripherals (AirPods, headphones, car audio). Keeping them in a unified `Sound & Bluetooth` category prevents artificial boundary confusion.
* **Natural Naming:** Category names are short, natural support queue names (avoiding AI jargon like `HARDWARE_POWER_BATTERY` or `SYSTEM_UPDATE_STABILITY`).

---

## Human Annotation Guide (Golden Set)

This guide defines the manual labeling rules for reviewing `Dataset/processed/label_candidates.csv` to construct our verified Golden Set (150–250 examples).

### Intent Category Definitions:
* **Battery:** Battery drain, charging problems, battery dying unexpectedly, battery health.
* **Phone Performance:** Phone/system freezing, crashing, lagging, rebooting, boot loops, or the whole device becoming unresponsive.
* **Keyboard:** Typing, autocorrect, keyboard, predictive text, or character/symbol problems.
* **Apple ID:** Apple ID, iCloud account, password, login, verification, account lockout, or similar account-access problems.
* **Sound & Bluetooth:** AirPods, Bluetooth connection, speakers, microphones, headphones, call/audio volume, or other sound problems.
* **Screen & Camera:** Screen/display/touch problems, cracked display, camera problems, flashlight/display-related hardware symptoms.
* **Apps & Storage:** App Store, downloading/updating apps, individual app problems, or storage/space problems.
* **Other:** The customer's actual problem cannot confidently be placed into one of the above categories, including vague complaints, feature feedback, social chatter, or unrelated requests.

### Core Labeling Rules:
1. **Label based on the customer's MAIN problem, not merely a keyword:**
   * *"After updating iOS 11 my battery dies in two hours"* $\rightarrow$ **Battery**
   * *"After updating iOS 11 my phone keeps freezing"* $\rightarrow$ **Phone Performance**
2. **Do not use "Other" just because the message is short:**
   * Use the provided `context` column when it clearly reveals the underlying issue.
3. **Use "Other" when context remains genuinely ambiguous:**
   * If the conversation still does not provide enough information to identify the technical problem, label as **Other**.
4. **Select the primary grievance when two issues are present:**
   * If multiple symptoms are mentioned, choose the issue that appears to be the customer's primary reason for contacting Apple.
5. **Strict taxonomy adherence:**
   * Do not invent an intent outside the eight categories.
6. **Data integrity:**
   * Do not modify `customer_message` or `context`.
7. **Exact category casing:**
   * Keep the `intent` values exactly as named: `Battery`, `Phone Performance`, `Keyboard`, `Apple ID`, `Sound & Bluetooth`, `Screen & Camera`, `Apps & Storage`, or `Other`.

---

## Weak-Label Quality Audit (Training Set)

Prior to training the baseline classifier on `Dataset/processed/training_data.csv` (894 weak-labeled examples), a quality audit was conducted to measure label precision and diagnose systemic rule error modes.

- **Sample Size:** 100 examples
- **Sampling Seed:** 42 (deterministic, reproducible sample across the 894 candidate rows)
- **Audit Method:** Automated evaluation of the 100 sampled messages against the canonical intent definitions and boundary rules established in this decision log.
- **Overall Precision:** 91.0% (91 correct, 9 incorrect)

### Per-Intent Precision Breakdown:
* **Battery:** 100.0% (13/13)
* **Other:** 100.0% (3/3)
* **Keyboard:** 94.1% (16/17)
* **Sound & Bluetooth:** 94.1% (16/17)
* **Phone Performance:** 92.3% (12/13)
* **Apple ID:** 91.7% (11/12)
* **Screen & Camera:** 83.3% (5/6)
* **Apps & Storage:** 78.9% (15/19)

### Main Error Patterns Identified:
1. **iTunes Account vs. App Ambiguity:** Queries mentioning "sign into iTunes" or "merge iTunes accounts" triggered `Apps & Storage` instead of `Apple ID` due to generic iTunes keywords.
2. **SpringBoard Resprings:** "Blank screen with spinning circle" was labeled `Screen & Camera` instead of `Phone Performance`.
3. **App/Service Playback vs. Device Issue:** Specific TV show streaming stalls on Apple TV were labeled `Phone Performance` instead of `Apps & Storage`.
4. **Third-Party / Carrier Ambiguity:** Carrier upgrade lockout (AT&T) was captured by `Apple ID` rules instead of `Other`.
5. **Multi-Issue Mentions:** Tweets listing multiple device symptoms (e.g., freezing, crashing, and AirPods) were occasionally caught by secondary peripheral keywords.

### Decision:
The weak labels demonstrate a **91.0% precision**, which substantially exceeds typical weak-supervision benchmarks (~80%). The training data is of high quality and ready to proceed with featurization and baseline classification (TF-IDF + Logistic Regression) without manual alterations or rule overfitting.

---

## Baseline Modeling Decision: TF-IDF + Multinomial Logistic Regression

### 1. Why TF-IDF + Logistic Regression as the Initial Model?
1. **Explainability & Line-by-Line Interpretability:**
   - Feature weights can be directly printed and audited. If an inquiry is misclassified, we can instantly inspect which n-grams contributed the most log-odds to the decision.
   - Ideal for production support engineering and interview defense: establishes an empirical lower bound before introducing heavier architectures.
2. **Deterministic, Millisecond Training:**
   - Trains in under a second on standard CPU with zero GPU dependencies or external API overhead.
3. **Calibrated Confidence for Escalation:**
   - Multinomial Logistic Regression produces well-calibrated softmax posterior probabilities ($P(y | x)$), making it naturally suited for setting downstream confidence thresholds to trigger human escalation.
4. **Resilience to Weak Label Noise:**
   - Standard L2 regularization ($C=1.0$) prevents the model from overfitting to idiosyncratic phrasing or minor errors in the weakly supervised training labels.

### 2. Hyperparameter Choices & Rationale:
* **`ngram_range=(1, 2)`:** Captures both unigrams (`battery`, `airpods`, `passcode`) and critical compound phrases (`spinning wheel`, `battery drain`, `sign in`, `touch screen`).
* **`min_df=2`:** Discards single-occurrence typos, garbled text, and isolated handles.
* **`max_features=5000`:** Bounds vocabulary dimensionality to prevent sparse memory bloat.
* **`sublinear_tf=True`:** Uses sublinear term frequency scaling ($1 + \log(\text{tf})$) so repeated rant words do not disproportionately dominate the document vector.
* **`class_weight='balanced'`:** Penalizes mistakes inversely proportional to class frequencies, ensuring minority classes are not suppressed.

### 3. Empirical Performance Summary:
* **Validation Split (20% holdout, 179 samples):**
  - Accuracy: **91.06%**
  - Macro F1: **90.03%**
* **Golden Evaluation Set (Held-Out Ground Truth, 214 samples):**
  - Accuracy: **70.56%** (151 / 214 correct)
  - Macro F1: **61.98%**
  - Strong performers: `Keyboard` (91.8% F1), `Battery` (83.0% F1), `Sound & Bluetooth` (75.3% F1), `Apps & Storage` (67.5% F1).
  - Main challenge: The open-ended `Other` category (5 support in Golden Set) received 0 predictions due to training scarcity (only 14 examples in training pool), pulling down the unweighted macro average.

---

## Retrieval Engine Decision: TF-IDF + Cosine Similarity

To ground automated reply generation and escalation decisions in historical brand behavior, a retrieval engine was developed to find the most relevant past customer conversations and official support responses.

### Why TF-IDF + Cosine Similarity Over Embeddings / Vector Databases?
1. **Lexical Precision for Technical Diagnostics:**
   - Technical support inquiries rely heavily on exact identifiers: OS versions (`iOS 11.1`), product models (`iPhone 7 Plus`, `AirPods`), specific error strings (`error 3014`), and distinct symptoms (`spinning wheel`).
   - Dense neural embeddings often exhibit semantic drift, falsely scoring unrelated hardware issues as similar simply because both tweets convey frustrated support sentiment. TF-IDF strictly rewards exact technical term overlap.
2. **Deterministic & Ultra-Fast Execution:**
   - Indexing 8,786 conversations takes ~0.5 seconds on a single CPU core.
   - Query retrieval runs in ~5 milliseconds via sparse dot product (`cosine_similarity`).
   - Zero infrastructure bloat: no external vector database servers, no Docker containers, and no recurring API embedding charges.
3. **Line-by-Line Explainability:**
   - Similarity scores can be directly audited by inspecting shared n-gram dot products. In an interview or production post-mortem, every retrieved reply can be justified deterministically.
4. **Guaranteed Zero Data Leakage:**
   - The retrieval corpus programmatically filters out all 214 Golden Set conversation IDs, ensuring that downstream evaluation of RAG responses remains strictly blind to test set conversations.






