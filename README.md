# AI Customer Support Assistant (Hiver SDE Take-Home)

An automated AI customer-support system designed to understand customer technical inquiries, retrieve grounded historical brand solutions, draft support replies, and determine whether to auto-handle or escalate to human agents.

---

## 1. Project Context & Dataset

* **Dataset:** [Customer Support on Twitter (TWCS)](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) containing ~2.81 million customer-service tweets.
* **Raw Data Location:** The raw dataset file must be downloaded and placed at:
  ```
  Dataset/twcs/twitter_support.csv
  ```
  *(Note: `Dataset/twcs/twitter_support.csv` is ~493 MB and ignored via `.gitignore` to prevent large commits).*

---

## 2. Selected Brand: AppleSupport

After an empirical evaluation of candidate brands (`AppleSupport`, `SpotifyCares`, and `Delta`), **`AppleSupport`** was selected as the optimal domain.

### Why AppleSupport Was Selected:
1. **Instructional Troubleshooting Evidence (22.25%):** Human agents provide concrete diagnostic workflows (settings navigation, OS checks, reset button sequences) needed to ground RAG retrieval.
2. **Clean Ground Truth (0% Agent Signatures):** Unlike Delta (82% `*QB`) or Spotify (74% `/LS`), Apple adheres to a unified corporate voice with zero employee initials.
3. **Intuitive Intent Taxonomy:** Real-world technical problems (battery drain, iOS update glitches, screen/hardware, audio/AirPods, Apple ID/passcodes).
4. **Natural Escalation Boundaries:** Clear division between informational troubleshooting (auto-handle) vs. hardware faults, persistent boot loops, and private credentials (escalate to human/DM).

*Detailed metrics and evidence are documented in [`docs/decision_log.md`](docs/decision_log.md).*

---

## 3. Current Data Pipeline

```
TWCS raw dataset (Dataset/twcs/twitter_support.csv)
       │
       ▼
AppleSupport filtering (106,860 brand replies)
       │
       ▼
Conversation reconstruction (Graph traversal to root customer tweet)
       │
       ▼
Reproducible conversation sampling (9,000 threads, seed=42)
       │
       ▼
AppleSupport working dataset (Dataset/processed/apple_conversations.csv)
       │
       ▼
Conservative data cleaning & preprocessing (src/data/clean_apple.py)
       │
       ▼
Clean AppleSupport dataset (Dataset/processed/apple_cleaned.csv)
       │
       ▼
Annotation candidate sampling (src/data/create_labels.py)
       │
       ▼
Candidate annotation dataset (Dataset/processed/label_candidates.csv)
```

The resulting cleaned dataset contains **9,000 complete conversation threads** (**26,129 tweets**, **7.62 MB**) with derived normalized `text_clean` while preserving original raw `text` and structural relationships. From this, a balanced candidate pool of **280 conversations** was sampled for Golden Set human annotation.

---

## 4. How to Run the Pipeline

### Step 1: Extract AppleSupport Conversations
```bash
python src/data/extract_apple.py
```
*(Optional flags: `--brand AppleSupport --sample-size 9000 --seed 42`)*

### Step 2: Clean & Preprocess Conversations
```bash
python src/data/clean_apple.py
```
*(Optional flags: `--input-path Dataset/processed/apple_conversations.csv --output-path Dataset/processed/apple_cleaned.csv`)*

### Step 3: Sample Annotation Candidates (for Golden Set)
```bash
python src/data/create_labels.py
```
*(Optional flags: `--input-path Dataset/processed/apple_cleaned.csv --output-path Dataset/processed/label_candidates.csv --seed 42`)*

*Human labeling instructions and category boundaries are documented in [`docs/decision_log.md`](docs/decision_log.md#human-annotation-guide-golden-set).*

---

## 5. Repository Structure

```
Hiver/
├── .gitignore
├── README.md
├── Dataset/
│   ├── twcs/
│   │   └── twitter_support.csv               # Raw dataset (downloaded locally, gitignored)
│   └── processed/
│       ├── apple_conversations.csv           # Extracted working dataset (26,129 rows)
│       ├── apple_cleaned.csv                 # Cleaned dataset with text_clean (26,129 rows)
│       └── label_candidates.csv              # Candidate pool for Golden Set review (280 rows)
├── docs/
│   └── decision_log.md                       # Brand selection, cleaning, & taxonomy decision record
└── src/
    └── data/
        ├── extract_apple.py                  # Conversation extraction script
        ├── clean_apple.py                    # Conservative text preprocessing script
        └── create_labels.py                  # Stratified candidate sampling script
```
