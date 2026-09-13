# AppleSupport Customer Support Agent

**Hiver SDE Intern**

---

## What This Is

I built a small AI customer support agent for AppleSupport, using a real dataset of Twitter support conversations (the Kaggle TWCS dataset — about 2.8 million tweets from 108 different brands).

The system does three things when a customer message comes in:

1. **Figures out what the customer needs** (e.g., is it a battery problem, an Apple ID issue, a cracked screen?)
2. **Decides whether to handle it automatically or escalate to a human** (e.g., never let a bot deal with someone locked out of their account)
3. **Writes a reply grounded in how AppleSupport actually handled similar problems in the past** — so it does not make things up

---

## Why AppleSupport?

I compared the three biggest brands in the dataset before picking one.

| | AppleSupport | SpotifyCares | Delta |
|---|---|---|---|
| Total support replies in dataset | 106,860 | 43,265 | 42,253 |
| Conversations with real troubleshooting steps | 22.25% | 18.14% | 4.99% |
| Agent signature noise (e.g. `/LS`, `*QB`) | 0% | 74.68% | 82.30% |

Delta's replies were mostly "your flight is delayed, call this number." Spotify's replies were contaminated with employee initials that would pollute any retrieval system. AppleSupport had the richest technical content — real steps like "Go to Settings > General > About and check your iOS version" — and zero noise. That made it the best corpus to learn from and retrieve from.

---

## What "Good" Looks Like for AppleSupport

Before I wrote a single line of code, I thought about what a good AppleSupport reply actually looks like.

A good reply:
- Addresses the customer's specific problem, not a generic template
- Never asks for a password or account credentials in a public chat thread
- Does not guess at hardware repair costs or promise outcomes
- Is concise — AppleSupport replies on Twitter average about 30 words
- Gives concrete next steps (a settings path, a reset sequence, an official support link)

A bad reply:
- Copies a previous customer's name ("Hey Ryan, thanks for reaching out!")
- Refers the customer to a dead 2017 Apple support link
- Pretends it can fix a cracked screen with software advice

**What I chose NOT to build:**
- A fine-tuned language model (the dataset is real customer data; fine-tuning risks memorising private information)
- A retrieval system that just pastes the most similar old reply verbatim (this is Baseline 2, and it fails badly — see results)
- A rule-based chatbot with 50 handcrafted if/else branches (would collapse the moment a customer uses unexpected phrasing)
- Any dependency on paid cloud APIs (everything runs locally, no keys needed)

---

## How It Works

```
Customer message
       |
[Step 1] Intent classifier
         TF-IDF + Logistic Regression -> "Battery", "Apple ID", "Keyboard", etc.
       |
[Step 2] Historical retrieval
         Finds the top 4 most similar real AppleSupport conversations from the past
       |
[Step 3] Escalation check (runs BEFORE any LLM)
         Rule-based: account credentials? cracked hardware? boot loop? -> ESCALATE
         Classifier not confident (< 35%)? -> ESCALATE
         Everything else -> AUTO-HANDLE
       |
[Step 4] LLM reply writer (only for AUTO-HANDLE cases)
         Rewrites the retrieved evidence for this specific customer
         Runs locally via Ollama (phi3:mini, no internet required)
       |
Final reply
```

The escalation check happens **before** the LLM, not after. The LLM is never allowed to decide whether something is safe to handle — that is always a deterministic rule.

---

## The 8 Intent Categories

I went through roughly 9,000 conversation threads and grouped what customers actually complained about:

1. **Battery** — fast drain, sudden shutdowns, won't charge
2. **Phone Performance** — freezing, stuttering, random reboots
3. **Keyboard** — the iOS 11 autocorrect "I" to "A [?]" bug was a huge spike in complaints
4. **Apple ID** — forgotten passwords, lockouts, 2FA, mystery billing charges
5. **Sound & Bluetooth** — AirPods disconnecting, Bluetooth won't pair, speaker issues
6. **Screen & Camera** — cracked glass, black camera viewfinder, touch not working
7. **Apps & Storage** — app won't download, storage full, app crashing on open
8. **Other** — everything else: general venting, unrelated questions, social chatter

"Other" is 59% of raw incoming messages but only 2.3% of the Golden Set (the evaluation benchmark), because most "Other" messages do not need a support agent response at all.

---

## The Data Setup

I kept four things strictly separate to avoid cheating in the evaluation:

| Data | What it is | Used for |
|---|---|---|
| `apple_cleaned.csv` | 8,786 historical AppleSupport conversations | Retrieval only |
| `training_data.csv` | 894 labeled examples (keyword rules) | Training the classifier |
| `golden_set.csv` | 214 held-out examples (manually reviewed) | Evaluation only |
| Escalation labels | Policy-derived labels (not real observed outcomes) | Testing escalation rules |

The 214-example Golden Set was sampled at the conversation level, initially labeled using automated rules, then manually reviewed and corrected. All 214 conversation IDs are blocked from appearing in retrieval — so the agent can never "cheat" by finding its own test cases.

---

## Results vs. the Two Baselines

I ran everything on all 214 Golden Set conversations.

### Intent Classification

| Evaluated on | Accuracy | Macro F1 |
|---|---|---|
| Validation split (20% of training data) | 91.06% | 90.03% |
| Golden Set (real held-out benchmark) | 70.56% | 61.98% |

The 91% number is misleading — see the section below on that.

Per-intent F1 on the Golden Set:
`Keyboard` 91.8% -> `Battery` 83.0% -> `Sound & Bluetooth` 75.3% -> `Apps & Storage` 67.5% -> `Phone Performance` 62.2% -> `Apple ID` 58.8% -> `Screen & Camera` 54.2% -> `Other` 0.0%

`Other` scored 0% because it had only 5 examples in the benchmark and a logistic regression trained on keyword patterns cannot reliably catch "everything else."

### Escalation Policy

| | Accuracy | Macro F1 |
|---|---|---|
| Escalation adherence | 62.15% | 59.04% |

Out of 143 conversations that should have been escalated: 96 correctly escalated, 47 incorrectly auto-handled.
Out of 71 conversations that should have been auto-handled: 37 correctly handled, 34 over-escalated.

The escalation score is tied to the classifier — if the classifier calls an Apple ID issue "Apps & Storage," the escalation rule for account credentials never fires. So errors cascade.

### Reply Quality (all 214 cases, automated word-overlap proxy)

| System | Avg reply length | Handles escalation? | Personalised? |
|---|---|---|---|
| Baseline 1 — Canned reply | 27 words | No | No |
| Baseline 2 — Copy top retrieval hit | 23 words | No | No (copies old names) |
| **Final Agent** | **32 words** | **Yes (60.7% of cases)** | **Yes** |

Neither baseline handles escalation at all — they send a troubleshooting reply to everyone, including people with cracked screens or locked accounts. The final agent routes 130 out of 214 conversations to human agents with an explanation, and generates a grounded reply only for the remaining cases it can safely handle.

### LLM-as-Judge Results (25-case stratified sample, local phi3:mini)

The judge evaluated each reply on five criteria, scored 1-5:

| System | Groundedness | Relevance | Actionability | Tone | Unsupported claim rate |
|---|---|---|---|---|---|
| Baseline 1 — Canned | 3.44 | 3.52 | 4.20 | 4.56 | 0.00 |
| Baseline 2 — Retrieval | 3.88 | 4.28 | 3.96 | 4.56 | 0.04 |
| **Final Agent** | **4.36** | **4.40** | **4.48** | **4.84** | 0.04 |

The final agent scored highest on every dimension. The canned baseline had zero unsupported claims because it is a fixed template — but it also scored lowest on relevance and groundedness because the template does not adapt to the actual complaint.

**Important caveat:** The same local model (`phi3:mini`) that generated the agent's replies also judged them. A model tends to prefer output in its own style. Treat these scores as directional signals, not as a reliable quality measure. The human ratings below show how large the gap can be.

### Human vs LLM Judge Agreement (25 paired cases)

After completing the LLM judge run, the same 25 cases were rated manually on the Final Agent's replies:

| Criterion | LLM judge avg | Human avg | MAE |
|---|---|---|---|
| Groundedness | 4.36 | 2.76 | 1.760 |
| Relevance | 4.40 | 2.80 | 1.920 |
| Actionability | 4.48 | 2.72 | 1.840 |
| Tone | 4.84 | 4.36 | 0.720 |
| Unsupported claim exact agreement | — | — | 76.0% |

The MAE on groundedness, relevance, and actionability is close to 2 points on a 5-point scale. The LLM judge rated itself far higher than the human reviewer did on the same replies. Tone was the only dimension where there was reasonable agreement (MAE 0.72), which makes sense — tone is easier to assess from surface features than factual grounding is.

This confirms the self-grading bias documented above. The LLM judge results should be read as a ranking tool (which system did better relative to the others?) rather than as an absolute quality score.

---

## Top 5 Failure Modes

These are real cases from the Golden Set where things went wrong, with my best hypothesis for why.

**1. Keyboard misclassified as Screen & Camera (conversation 427841)**

The customer wrote: *"Hi I'm using ios 11.0.2 on iPhone 6, keyboard doesn't fill screen when device rotated on landscape? Please solve this."*

The model predicted `Screen & Camera`. Why? The message contains both "keyboard" and "screen ... landscape" — and the TF-IDF weight for "screen" is higher because screen complaints are more common in training. The actual issue is a keyboard layout bug. A model that understands syntax would read "landscape mode" as a modifier of "keyboard," not as an independent screen complaint.

**2. Apple ID misclassified as Apps & Storage (conversation 811638)**

The customer wrote: *"Hi trying to download a game on to my phone but keeps telling payment failed, can't sign in to itunes too."*

The model predicted `Apps & Storage`. The word "download" is strongly associated with app downloads in training data. But "payment failed" and "can't sign in to iTunes" are Apple ID signals. A bag-of-words model cannot figure out that the download problem is a symptom of an account problem. This one matters beyond accuracy — the escalation rule for account credentials only triggers on `Apple ID`, so this customer's billing issue gets auto-handled instead of being escalated.

**3. Screen & Camera misclassified as Phone Performance (conversation 2064248)**

The customer wrote: *"updated 11.0.2 and now my phone is freezing up and screen rotation sticking in landscape mode!!!!"*

Multiple symptoms in one message. "Freezing up" is a strong `Phone Performance` signal. "Screen rotation sticking" points more to `Screen & Camera`. The classifier has to pick one and it picks the more frequent one. There is no way for a single-label classifier to know which grievance the customer cares about more without understanding the whole sentence.

**4. Battery misclassified as Sound & Bluetooth (conversation 1028445)**

The customer wrote: *"my phone died and now ive plugged it in and its completely blackscreen and it wont even show the charging symbol HELP @AppleSupport."*

There is no word "battery" in this message. The training data used keywords like "battery," "drain," "charge" to label Battery examples. But "phone died," "won't show charging symbol," and "blackscreen after plugging in" are also battery failure signals — just described differently. The weak supervision rules did not cover these phrasings, so the classifier had nothing to latch onto.

**5. Retrieval drift for AirPods (resolved in the final agent)**

For an AirPods Bluetooth disconnection query, the top-1 retrieval result was a generic "Please DM us" response from a lost AirPod case — useless. If you only take the top-1 match, you generate a useless routing reply. The fix was to retrieve the top 4 historical cases, letting the LLM ignore the unhelpful match and synthesize the diagnostic questions from the other three. This is now working correctly in the final agent.

---

## What Is Misleading About My Headline Number?

The classifier validation accuracy is **91.06%**.

That number sounds good. It is not the number you should care about.

Here is why:

**The validation data was generated by the same keyword rules used for training.** I wrote patterns like `"battery|drain|plugged"` to label training examples as `Battery`. Then I trained on 80% of those, tested on 20%, and got 91% accuracy. Of course the model does well — it is being tested on examples constructed by the same rules it was trained on. It is like studying a practice test and then being given the same questions.

**The real test is the Golden Set: 70.56%.** That is a 20-point drop. The Golden Set contains real customer messages with slang, typos, emotional venting, and multi-symptom complaints that keyword patterns never anticipated.

Two more things that make the headline number misleading:

- **`Other` is invisible.** The `Other` class scored 0.0% F1 on the Golden Set. But because it is a minority class, it barely moves the average accuracy metric. A system that completely fails at "ambiguous messages" looks fine on paper.

- **Escalation errors are not independent.** The 70.56% classifier accuracy means roughly 30% of queries go into escalation with the wrong intent label. The 62.15% escalation adherence is partly a consequence of that — not a separate failure. Reporting them as two separate numbers makes both look more independent than they are.

---

## What I Would Do With One More Week

**Add semantic retrieval alongside keyword retrieval.**
Right now retrieval is purely lexical. A customer who writes "my phone won't turn on after the update" and one who writes "device stuck on black screen post-iOS install" describe the same problem but share few words. I would add a small sentence embedding model (like `all-MiniLM-L6-v2`, which runs on CPU in under 1 second) and combine its scores with TF-IDF using reciprocal rank fusion.

**Tune the escalation confidence threshold properly.**
The current threshold of 0.35 was set by intuition — "below 35% softmax confidence, escalate to be safe." I would do a proper ROC analysis: sweep the threshold from 0.1 to 0.9, measure false auto-handle rate (safety risk) versus false escalation rate (customer friction), and pick the threshold that minimises safety risk while keeping escalation volume manageable.

**Add multi-label intent support.**
Several failure cases involved compound complaints (battery problem and black screen, keyboard bug and screen layout). A multi-label classifier would let the agent say "this is primarily Battery, but also Screen & Camera" and adjust both retrieval and escalation accordingly.

**Validate the LLM judge with actual humans.**
Right now the judge is `phi3:mini` evaluating `phi3:mini`. I would get a few people to each score a random sample of replies across all three systems and compute Spearman correlation and Cohen's kappa against the LLM judge. If correlation is high, the automated judge is trustworthy. If not, I need a better judge or a different evaluation approach entirely.

---

## How to Run It

```bash
git clone https://github.com/sohaa-khan11/Hiver-SDE.git
cd Hiver-SDE
pip install -r requirements.txt
```

**The full deterministic pipeline (no Ollama needed):**
```bash
python src/models/train_intent_classifier.py   # train classifier, see Golden Set results
python src/agent/baseline_canned.py            # run canned-reply baseline
python src/agent/baseline_retrieval.py         # run retrieval-only baseline
python src/evaluation/evaluate_system.py       # full evaluation (escalation + proxy metrics)
```

**Live LLM generation and judge (requires Ollama):**
```bash
# Install Ollama from https://ollama.com, then:
ollama pull phi3:mini
python src/evaluation/evaluate_system.py
# Results cache to Dataset/processed/llm_generation_cache.csv and llm_judge_results.csv
# Re-runs load from cache instantly
```

**To regenerate everything from the raw Kaggle data:**
```bash
# Place twcs.csv at Dataset/twcs/twitter_support.csv (~493 MB, not tracked in git)
python src/data/extract_apple.py
python src/data/clean_apple.py
python src/data/create_labels.py
python src/data/audit_labels.py
```

---

## Repository Layout

```
Hiver/
├── README.md                            <- This file
├── requirements.txt                     <- pandas, numpy, scikit-learn (no paid APIs)
├── .gitignore
├── docs/
│   └── decision_log.md                  <- 15 plain-English decisions and why I made them
├── Dataset/
│   └── processed/
│       ├── golden_set.csv               <- 214-example evaluation benchmark (tracked)
│       └── training_data.csv            <- 894-example training set (tracked)
│       (other CSVs are generated at runtime and git-ignored)
└── src/
    ├── data/
    │   ├── extract_apple.py             <- pull AppleSupport threads from TWCS
    │   ├── clean_apple.py               <- strip noise, decode HTML, normalise URLs
    │   ├── create_labels.py             <- generate weak-supervised training labels
    │   └── audit_labels.py              <- verify label quality (91% precision check)
    ├── models/
    │   └── train_intent_classifier.py   <- TF-IDF + Logistic Regression classifier
    ├── retrieval/
    │   └── retrieve_similar.py          <- TF-IDF cosine similarity retrieval engine
    ├── agent/
    │   ├── baseline_canned.py           <- Baseline 1: static canned replies by intent
    │   ├── baseline_retrieval.py        <- Baseline 2: verbatim top-1 historical reply
    │   └── support_agent.py             <- Final agent: retrieval + escalation + LLM
    └── evaluation/
        └── evaluate_system.py           <- evaluation harness (metrics, judge, caching)
```
