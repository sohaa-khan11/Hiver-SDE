# Decision Log

These are the 15 non-obvious decisions I made while building this project, and why I made them. Written as plain bullet points, not corporate documentation.

---

**1. I picked AppleSupport, not the biggest brand by tweet count**

There are 108 brands in the dataset. I did not just pick the one with the most tweets. I filtered by what actually matters for a customer support agent: does the brand write replies with real troubleshooting steps? AppleSupport had 22% of replies containing concrete steps (settings paths, reset sequences, iOS version checks). Delta had 5% — mostly flight status. Spotify had useful replies but 74% of them were contaminated with employee initials like `/LS` that would poison any retrieval system. AppleSupport was the clear choice.

---

**2. I reconstructed full conversation threads, not isolated tweet pairs**

About 37% of AppleSupport interactions are multi-turn. A tweet that says "it still didn't work" means nothing in isolation. I traced every reply back to the root customer tweet using a graph traversal across the full 2.8 million row dataset. This took about 2 minutes to run but meant that every historical example I indexed had the full context of what the problem actually was.

---

**3. I defined 8 intent categories based on real complaints, not what seemed logical**

I could have guessed at categories ("billing," "technical," "general"). Instead I read through thousands of actual customer messages and grouped them by what resolution they needed. That is why "Keyboard" is its own category — not because keyboards are especially important, but because the iOS 11 autocorrect bug ("I" becoming "A [?]") generated a massive spike of almost identical complaints that needed a specific settings workaround. Lumping it into "Phone Performance" would have buried it.

---

**4. I kept emojis and punctuation in the text, even though it made encoding harder**

Most text preprocessing tutorials say "strip everything, lowercase it all." I did not do that here. "Settings > General > About" contains a navigation path that is meaningless if you remove the angle brackets. All-caps words and repeated exclamation marks signal frustration, which is a useful signal for deciding whether to escalate. Keeping emojis added encoding headaches on Windows (had to use `utf-8-sig` everywhere), but it was worth it.

---

**5. I used keyword-based weak supervision to label training data, and I audited it**

I did not manually label 894 training examples one by one. I wrote regex patterns like `"battery|drain|plugged in"` to automatically assign intent labels. This is called weak supervision. To check how accurate it was, I randomly sampled 100 labeled examples and verified each one against the intent definition. 91% were correctly labeled. That is the number I report for training data quality — not a gut feeling, an actual audit with a fixed random seed.

---

**6. I built a separate Golden Set and never touched it during development**

I set aside 214 conversations as a frozen evaluation benchmark before I trained anything. "Manually reviewed" means I went through the automated labels and corrected obvious errors — I did not label all 214 from scratch. All 214 conversation IDs are hard-excluded from the retrieval index so the agent cannot cheat by retrieving its own test cases. This is the one rule I did not bend.

---

**7. I used TF-IDF + Logistic Regression for the classifier, not a transformer**

This was a deliberate choice, not a limitation I had to apologise for. TF-IDF runs in milliseconds on CPU, needs no GPU, and produces well-calibrated softmax confidence scores that I can use directly as an escalation signal. A BERT-based classifier would likely get higher accuracy on this task but would require a GPU, take much longer to train, and would need additional calibration work to produce reliable confidence scores. The tradeoff was not worth it for a Twitter-length classification problem.

---

**8. I set the escalation confidence threshold at 0.35 before looking at the Golden Set**

The classifier outputs a softmax confidence score. I needed a threshold below which "I am not sure what this is, so send it to a human." I set 0.35 based on what felt like a reasonable safety floor for a support context — not by tuning it against the Golden Set. If I had tuned it on the Golden Set, I would be overfitting my safety policy to my own benchmark, which is not how real safety thresholds work. This decision means the threshold might not be optimal, but it is honest.

---

**9. Escalation runs before the LLM, not after**

Some systems generate a reply first and then decide whether to send it. I do the opposite: decide whether to escalate first, then generate. The LLM never sees account credential messages, cracked screen reports, or critical boot loop cases. This was non-negotiable — you cannot let a language model decide whether something is safe enough to handle autonomously. That decision has to be deterministic and explainable.

---

**10. I retrieve the top 4 historical cases, not just the best one**

The obvious design is: find the most similar past conversation, use that reply. It does not work. In practice, the top-1 result is often a tangentially related conversation — for AirPods disconnection queries, the top hit was sometimes a "lost AirPod" case whose only reply was "Please DM us." By retrieving 4 cases instead, the LLM can ignore the unhelpful one and synthesise the relevant diagnostic questions from the other three.

---

**11. I built two baselines before I built the final agent**

Baseline 1: map the predicted intent to a fixed canned reply. Baseline 2: return the verbatim text of the most similar historical reply. I ran both across all 214 Golden Set cases before I wrote a single line of the final agent. This was important because it forced me to understand what was actually failing — and the answer was clear: canned replies are not personalised, and verbatim retrieval copies previous customers' names and dead links. The final agent exists to fix those specific problems.

---

**12. I constrained the LLM to rewrite evidence, not answer from memory**

The LLM prompt does not say "answer this customer's question." It says "here are 4 real AppleSupport replies to similar problems — adapt them for this specific customer, and do not add any facts that are not in the evidence." If the retrieved evidence is unhelpful, the model is instructed to say so and point to official support rather than invent a solution. This is what keeps unsupported claims low.

---

**13. I used a local open-weight model instead of OpenAI**

I started with OpenAI (gpt-4o-mini) and replaced it with `phi3:mini` running locally via Ollama. The reasons: zero API cost, zero rate limits, no external credentials required, complete reproducibility, and the customer data stays on-device. The tradeoff is that phi3:mini sometimes leaks the prompt structure into its output — fixed with explicit stop tokens and a post-generation sanitisation function that strips anything that looks like a prompt header.

---

**14. I used the same local model as both generator and judge**

This is a problem I documented rather than pretended did not exist. Using `phi3:mini` to judge `phi3:mini`'s replies introduces self-grading bias — a model will tend to prefer output in its own style. I mitigated this with temperature=0 and a structured JSON scoring schema, but the bias cannot be eliminated without an independent judge. The evaluation section of the README flags this explicitly. If I had another week I would collect human ratings to validate the judge.

---

**15. I reported the 70.56% Golden Set accuracy as the headline, not the 91.06% validation accuracy**

The 91% number came from testing on data generated by the same keyword rules used for training. It is an artificially easy test. The 70.56% number came from real customer messages the system had never seen, with slang, typos, and multi-symptom complaints. That is the honest headline. Reporting 91% would be misleading even if it is technically accurate.

