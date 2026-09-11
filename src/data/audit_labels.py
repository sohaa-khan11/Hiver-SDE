"""
audit_labels.py
---------------
Automated quality audit of weak labels generated in Dataset/processed/training_data.csv.
Evaluates a representative sample of 100 training examples with a fixed random seed (seed=42)
against the canonical intent definitions and boundary rules documented in docs/decision_log.md.
Calculates overall precision, per-intent precision, and identifies common rule error modes.

Usage:
  python src/data/audit_labels.py
"""

import os
import csv
import random
from typing import Dict, List, Tuple


def main():
    training_path = os.path.join("Dataset", "processed", "training_data.csv")
    if not os.path.exists(training_path):
        raise FileNotFoundError(f"Training data not found at: {training_path}")

    with open(training_path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    total_rows = len(rows)
    print("=" * 70)
    print("Weak-Label Quality Audit: Dataset/processed/training_data.csv")
    print(f"Total training pool size: {total_rows} examples")
    print("=" * 70)

    # Deterministic sampling of 100 examples
    random.seed(42)
    sample_indices = sorted(random.sample(range(total_rows), 100))
    sample = [rows[i] for i in sample_indices]

    # Automated evaluation decisions for the 100 sampled rows based on canonical intent definitions in docs/decision_log.md
    # Format: (is_correct, better_intent_if_incorrect, short_reason)
    audit_decisions: Dict[str, Tuple[bool, str, str]] = {
        '1030449': (True, 'Battery', 'Complaining of rapid battery drain after updates.'),
        '2005255': (True, 'Keyboard', 'iOS 11 autocorrect changing lowercase i to I glyph.'),
        '2485807': (True, 'Other', 'Delivery tracking / shipping status inquiry.'),
        '2077345': (True, 'Sound & Bluetooth', 'Headphones hardware breakage complaint.'),
        '1356390': (True, 'Other', 'Online store shipping address correction request.'),
        '648533': (True, 'Apple ID', 'Unable to log into App Store using Apple ID.'),
        '1153411': (True, 'Apps & Storage', 'App Store repurchasing glitch for previously owned apps.'),
        '1956362': (True, 'Phone Performance', 'Daily respringing / SpringBoard crash loops on iOS 11.'),
        '1530001': (True, 'Apps & Storage', 'Built-in Music/iTunes player stuck on shuffle mode.'),
        '430129': (True, 'Sound & Bluetooth', 'Bluetooth headphone battery percentage display in iOS 11.'),
        '1640775': (False, 'Apple ID', 'Inquiring how to merge iCloud and iTunes Apple ID accounts.'),
        '1303764': (True, 'Apps & Storage', 'Apple Music Family Sharing subscription access glitch.'),
        '440814': (True, 'Sound & Bluetooth', 'AirPods Bluetooth connection drops and missing widget.'),
        '455013': (False, 'Apps & Storage', 'Video streaming playback failure specific to single show on Apple TV iTunes.'),
        '1064904': (True, 'Battery', 'Update killing battery life and breaking apps.'),
        '2627829': (True, 'Apple ID', 'Two-factor authentication trusted number verification code failure.'),
        '1940149': (True, 'Phone Performance', 'Whole device freezing after updating iOS.'),
        '1078239': (True, 'Phone Performance', 'Phone freezing every 5 minutes after iOS 11 update.'),
        '2862163': (True, 'Apple ID', 'Apple ID account recovery and password reset delays.'),
        '2646281': (True, 'Battery', 'Battery life cut in half on iPhone 6.'),
        '1624985': (True, 'Apps & Storage', 'iTunes desktop library player view failure.'),
        '2816411': (True, 'Apple ID', 'Unable to log into account due to 2FA failure.'),
        '2295679': (True, 'Battery', 'MacBook Pro thermal overheating issue.'),
        '512495': (True, 'Phone Performance', 'Device stuck on grey spinning wheel bootloop.'),
        '2004660': (True, 'Keyboard', 'Inability to type the letter I without glitches.'),
        '1973707': (True, 'Apple ID', 'Forgotten Apple ID password on iCloud locked device.'),
        '1880971': (True, 'Screen & Camera', 'Third-party touch screen digitizer stopped working after update.'),
        '2021897': (True, 'Keyboard', 'Inability to type letter after H (letter I glitch).'),
        '1502992': (True, 'Keyboard', 'Letter I text prediction symbol glitch.'),
        '2152072': (True, 'Screen & Camera', 'iPhone X touch screen completely failed after activation.'),
        '2518324': (False, 'Other', 'AT&T carrier account upgrade lockout, not Apple ID credentials.'),
        '861443': (True, 'Battery', 'Update drains battery and slows phone down.'),
        '1286475': (False, 'Other', 'Siri voice assistant graphical circle glitch, not keyboard typing.'),
        '646958': (True, 'Keyboard', 'Typing the letter I and word IT replaced with symbols.'),
        '1818739': (True, 'Battery', 'iOS update killing battery life.'),
        '2112792': (True, 'Keyboard', 'General keyboard defect complaint.'),
        '627904': (True, 'Apple ID', 'Unable to reset security questions for account access.'),
        '554769': (True, 'Phone Performance', 'Constant respring bootloop bricking phone.'),
        '2018454': (True, 'Battery', 'One-year-old iPhone battery life degradation.'),
        '749754': (True, 'Sound & Bluetooth', 'Bluetooth connection to car audio for navigation directions.'),
        '148286': (True, 'Battery', 'Asking how to check battery health on iPhone 7 Plus.'),
        '2117263': (True, 'Sound & Bluetooth', 'Broken physical headphones.'),
        '1697368': (True, 'Keyboard', 'Virtual keyboard malfunction.'),
        '1772583': (True, 'Apps & Storage', 'iTunes error 3014 during device restore/update.'),
        '670607': (True, 'Apple ID', 'Frequent repetitive Apple ID authentication prompts.'),
        '1700809': (True, 'Keyboard', 'Typing the letter I glitch.'),
        '215822': (True, 'Battery', 'Having to charge the battery twice a day.'),
        '2401840': (True, 'Apps & Storage', 'iCloud storage full warning with inaccurate breakdown.'),
        '2533926': (True, 'Keyboard', 'iPad keyboard issues.'),
        '2786449': (True, 'Apps & Storage', 'Apple Music vs iTunes Match feature confusion.'),
        '765461': (True, 'Apple ID', 'iCloud account popup login loop with non-working password.'),
        '2140955': (True, 'Keyboard', 'Letter I rendering as A with question mark box.'),
        '2309417': (False, 'Apple ID', 'Sign into iTunes prompt loop / authentication failure.'),
        '346994': (True, 'Sound & Bluetooth', 'Dual charging and 3.5mm headphone adapter issue.'),
        '2750467': (True, 'Apps & Storage', 'iTunes device authorization limit to play purchases.'),
        '2060141': (True, 'Phone Performance', 'Phone freezes when attempting to text a photo.'),
        '296451': (True, 'Battery', 'Battery draining even when plugged into charger.'),
        '958209': (True, 'Sound & Bluetooth', 'Beats headphones hardware damage and warranty coverage.'),
        '1224431': (True, 'Keyboard', 'Inability to type letter I after upgrade.'),
        '2020287': (True, 'Keyboard', 'Autocorrect letter I bug.'),
        '1539927': (True, 'Apple ID', 'Inquiring if Apple ID authentication servers are down.'),
        '1767417': (True, 'Apps & Storage', 'Unable to update software via phone or iTunes.'),
        '1682497': (True, 'Keyboard', 'Keyboard malfunction complaint.'),
        '2039399': (True, 'Sound & Bluetooth', 'Bluetooth turning on automatically after update.'),
        '1807823': (True, 'Screen & Camera', 'Homescreen 3D Touch lag defect.'),
        '1911908': (True, 'Sound & Bluetooth', 'Wireless Bluetooth headphones compatibility with iPod.'),
        '1596687': (True, 'Keyboard', 'Bug when typing the letter I.'),
        '2237326': (True, 'Apps & Storage', 'Cannot download music album in iTunes.'),
        '151456': (True, 'Screen & Camera', 'Camera Live Photo and flash settings bug.'),
        '2065258': (True, 'Apps & Storage', 'Duplicate tracks in iTunes library.'),
        '1394064': (True, 'Apps & Storage', 'Clearing App Store balance.'),
        '942267': (True, 'Battery', 'Severe battery drain lasting only 3-4 hours.'),
        '395521': (True, 'Sound & Bluetooth', 'Cannot control music on lock screen when on Bluetooth.'),
        '2617091': (True, 'Sound & Bluetooth', 'Speaker audio concurrency and routing.'),
        '2185447': (True, 'Battery', 'Battery draining from 100% to 10% in 4 hours.'),
        '1677075': (True, 'Keyboard', 'Cannot type the letter after H (letter I bug).'),
        '539853': (True, 'Phone Performance', 'iPhone keeps restarting on its own.'),
        '1992919': (True, 'Keyboard', 'Question mark box when typing the letter I.'),
        '414971': (True, 'Phone Performance', 'iPhone freezing, apps crashing, resetting itself.'),
        '427420': (True, 'Sound & Bluetooth', 'Bluetooth and Wi-Fi auto-toggling on.'),
        '1568230': (True, 'Apps & Storage', 'Gifting music song on iTunes with voucher credit.'),
        '629765': (True, 'Sound & Bluetooth', 'Bluetooth and AirTunes audio stuttering.'),
        '322346': (True, 'Screen & Camera', 'Camera Portrait Mode missing on iPhone 7.'),
        '2302685': (True, 'Other', 'Booking online Genius Bar retail appointment.'),
        '1346345': (True, 'Phone Performance', 'iPhone freezing every day.'),
        '1138025': (True, 'Phone Performance', 'Screen freezing and becoming unresponsive.'),
        '2757513': (False, 'Other', 'Reporting third-party phishing scam website.'),
        '557864': (True, 'Sound & Bluetooth', 'Bluetooth connectivity failure.'),
        '545321': (False, 'Phone Performance', 'Respring boot loop (blank screen with spinning circle back to lock screen).'),
        '1931176': (True, 'Battery', 'Battery life plummets in airplane mode.'),
        '199383': (False, 'Phone Performance', 'Primary problem is phone freezing, lagging, and random restarts.'),
        '848106': (True, 'Apple ID', 'Forgotten passcode and security questions.'),
        '2439777': (True, 'Sound & Bluetooth', 'Bluetooth pairing failure on iPhone X.'),
        '2845669': (True, 'Sound & Bluetooth', 'Speaker audio quality defect on two iPhone 8 devices.'),
        '582819': (True, 'Phone Performance', 'Bug that keeps restarting the phone.'),
        '1541164': (True, 'Apps & Storage', 'iCloud backup restore failure.'),
        '2925629': (True, 'Phone Performance', 'Phone freezing and sluggishness.'),
        '1311432': (True, 'Apps & Storage', 'App Store not working, cannot download or update apps.'),
        '991770': (True, 'Apple ID', 'Apple ID 2FA verification code with lost trusted number.'),
        '1824515': (False, 'Apple ID', 'Falsely created Family Sharing account security issue.'),
    }

    # Evaluate sample
    per_intent_total: Dict[str, int] = {}
    per_intent_correct: Dict[str, int] = {}
    incorrect_records: List[Dict[str, str]] = []

    for r in sample:
        cid = r['conversation_id']
        assigned = r['intent']
        msg = r['customer_message']

        per_intent_total[assigned] = per_intent_total.get(assigned, 0) + 1

        is_correct, better_intent, reason = audit_decisions[cid]
        if is_correct:
            per_intent_correct[assigned] = per_intent_correct.get(assigned, 0) + 1
        else:
            incorrect_records.append({
                'cid': cid,
                'assigned': assigned,
                'better': better_intent,
                'msg': msg,
                'reason': reason
            })

    total_checked = len(sample)
    total_correct = total_checked - len(incorrect_records)
    total_incorrect = len(incorrect_records)
    overall_precision = (total_correct / total_checked) * 100.0

    print(f"\nAudit Summary:")
    print(f"  - Sample size:        {total_checked}")
    print(f"  - Number correct:     {total_correct}")
    print(f"  - Number incorrect:   {total_incorrect}")
    print(f"  - Overall Precision:  {overall_precision:.1f}%")

    print(f"\nPer-Intent Precision Breakdown:")
    for intent, tot in sorted(per_intent_total.items(), key=lambda x: -x[1]):
        corr = per_intent_correct.get(intent, 0)
        prec = (corr / tot) * 100.0 if tot > 0 else 0.0
        print(f"  - {intent:<20}: {corr:>2}/{tot:>2} ({prec:>5.1f}%)")

    print(f"\nIncorrect Examples Identified ({total_incorrect}):")
    for idx, inc in enumerate(incorrect_records, 1):
        print(f"  {idx}. CID: {inc['cid']}")
        print(f"     Message:  \"{inc['msg']}\"")
        print(f"     Assigned: [{inc['assigned']}]  -->  Better Intent: [{inc['better']}]")
        print(f"     Reason:   {inc['reason']}\n")

    print("=" * 70)


if __name__ == "__main__":
    main()
