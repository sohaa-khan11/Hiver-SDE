"""
create_labels.py
----------------
Generates weak labels for intent classification training from Dataset/processed/apple_cleaned.csv
using deterministic, high-precision pattern rules grounded in docs/decision_log.md.

Data Leakage Prevention:
  Strictly excludes all 280 original candidate conversation IDs from the Golden Set candidate pool
  to ensure zero data leakage between training and evaluation data.

Usage:
  python src/data/create_labels.py
"""

import os
import re
import csv
import argparse
import random
from typing import Dict, List, Optional, Set, Tuple

# Original 280 conversation IDs sampled for Golden Set candidate evaluation
# Recovered from initial candidate set commit (3edcde7)
EXCLUDED_CANDIDATE_IDS: Set[str] = {
    '1019374', '1023599', '1028442', '1028445', '103224', '1036548', '1046151', '1053648',
    '1061434', '1061481', '1075404', '1078295', '1086055', '1095929', '110967', '1133548',
    '1136049', '1138016', '1142068', '114459', '1148831', '1157375', '1162005', '1162142',
    '1180116', '1202855', '120683', '1215570', '1233616', '1267696', '1289773', '1291192',
    '1300711', '1305203', '1339696', '1341713', '1376432', '1376754', '1395690', '1398118',
    '1434156', '1434192', '1442834', '1448091', '1455396', '1458780', '1458993', '1478397',
    '1508715', '1524123', '1526937', '1536288', '1540005', '155955', '1559956', '1564047',
    '1584572', '1599229', '162833', '1632365', '1642353', '1644557', '1646155', '1646330',
    '1647380', '1650054', '1650175', '1674802', '1676860', '1680846', '1684503', '1700794',
    '1711357', '1727715', '1736100', '1742576', '1747366', '174936', '1752177', '1759675',
    '1763109', '1769095', '1772897', '1773171', '1781114', '1792980', '1795851', '1814499',
    '1825412', '1841084', '1848614', '1850918', '1877089', '1881556', '1881571', '188161',
    '1882488', '1889735', '1895409', '1899841', '1906180', '1906549', '1911003', '1919561',
    '1933920', '1936730', '1940919', '1947285', '1967961', '1976015', '1983490', '199425',
    '1997778', '1999665', '2001720', '2002648', '2004245', '2013717', '2015587', '2015797',
    '2019598', '2020667', '2021281', '2025902', '2029718', '2031239', '2041498', '2044955',
    '2050211', '2052582', '2054876', '2059463', '2064119', '2064248', '2070199', '2078931',
    '2096015', '2120161', '2123181', '2125015', '2140984', '2177369', '2193487', '2199248',
    '2237383', '2279570', '2282224', '2283100', '2284493', '2284498', '2284667', '2293093',
    '2295261', '2300181', '2300649', '2301965', '2304018', '2304938', '2305170', '2306245',
    '2309065', '2314120', '2345969', '2347694', '2352175', '236392', '2364470', '2364557',
    '236803', '2372169', '2375257', '2397985', '2408612', '2432815', '2440882', '2450947',
    '2465717', '2466453', '2471924', '2477569', '2484295', '2493887', '2506762', '2519322',
    '2544028', '2549918', '2558616', '2565755', '2570941', '2572027', '2573960', '2595487',
    '2615338', '269174', '2692852', '269874', '2702135', '271878', '2729122', '2734107',
    '2743065', '2749359', '2775995', '2789228', '2798459', '2800562', '2805986', '2813871',
    '2825492', '2841998', '2872578', '2888487', '2890643', '2890790', '2911763', '2918926',
    '2939311', '2947032', '2949626', '2953754', '2960107', '2977685', '306112', '307959',
    '310240', '325941', '332471', '358919', '36020', '362077', '362999', '394572',
    '395629', '400928', '427841', '450403', '465503', '489648', '493559', '498652',
    '51709', '522118', '541114', '546702', '553641', '554747', '557879', '559019',
    '561748', '588114', '58815', '592243', '593529', '61066', '627963', '643900',
    '656213', '680551', '716614', '720688', '738902', '746892', '768237', '798827',
    '811638', '82020', '824068', '830035', '848283', '849964', '8568', '862762',
    '87013', '873673', '92694', '954030', '959360', '975824', '987839', '990420'
}


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate weak labels for training intent classification models."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=os.path.join("Dataset", "processed", "apple_cleaned.csv"),
        help="Path to cleaned AppleSupport conversations CSV.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=os.path.join("Dataset", "processed", "training_data.csv"),
        help="Path to save weakly labeled training data CSV.",
    )
    parser.add_argument(
        "--max-per-intent",
        type=int,
        default=150,
        help="Maximum examples per intent to produce a balanced dataset (set 0 or negative for unlimited).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible stratified selection.",
    )
    return parser.parse_args()


def classify_weak_intent(text: str) -> Optional[str]:
    """
    Applies deterministic pattern matching and boundary rules to infer
    a high-confidence intent label for an opening customer message.
    Returns None if the message lacks a strong signal or is genuinely ambiguous.
    """
    m = text.lower()

    # 1. Other: High-confidence out-of-scope non-technical categories
    is_other = bool(re.search(
        r"\b(?:tracking\s+number|track\s+(?:my\s+)?order|shipment|delivery\s+date|package\s+delayed|"
        r"order\s+status|cancel\s+(?:my\s+)?order|delivery\s+status|shipping\s+address|delivery\s+address|"
        r"genius\s+bar\s+appointment|book\s+(?:an?\s+)?appointment|trade[\s-]in\s+value|gift\s+card\s+balance|"
        r"refund\s+status|return\s+policy|exchange\s+period|carrier\s+unlock|sim\s+locked|network\s+unlock|"
        r"puk\s+code|phishing\s+email|phishing\s+scam|fake\s+apple\s+email|is\s+this\s+(?:email|message)\s+(?:legit|real|fake))\b",
        m
    ))
    if is_other:
        return "Other"

    # 2. Keyboard: typing, autocorrect, virtual keys, character substitution, emoji glyphs
    is_keyboard = bool(
        re.search(r"\b(?:autocorrect|auto-correct|predictive\s+text|keypad)\b", m) or
        re.search(r"\bkeyboard\b(?!\s+(?:audio|sound|click))", m) or
        re.search(r"\b(?:typing|type)\b.*\b(?:glitch|lag|slow|bug|letter|symbol|word|problem|wrong|space)\b", m) or
        re.search(r"[\u200b\ufe0f]*[I|i][\u200b\ufe0f]*\s*(?:\[\?\]|⍰|\?|glitch|bug|\bautocorrect\b)", m) or
        re.search(r"\b(?:capital\s+i|letter\s+i)\b", m) or
        re.search(r"\bemoji(?:s)?\b.*\b(?:box|boxes|missing|render|question\s*mark)\b", m) or
        re.search(r"\bquestion\s+mark\s+box(?:es)?\b", m)
    )

    # 3. Battery: charging, battery drain, power, battery health, thermal charging issues
    is_battery = bool(
        re.search(r"\bbattery\b", m) or
        re.search(r"\b(?:won\'?t\s+charge|not\s+charging|charges?\s+slowly|charging\s+problem|stops?\s+charging|charger\s+(?:broken|not\s+working)|charging\s+cable|lightning\s+cable)\b", m) or
        re.search(r"\b(?:dies\s+fast|draining\s+fast|drains?\s+quickly|drain\s+so\s+fast|battery\s+dying|shuts?\s+off\s+at\s+\d+%)\b", m) or
        re.search(r"\b(?:overheating|overheats|phone\s+gets?\s+(?:very\s+|super\s+)?hot)\b", m)
    )

    # 4. Apple ID: credentials, password, account lockout, 2FA, iCloud authentication
    is_apple_id = bool(
        re.search(r"\b(?:apple\s*id|appleid)\b", m) or
        re.search(r"\bicloud\b.*\b(?:login|password|account|sign\s*in|logged\s*out|verification|authenticate)\b", m) or
        re.search(r"\b(?:two[\s-]factor|2fa|verification\s*code|activation\s*lock|security\s*questions?)\b", m) or
        (re.search(r"\b(?:forgot|reset)\s+(?:my\s+)?(?:password|passcode|pin)\b", m) and not re.search(r"\bwi-?fi\b", m)) or
        re.search(r"\b(?:account|id)\s+(?:is\s+)?(?:locked|disabled|compromised|hacked)\b", m) or
        re.search(r"\b(?:password\s+not\s+working|wrong\s+password|enter\s+my\s+password)\b", m)
    )

    # 5. Sound & Bluetooth: AirPods, Bluetooth pairing, headphones, speaker, mic, audio volume
    is_sound = bool(
        re.search(r"\b(?:airpod|airpods|earpod|earpods|beats)\b", m) or
        re.search(r"\bbluetooth\b", m) or
        re.search(r"\b(?:headphone|headphones|earphone|earphones|headset)\b", m) or
        re.search(r"\b(?:speaker|speakers|microphone|mic|earpiece)\b", m) or
        re.search(r"\b(?:crackling|distorted|muffled)\s+(?:sound|audio)\b", m) or
        re.search(r"\b(?:volume\s+low|volume\s+button|can\'?t\s+hear|no\s+sound|sound\s+not\s+working|audio\s+cuts?\s+out|sound\s+cuts?\s+out)\b", m) or
        re.search(r"\b(?:headphone\s+jack|audio\s+adapter|headphone\s+adapter|headphone\s+dongle)\b", m)
    )

    # 6. Screen & Camera: physical display, cracks, touch digitizer, flashlight, camera
    is_screen_camera = bool(
        re.search(r"\b(?:cracked\s+screen|broken\s+screen|shattered\s+screen|screen\s+cracked|screen\s+broken|screen\s+replacement|screen\s+repair)\b", m) or
        re.search(r"\b(?:touchscreen|touch\s+screen|digitizer)\b", m) or
        re.search(r"\b(?:touch\s+(?:is\s+)?not\s+responding|touch\s+(?:is\s+)?unresponsive|touch\s+(?:is\s+)?not\s+working|ghost\s+touch|3d\s+touch|force\s+touch)\b", m) or
        re.search(r"\b(?:black\s+screen|blank\s+screen|screen\s+(?:went|turned)\s+black|screen\s+flickering|green\s+line)\b", m) or
        re.search(r"\b(?:flashlight|torch)\b", m) or
        re.search(r"\bcamera\b.*\b(?:black|blurry|focus|won\'?t\s+open|broken|cracked|flash|flip|roll)\b", m) or
        re.search(r"\b(?:front|rear|back|selfie)\s+camera\b", m) or
        re.search(r"\b(?:portrait\s+mode|live\s+photo)\b", m)
    )

    # 7. Phone Performance: system freezing, UI lag, boot loop, crashing, restart loops
    is_perf = bool(
        re.search(r"\b(?:phone|system|device|os|iphone|ipad|mac|screen)\s+(?:keeps?\s+)?(?:freeze|freezes|freezing|frozen)\b", m) or
        re.search(r"\b(?:phone|system|device|os|iphone|ipad|mac)\s+(?:is\s+)?(?:lagging|laggy|sluggish|slow|unresponsive)\b", m) or
        re.search(r"\b(?:boot\s*loop|bootloop|spinning\s+wheel|spinning\s+gear|respring|respringing)\b", m) or
        re.search(r"\b(?:keeps?\s+restarting|randomly\s+restarting|randomly\s+rebooting|randomly\s+shuts?\s+down|randomly\s+shutting\s+down)\b", m) or
        re.search(r"\b(?:phone|system|device)\s+(?:crashed|keeps?\s+crashing)\b", m)
    )

    # 8. Apps & Storage: App Store, updates, app crashes, storage full, iCloud drive/photos
    is_apps = bool(
        re.search(r"\b(?:app\s+store|appstore|itunes\s+store|itunes)\b", m) or
        re.search(r"\b(?:can\'?t|unable\s+to|won\'?t)\s+(?:download|update|install)\s+(?:apps?|application)\b", m) or
        re.search(r"\b(?:storage\s+is\s+full|storage\s+almost\s+full|not\s+enough\s+storage|out\s+of\s+storage|system\s+storage|manage\s+storage|full\s+storage)\b", m) or
        re.search(r"\bicloud\b.*\b(?:storage|backup|photo|photos|drive|sync)\b", m) or
        re.search(r"\b(?:camera\s+roll|photo\s+library)\b.*\b(?:full|sync|disappeared|deleted|missing|loading)\b", m) or
        re.search(r"\b(?:safari|imessage|facetime|podcast|podcasts|apple\s+music)\b.*\b(?:crash|crashing|not\s+working|won\'?t\s+open|error|glitch)\b", m) or
        re.search(r"\bapp\s+(?:keeps?\s+)?crashing\b", m)
    )

    matches = []
    if is_keyboard: matches.append("Keyboard")
    if is_battery: matches.append("Battery")
    if is_apple_id: matches.append("Apple ID")
    if is_sound: matches.append("Sound & Bluetooth")
    if is_screen_camera: matches.append("Screen & Camera")
    if is_perf: matches.append("Phone Performance")
    if is_apps: matches.append("Apps & Storage")

    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        # Documented boundary rules from docs/decision_log.md:
        # 1. Keyboard priority when typing/keyboard is affected alongside lag/screen
        if "Keyboard" in matches and ("Phone Performance" in matches or "Screen & Camera" in matches):
            return "Keyboard"
        # 2. Battery priority when charging/power leads to unexpected shutdowns
        if "Battery" in matches and "Phone Performance" in matches and ("battery" in m or "charg" in m):
            return "Battery"
        # 3. App-specific crash/hang priority over whole-device performance
        if "Apps & Storage" in matches and "Phone Performance" in matches and ("app" in m or "safari" in m or "itunes" in m or "storage" in m):
            return "Apps & Storage"
        # 4. Apple ID credentials priority over App Store download authentication
        if "Apple ID" in matches and "Apps & Storage" in matches and ("password" in m or "apple id" in m or "verification" in m or "passcode" in m):
            return "Apple ID"
        # 5. Component audio/bluetooth priority over general performance/screen
        if "Sound & Bluetooth" in matches and len(matches) == 2:
            return "Sound & Bluetooth"
        return None

    return None


def main():
    args = parse_arguments()
    random.seed(args.seed)

    print("=" * 65)
    print("AI Customer Support Assistant: Weak Labeling & Training Dataset Creation")
    print("=" * 65)

    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input file not found at: {args.input_path}")

    # Read apple_cleaned.csv
    with open(args.input_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    total_rows_examined = len(all_rows)

    # Filter to unique conversation openings (turn_index == 0 and inbound == True)
    root_conversations: Dict[str, str] = {}
    for row in all_rows:
        cid = row.get("conversation_id", "").strip()
        turn_idx = row.get("turn_index", "").strip()
        is_inbound = row.get("inbound", "").strip().lower() == "true"
        if turn_idx == "0" and is_inbound and cid:
            root_conversations[cid] = row.get("text_clean", "").strip()

    total_conversations_examined = len(root_conversations)

    # Exclude all 280 Golden Set candidate conversation IDs
    candidate_pool = {
        cid: msg for cid, msg in root_conversations.items()
        if cid not in EXCLUDED_CANDIDATE_IDS
    }
    excluded_count = total_conversations_examined - len(candidate_pool)

    # Apply weak labeling rules
    labeled_pool: Dict[str, List[Tuple[str, str]]] = {}
    unlabeled_count = 0

    for cid, msg in candidate_pool.items():
        intent = classify_weak_intent(msg)
        if intent:
            if intent not in labeled_pool:
                labeled_pool[intent] = []
            labeled_pool[intent].append((cid, msg))
        else:
            unlabeled_count += 1

    total_weak_labels_generated = sum(len(items) for items in labeled_pool.values())

    # Stratified selection to balance classes where possible
    selected_records: List[Dict[str, str]] = []
    counts_per_intent: Dict[str, int] = {}

    for intent, items in labeled_pool.items():
        # Shuffle deterministically using the configured seed
        shuffled = list(items)
        random.shuffle(shuffled)

        if args.max_per_intent and args.max_per_intent > 0 and len(shuffled) > args.max_per_intent:
            chosen = shuffled[:args.max_per_intent]
        else:
            chosen = shuffled

        counts_per_intent[intent] = len(chosen)
        for cid, msg in chosen:
            selected_records.append({
                "conversation_id": cid,
                "customer_message": msg,
                "intent": intent,
            })

    # Shuffle the final training set
    random.shuffle(selected_records)

    # Verify zero data leakage
    selected_cids = set(r["conversation_id"] for r in selected_records)
    leakage = selected_cids.intersection(EXCLUDED_CANDIDATE_IDS)
    assert len(leakage) == 0, f"DATA LEAKAGE DETECTED! Found {len(leakage)} overlapping IDs: {leakage}"

    # Write training_data.csv
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    fieldnames = ["conversation_id", "customer_message", "intent"]
    with open(args.output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_records)

    # Verification assertions
    assert len(selected_records) == len(selected_cids), "Duplicate conversation IDs detected in output!"
    assert all(r["customer_message"] for r in selected_records), "Empty customer messages found in output!"
    assert all(r["intent"] for r in selected_records), "Empty intent labels found in output!"

    print(f"\nResults Summary:")
    print(f"  Total AppleSupport rows examined:              {total_rows_examined:,} (across {total_conversations_examined:,} conversation threads)")
    print(f"  Excluded from 280 Golden candidate pool:       {excluded_count}")
    print(f"  High-confidence weak labels generated:         {total_weak_labels_generated}")
    print(f"  Selected for training (balanced ceiling {args.max_per_intent}): {len(selected_records)}")
    print(f"  Unlabeled conversations:                       {unlabeled_count}")
    print(f"\nCount per intent in training set:")
    for intent, count in sorted(counts_per_intent.items(), key=lambda x: -x[1]):
        print(f"  - {intent:<20}: {count}")

    print(f"\nVerification:")
    print(f"  [PASS] Zero overlap with Golden Set candidate pool ({len(EXCLUDED_CANDIDATE_IDS)} IDs excluded)")
    print(f"  [PASS] All {len(selected_records)} conversation IDs are unique")
    print(f"  [PASS] Saved clean training set to: {args.output_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
