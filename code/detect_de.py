#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan GND JSONL and flag fields that likely contain German text.
- Input: JSONL, 1 record per line, keys:
  Code, Classification Name, Name, Alternate Name, Related Subjects, Definition
- Output:
  1) Console summary per field
  2) CSV report with suspected German snippets

Usage:
  python gnd_detect_german.py \
    --input cleaned_data/gnd_translated.jsonl \
    --report reports/german_suspects.csv

Options:
  --min-signals-def 2   # Definition에서 '독일어 신호' 최소 개수(기본 2)
  --min-signals-else 1  # 나머지 필드에서 최소 개수(기본 1)
"""

import argparse, csv, json, re, unicodedata
from typing import Any, Dict, Iterable, List, Tuple

TARGET_FIELDS = [
    "Classification Name",
    "Name",
    "Alternate Name",
    "Related Subjects",
    "Definition",
]

GER_UMLAUT_RE = re.compile(r"[äöüÄÖÜß]")
# 흔한 독어 단어/관사/접속사/전치사 (필요 시 추가)
GER_COMMON_WORDS = {
    "der","die","das","und","oder","mit","für","ohne","nicht","im","am","an","auf","aus",
    "von","nach","wie","auch","ohne","über","unter","zwischen","gegen","sowie",
    "des","den","dem","ein","eine","einer","einem","einen","zur","zum","bei","bis","beim",
    "z.b.","z. b.","z.b", "z. b.",  # 예: z. B. (zum Beispiel)
    "allgemeinbegriff","verknüpfe","bereich","ordnung","deutung",
}
# 독어형 접미사 (단어 길이≥5일 때만 카운트)
GER_SUFFIXES = ("ung","keit","heit","schaft","lich","isch","los","chen","lein")

WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß]+")  # 영/독문자 단어 토큰

def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s).strip()

def to_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(e) for e in x if isinstance(e, (str,int,float))]
    if isinstance(x, (str,int,float)):
        return [str(x)]
    return []

def german_signals(text: str) -> Tuple[int, List[str]]:
    """
    Return (#signals, reasons)
    """
    reasons = []
    t = nfc(text)
    if not t:
        return 0, reasons

    # 1) Umlaut/ß
    if GER_UMLAUT_RE.search(t):
        reasons.append("umlaut/ß")

    # 2) common words (case-insensitive)
    tokens = [w.lower() for w in WORD_RE.findall(t)]
    ger_hits = [w for w in tokens if w in GER_COMMON_WORDS]
    if ger_hits:
        reasons.append(f"common_words:{','.join(sorted(set(ger_hits))[:5])}")

    # 3) suffix patterns
    suf_hits = []
    for w in tokens:
        if len(w) >= 5 and any(w.endswith(suf) for suf in GER_SUFFIXES):
            suf_hits.append(w)
    if suf_hits:
        reasons.append("suffix_patterns")

    return len(reasons), reasons

def iter_jsonl(path: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield line

def scan_record(rec: Dict[str, Any],
                min_signals_else: int,
                min_signals_def: int) -> List[Dict[str, str]]:
    """
    Return a list of flagged entries:
    [{"Code":..., "Field":..., "Value":..., "Signals":..., "Reasons":...}, ...]
    """
    code = rec.get("Code", "")
    flagged = []

    def check_field(field: str, value: str, threshold: int):
        sig, reasons = german_signals(value)
        if sig >= threshold:
            flagged.append({
                "Code": str(code),
                "Field": field,
                "Value": value,
                "Signals": str(sig),
                "Reasons": ";".join(reasons)
            })

    # strings
    cname = rec.get("Classification Name", "")
    name = rec.get("Name", "")
    defin = rec.get("Definition", "")

    if isinstance(cname, str): check_field("Classification Name", cname, min_signals_else)
    if isinstance(name, str):  check_field("Name", name, min_signals_else)
    if isinstance(defin, str): check_field("Definition", defin, min_signals_def)

    # lists
    alts = to_list(rec.get("Alternate Name", []))
    rels = to_list(rec.get("Related Subjects", []))
    for v in alts:
        check_field("Alternate Name", v, min_signals_else)
    for v in rels:
        check_field("Related Subjects", v, min_signals_else)

    return flagged

def main():
    ap = argparse.ArgumentParser(description="Detect likely German text in GND JSONL (offline heuristic).")
    ap.add_argument("--input", required=True, help="Input JSONL path")
    ap.add_argument("--report", required=True, help="Output CSV report path")
    ap.add_argument("--min-signals-def", type=int, default=2, help="Min German signals for Definition (default 2)")
    ap.add_argument("--min-signals-else", type=int, default=1, help="Min German signals for other fields (default 1)")
    args = ap.parse_args()

    total = 0
    flagged_total = 0
    per_field_counts = {f: 0 for f in TARGET_FIELDS}

    rows: List[Dict[str,str]] = []

    for line in iter_jsonl(args.input):
        total += 1
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        flagged = scan_record(rec, args.min_signals_else, args.min_signals_def)
        if flagged:
            flagged_total += 1
            rows.extend(flagged)
            # field별 카운트
            for ent in flagged:
                if ent["Field"] in per_field_counts:
                    per_field_counts[ent["Field"]] += 1

    # CSV 저장
    # columns: Code, Field, Value, Signals, Reasons
    with open(args.report, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Code","Field","Value","Signals","Reasons"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # 콘솔 요약
    print(f"[Summary] scanned records: {total}")
    print(f"[Summary] records with suspected German: {flagged_total}")
    print("[Per-field counts]")
    for k, v in per_field_counts.items():
        print(f"  - {k}: {v}")

if __name__ == "__main__":
    main()