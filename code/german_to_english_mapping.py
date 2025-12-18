#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replace 'Classification Name' in GND JSONL using a strict german→english CSV mapping.

Assumptions
- Mapping CSV has EXACTLY these headers (lowercase): german,english
- Encoding: UTF-8 (BOM 허용)
- Default delimiter: ','  (지역설정으로 세미콜론이면 --delimiter ';')

Behavior
- Only 'Classification Name' 값을 치환 (다른 필드는 그대로)
- 키/값은 NFC 정규화, 키 비교는 기본적으로 case-insensitive
- 통계와 미치환 목록(optional) 출력

Usage
  python apply_classname_mapping_simple.py \
    --input cleaned_data/gnd_translated.jsonl \
    --mapping /mnt/data/german_to_english.csv \
    --output cleaned_data/gnd_translated_mapped.jsonl

Options
  --exact-case     : 키 매칭을 대소문자 구분 (기본은 구분 안 함)
  --dump-misses F  : 매핑이 안 된 'Classification Name' 고유값을 F에 저장
  --delimiter D    : CSV 구분자 지정(기본 ',')
"""

import argparse, csv, json, sys, unicodedata
from typing import Dict, Set

def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", str(s) if s is not None else "").strip()

def norm_key(s: str, exact_case: bool) -> str:
    s = nfc(s)
    return s if exact_case else s.casefold()

def load_strict_mapping(path: str, exact_case: bool, delimiter: str) -> Dict[str, str]:
    """
    Strictly load CSV with headers 'german,english'.
    Uses utf-8-sig to handle BOM.
    """
    mp: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f, delimiter=delimiter)
        # 엄밀히 검사: 헤더가 정확히 german, english 여야 함
        if not r.fieldnames:
            raise SystemExit("Mapping CSV has no header row.")
        headers = [h.strip().lower() for h in r.fieldnames]
        if "german" not in headers or "english" not in headers:
            raise SystemExit("Mapping CSV must have headers exactly: german,english")

        for row in r:
            g = row.get("german", "")
            e = row.get("english", "")
            if not g or not e:
                continue
            key = norm_key(g, exact_case)
            val = nfc(e)
            if key and val:
                mp[key] = val
    return mp

def main():
    ap = argparse.ArgumentParser(description="Apply strict german→english mapping to Classification Name.")
    ap.add_argument("--input", required=True, help="Input JSONL path")
    ap.add_argument("--mapping", required=True, help="Mapping CSV path (headers: german,english)")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    ap.add_argument("--exact-case", action="store_true", help="Case-sensitive key match (default: off)")
    ap.add_argument("--dump-misses", help="Write unmapped Classification Names to this .txt file")
    ap.add_argument("--delimiter", default=",", help="CSV delimiter (default ',')")
    args = ap.parse_args()

    # 1) 매핑 로드
    mapping = load_strict_mapping(args.mapping, args.exact_case, args.delimiter)
    if not mapping:
        print("[Warn] mapping CSV loaded but empty.", file=sys.stderr)

    # 2) 적용
    total = 0
    replaced = 0
    hits: Set[str] = set()
    misses: Set[str] = set()

    with open(args.output, "w", encoding="utf-8") as fout, open(args.input, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            cname = obj.get("Classification Name", "")
            if isinstance(cname, str) and cname.strip():
                key = norm_key(cname, args.exact_case)
                if key in mapping:
                    new_val = mapping[key]
                    if new_val and new_val != cname:
                        obj["Classification Name"] = new_val
                        replaced += 1
                        hits.add(cname)
                else:
                    misses.add(cname)

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # 3) 미치환 목록(optional)
    if args.dump_misses:
        with open(args.dump_misses, "w", encoding="utf-8") as f:
            for v in sorted({ nfc(x) for x in misses }):
                f.write(v + "\n")

    # 4) 통계
    print(f"[Done] total lines: {total}, replaced lines: {replaced}", file=sys.stderr)
    print(f"[Unique] hits: {len(hits)}, misses: {len(misses)}", file=sys.stderr)
    if args.dump_misses:
        print(f"[Misses] written to: {args.dump_misses}", file=sys.stderr)

if __name__ == "__main__":
    main()