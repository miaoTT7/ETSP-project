import argparse
import json
import sys
import unicodedata
from typing import Any, Dict, Iterable, List

KEEP_KEYS = [
    "Code",
    "Classification Name",
    "Name",
    "Alternate Name",
    "Related Subjects",
    "Definition",
]

DEFAULT_REQUIRED = ["Code", "Name"]

def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s).strip()

def maybe_norm(s: Any, do_norm: bool) -> str:
    if not isinstance(s, str):
        return ""
    return nfc(s) if do_norm else s

def dedup_preserve(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if not isinstance(x, str):
            continue
        k = x if not x else x  # placeholder; actual dedup below
        # dedup 은 완전 동일 문자열 기준 (정규화/strip 여부는 외부에서 결정)
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out

def as_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [e for e in x if isinstance(e, str)]
    if isinstance(x, str):
        return [x]
    return []

def load_records(path: str) -> Iterable[Dict[str, Any]]:
    if path.lower().endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    yield obj
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        for obj in data:
            if isinstance(obj, dict):
                yield obj
    elif isinstance(data, dict):
        yield data  # 단일 객체 허용

def to_separate(rec: Dict[str, Any], do_norm: bool) -> Dict[str, Any]:
    code = maybe_norm(rec.get("Code", ""), do_norm)
    cname = maybe_norm(rec.get("Classification Name", ""), do_norm)
    name = maybe_norm(rec.get("Name", ""), do_norm)
    alt_list_raw = as_list(rec.get("Alternate Name", []))
    related_raw = as_list(rec.get("Related Subjects", []))
    definition = maybe_norm(rec.get("Definition", ""), do_norm)

    # 리스트 항목 정규화/strip은 옵션에 따름
    alt_list = [nfc(x) if do_norm else x for x in alt_list_raw]
    related_list = [nfc(x) if do_norm else x for x in related_raw]

    # 리스트 내 중복 제거(완전 동일 문자열 기준, 순서 보존)
    alt_list = dedup_preserve(alt_list)
    related_list = dedup_preserve(related_list)

    return {
        "Code": code,
        "Classification Name": cname,
        "Name": name,
        "Alternate Name": alt_list,
        "Related Subjects": related_list,
        "Definition": definition,
    }

def valid_required(rec: Dict[str, Any], required_keys: List[str]) -> bool:
    for k in required_keys:
        v = rec.get(k, None)
        if isinstance(v, str):
            if not v.strip():
                return False
        elif isinstance(v, list):
            if len(v) == 0:
                return False
        else:
            if v is None:
                return False
    return True

def main():
    ap = argparse.ArgumentParser(description="Keep six GND fields with Name/Alternate separated and original labels.")
    ap.add_argument("--input", required=True, help="Path to input .json or .jsonl")
    ap.add_argument("--output", required=True, help="Path to output .jsonl")
    ap.add_argument("--strict", action="store_true", help="Drop records missing required fields (default required: Code, Name, Definition)")
    ap.add_argument("--require", nargs="*", default=None, help="Custom required fields for dropping (space-separated)")
    ap.add_argument("--no-normalize", action="store_true", help="Disable NFC/strip normalization (preserve raw strings)")
    args = ap.parse_args()

    do_norm = not args.no_normalize
    required = args.require if (args.strict and args.require) else (DEFAULT_REQUIRED if args.strict else None)

    total = kept = dropped = 0
    with open(args.output, "w", encoding="utf-8") as fout:
        for rec in load_records(args.input):
            total += 1
            out = to_separate(rec, do_norm=do_norm)
            if required and not valid_required(out, required):
                dropped += 1
                continue
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[Done] total: {total}, kept: {kept}, dropped: {dropped}", file=sys.stderr)

if __name__ == "__main__":
    main()