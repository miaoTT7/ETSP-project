#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

ROOT_IN = Path("processed_data") / "TIBKAT" / "translating"
SPLITS = ["train", "test"]
DOC_TYPES = ["Article", "Book", "Conference", "Report", "Thesis"]
LANGS = ["de", "en"]  # *_de.jsonl + *_en.jsonl

def merge_split(split: str):
    in_dir = ROOT_IN / split
    out_path = ROOT_IN / f"{split}_all.jsonl"

    print(f"[{split}] merging into {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        out_path.unlink()

    count = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for doc_type in DOC_TYPES:
            for lang in LANGS:
                path = in_dir / f"{doc_type}_{lang}.jsonl"
                if not path.exists():
                    print(f"  [WARN] missing: {path}")
                    continue

                print(f"  + {path}")
                with path.open("r", encoding="utf-8") as fin:
                    for line in fin:
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            print(f"    [SKIP] invalid JSON line in {path}")
                            continue

                        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        count += 1

    print(f"[{split}] done. {count} records written.\n")


def main():
    for split in SPLITS:
        merge_split(split)

if __name__ == "__main__":
    main()