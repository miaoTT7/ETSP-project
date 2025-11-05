#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TIBKAT jsonld -> project JSONL (10 files: 5 types x 2 languages)

Usage:
  python tibkat_preprocess.py --root /path/to/TIBKAT --out data/processed/tibkat

Creates files like:
  data/processed/tibkat/Article_en.jsonl
  data/processed/tibkat/Article_de.jsonl
  ...
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

TIBKAT_ID_RE = re.compile(r"(TIBKAT%3A[^\s/?#]+)")

def extract_paper_id(url: str) -> Optional[str]:
    """
    From something like:
      https://www.tib.eu/de/suchen/id/TIBKAT%3A72999130X
    return:
      TIBKAT%3A72999130X
    """
    if not isinstance(url, str):
        return None
    m = TIBKAT_ID_RE.search(url)
    return m.group(1) if m else None

def normalize_text(x: Union[str, List[str], None]) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, list):
        # join list pieces with a space, drop empties
        x = " ".join([str(t).strip() for t in x if str(t).strip()])
    if isinstance(x, str):
        s = " ".join(x.split())  # collapse whitespace
        return s if s else None
    return None

def to_list(x: Union[str, List[str], None]) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(t) for t in x]
    return [str(x)]

def find_main_record(graph: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Heuristic: prefer node with '@type' starting 'bibo:' OR node that has 'title'/'abstract'.
    Fallback to the longest dict.
    """
    candidates = []
    for node in graph:
        if isinstance(node, dict):
            t = node.get("@type", "")
            has_core = ("title" in node) or ("abstract" in node)
            score = 0
            if isinstance(t, str) and t.startswith("bibo:"):
                score += 2
            if has_core:
                score += 1
            # larger dicts tend to be the main record
            score += min(len(node), 10) / 10.0
            candidates.append((score, node))
    if not candidates:
        return {}
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

def extract_subject_ids(node: Dict[str, Any]) -> List[str]:
    subs = node.get("dcterms:subject") or node.get("dcterms:subjects") or node.get("subjects")
    if not subs:
        return []
    out = []
    if isinstance(subs, list):
        for s in subs:
            if isinstance(s, dict) and "@id" in s:
                out.append(str(s["@id"]))
            # ignore plain strings like "Finance and Investment"
    elif isinstance(subs, dict) and "@id" in subs:
        out.append(str(subs["@id"]))
    return out

def join_title_abstract(title: Optional[str], abstract: Optional[str]) -> Optional[str]:
    title = normalize_text(title)
    abstract = normalize_text(abstract)
    if title and abstract:
        # ensure single terminal period between title and abstract
        t = title.rstrip(" .")
        a = abstract.lstrip(" ")
        return f"{t}. {a}"
    return title or abstract  # if one is missing, use the other

def extract_language_from_url(lang_url: Optional[str]) -> Optional[str]:
    if not isinstance(lang_url, str):
        return None
    # examples: http://id.loc.gov/vocabulary/iso639-1/en  -> 'en'
    seg = lang_url.rstrip("/").split("/")[-1]
    return seg if seg else None

def as_str_or_none(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, list):
        # many records have single-element lists
        if not x:
            return None
        return normalize_text(x[0])
    if isinstance(x, (str, int, float)):
        return str(x)
    return None

def extract_authors(creator_field: Any) -> Optional[str]:
    vals = to_list(creator_field)
    vals = [normalize_text(v) for v in vals]
    vals = [v for v in vals if v]
    if not vals:
        return None
    # join authors with '; ' to keep it compact
    return "; ".join(vals)

def transform_record(rec: Dict[str, Any], type_name: str, lang_dir: str) -> Optional[Dict[str, Any]]:
    if not rec:
        return None

    paper_id = extract_paper_id(rec.get("@id", ""))
    # If still None, allow fallback to identifier list (ppn/doi), but keep None if not found
    if paper_id is None:
        ids = to_list(rec.get("identifier"))
        # prefer PPN-like code if present (ppn)XXXXXXXXX
        fallback = next((i for i in ids if "(ppn)" in i), None) or (ids[0] if ids else None)
        paper_id = fallback

    title = as_str_or_none(rec.get("title"))
    abstract = rec.get("abstract")
    # abstract can be str or list
    abstract = normalize_text(abstract)

    content_text = join_title_abstract(title, abstract)
    title_en = title  # 그대로 저장 (파일 언어와 무관)

    subjects = extract_subject_ids(rec)

    authors = extract_authors(rec.get("creator"))
    year = as_str_or_none(rec.get("issued"))

    # language_original: 폴더 기준으로 설정하지만, 레코드의 language URL도 참고해보자(없으면 폴더값 사용)
    lang_from_rec = extract_language_from_url(as_str_or_none(rec.get("language")))
    language_original = lang_from_rec or lang_dir

    out = {
        "paper_id": paper_id,
        "content": {
            "text": content_text,
            "title_en": title_en
        },
        "subject": {
            "labels": subjects
        },
        "metadata": {
            "authors": authors,
            "year": year,
            "type": type_name,
            "language_original": language_original
        }
    }

    # 최소 필수: paper_id와 content.text 혹은 title 중 하나는 있어야 의미 있음
    if not paper_id:
        return None
    if not (content_text or title_en):
        return None

    return out

def process_pair(root: Path, type_name: str, lang_dir: str, out_dir: Path) -> int:
    """
    Process all *.jsonld under {root}/{type_name}/{lang_dir}
    Write one JSONL file at out_dir/{type_name}_{lang_dir}.jsonl
    Returns number of written records.
    """
    in_dir = root / type_name / lang_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{type_name}_{lang_dir}.jsonl"

    count = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for path in sorted(in_dir.rglob("*.jsonld")):
            try:
                doc = read_json(path)
                graph = doc.get("@graph", [])
                if not isinstance(graph, list) or not graph:
                    continue
                main = find_main_record(graph)
                transformed = transform_record(main, type_name=type_name, lang_dir=lang_dir)
                if transformed is None:
                    continue
                fout.write(json.dumps(transformed, ensure_ascii=False) + "\n")
                count += 1
            except Exception as e:
                # 조용히 스킵하되 필요하면 로깅 추가 가능
                # print(f"[WARN] Failed on {path}: {e}")
                continue
    return count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True,
                        help="Path to TIBKAT root folder containing Article/Book/Conference/Report/Thesis")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output folder for JSONL files")
    args = parser.parse_args()

    type_dirs = ["Article", "Book", "Conference", "Report", "Thesis"]
    lang_dirs = ["de", "en"]

    args.out.mkdir(parents=True, exist_ok=True)

    summary = {}
    for t in type_dirs:
        for l in lang_dirs:
            n = process_pair(args.root, t, l, args.out)
            summary[f"{t}_{l}"] = n

    # 간단 요약 출력
    print("Done. Wrote:")
    for k, v in summary.items():
        print(f"  {k}.jsonl: {v} records")

if __name__ == "__main__":
    main()