import os
import re
import sys
import json
import time
import random
import argparse
import unicodedata
from typing import Any, Dict, Iterable, List, Tuple

import requests
from tqdm import tqdm

DEEPL_FREE_URL = "https://api.deepl.com/v2/translate"

# ------------------------- Text Utils -------------------------

def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s).strip()

def to_str_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(e) for e in x if isinstance(e, (str, int, float))]
    if isinstance(x, (str, int, float)):
        return [str(x)]
    return []

# ---------------------- Acronym Protector ---------------------

ACRO_RE = re.compile(r"\b([A-Z]{2,4}s?)\b")  # e.g., AI, GPU, NLP, CPUs

def protect_acronyms(texts: List[str]) -> Tuple[List[str], List[Dict[str, str]]]:
    protected, maps = [], []
    for t in texts:
        idx = 0
        mapping: Dict[str, str] = {}
        def repl(m):
            nonlocal idx
            token = f"__ACR{idx}__"
            mapping[token] = m.group(1)
            idx += 1
            return token
        new_t = ACRO_RE.sub(repl, t)
        protected.append(new_t)
        maps.append(mapping)
    return protected, maps

def restore_acronyms(texts: List[str], maps_list: List[Dict[str, str]]) -> List[str]:
    out = []
    for t, mp in zip(texts, maps_list):
        for token, val in mp.items():
            t = t.replace(token, val)
        out.append(t)
    return out

# ---------------------- DeepL Translate -----------------------

def deepl_translate_batch_auto(
    texts: List[str],
    api_key: str,
    target_lang: str = "EN",
    max_retries: int = 8,
    timeout: int = 60,
) -> List[str]:
    """
    DeepL FREE translate (AUTO source detection).
    Sends the texts without source_lang to let DeepL detect language.
    Retries on 408, 429, 456, 503 and 5xx.
    """
    if not texts:
        return []

    data = {"target_lang": target_lang}
    for t in texts:
        data.setdefault("text", []).append(t)

    headers = {"Authorization": f"DeepL-Auth-Key {api_key}"}
    retryable = {408, 429, 456, 503}
    delay = 1.0

    with requests.Session() as s:
        s.headers.update(headers)
        for attempt in range(1, max_retries + 1):
            try:
                r = s.post(DEEPL_FREE_URL, data=data, timeout=timeout)
                if r.status_code == 200:
                    js = r.json()
                    outs = [item["text"] for item in js.get("translations", [])]
                    if len(outs) < len(texts):  # pad (rare safety)
                        outs.extend(texts[len(outs):])
                    return outs

                # retryable?
                if r.status_code in retryable or (500 <= r.status_code < 600):
                    time.sleep(delay + random.uniform(0, 0.5))
                    delay = min(delay * 1.8, 20)
                    continue

                # non-retryable → raise with message
                try:
                    msg = r.json()
                except Exception:
                    msg = r.text
                raise RuntimeError(f"DeepL error {r.status_code}: {msg}")

            except requests.RequestException as e:
                if attempt == max_retries:
                    raise
                time.sleep(delay + random.uniform(0, 0.5))
                delay = min(delay * 1.8, 20)

    # Fallback (should not usually happen)
    return texts

# ------------------- Record Translation -----------------------

def translate_record(
    rec: Dict[str, Any],
    api_key: str,
    batch_size: int,
    protect: bool,
    cache: Dict[str, str],
) -> Dict[str, Any]:
    """
    Translate all fields except Code. Keep schema & list shapes.
    """
    code = rec.get("Code", "")

    cname = rec.get("Classification Name", "")
    name  = rec.get("Name", "")
    alts  = to_str_list(rec.get("Alternate Name", []))
    rels  = to_str_list(rec.get("Related Subjects", []))
    defin = rec.get("Definition", "")

    # soft normalize
    cname = nfc(cname) if isinstance(cname, str) else ""
    name  = nfc(name) if isinstance(name, str) else ""
    alts  = [nfc(x) for x in alts]
    rels  = [nfc(x) for x in rels]
    defin = nfc(defin) if isinstance(defin, str) else ""

    # Build queue for ALL non-empty values (except Code), caching duplicates
    queue: List[str] = []
    targets: List[Tuple[str, int | None]] = []

    def enqueue(field: str, value: str, idx: int | None = None):
        if value and value not in cache:
            queue.append(value); targets.append((field, idx))

    enqueue("Classification Name", cname)
    enqueue("Name", name)
    for i, s in enumerate(alts):
        enqueue("Alternate Name", s, i)
    for i, s in enumerate(rels):
        enqueue("Related Subjects", s, i)
    enqueue("Definition", defin)

    # Call DeepL in batches
    if queue:
        if protect:
            prot, maps = protect_acronyms(queue)
            translated: List[str] = []
            for i in range(0, len(prot), batch_size):
                chunk = prot[i:i+batch_size]
                tr = deepl_translate_batch_auto(chunk, api_key)
                translated.extend(tr)
            translated = restore_acronyms(translated, maps)
        else:
            translated = []
            for i in range(0, len(queue), batch_size):
                chunk = queue[i:i+batch_size]
                tr = deepl_translate_batch_auto(chunk, api_key)
                translated.extend(tr)

        # write-back & cache
        for (field, idx), src, tgt in zip(targets, queue, translated):
            cache[src] = tgt
            if field == "Classification Name":
                cname = tgt
            elif field == "Name":
                name = tgt
            elif field == "Alternate Name":
                alts[idx] = tgt
            elif field == "Related Subjects":
                rels[idx] = tgt
            elif field == "Definition":
                defin = tgt

    return {
        "Code": code,
        "Classification Name": cname,
        "Name": name,
        "Alternate Name": alts,
        "Related Subjects": rels,
        "Definition": defin,
    }

# -------------------- IO & Resume Helpers ---------------------

def iter_jsonl(path: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield line

def count_lines(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0

# ----------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Translate GND (6 fields) to EN via DeepL FREE (auto-detect)")
    ap.add_argument("--input", required=True, help="Input JSONL path")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    ap.add_argument("--batch-size", type=int, default=30, help="Batch size per API call")
    ap.add_argument("--protect-acronyms", action="store_true", help="Protect ALL-CAPS 2–4 letter acronyms")
    ap.add_argument("--resume", action="store_true", help="Skip already written lines and append")
    args = ap.parse_args()

    api_key = os.environ.get("DEEPL_API_KEY")
    if not api_key:
        raise SystemExit("DEEPL_API_KEY not set. Run: export DEEPL_API_KEY='YOUR_DEEPL_FREE_KEY'")

    # progress length
    try:
        total_lines = sum(1 for _ in iter_jsonl(args.input))
    except Exception:
        total_lines = None

    # resume mode
    start_skip = 0
    mode = "w"
    if args.resume:
        start_skip = count_lines(args.output)
        mode = "a" if start_skip > 0 else "w"
        if start_skip > 0:
            print(f"[Resume] Skipping first {start_skip} lines of input.", file=sys.stderr)

    cache: Dict[str, str] = {}
    error_log = args.output + ".errors"

    processed = 0
    written = 0

    lines = list(iter_jsonl(args.input))
    with open(args.output, mode, encoding="utf-8") as fout:
        for idx, line in enumerate(tqdm(lines, total=total_lines, desc="Translating (DeepL FREE auto)"), start=1):
            processed += 1
            if args.resume and idx <= start_skip:
                continue
            try:
                rec = json.loads(line)
                out_rec = translate_record(
                    rec=rec,
                    api_key=api_key,
                    batch_size=args.batch_size,
                    protect=args.protect_acronyms,
                    cache=cache,
                )
                fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                written += 1
            except Exception as e:
                # log and continue
                with open(error_log, "a", encoding="utf-8") as ef:
                    ef.write(json.dumps({"line_no": idx, "error": str(e), "raw": line}, ensure_ascii=False) + "\n")
                continue

    print(f"[Done] lines_seen: {processed}, written: {written}", file=sys.stderr)

if __name__ == "__main__":
    main()