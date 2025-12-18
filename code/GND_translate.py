#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GND 6-field translator to English using Helsinki-NLP/opus-mt-de-en (MarianMT).

- Code: 그대로 유지
- 나머지 5개 필드:
    Classification Name, Name, Alternate Name[], Related Subjects[], Definition
  의 텍스트를 전부 독일어→영어 번역 모델에 넣음.
  (이미 영어여도 약간의 패러프레이즈 정도만 날 수 있음)

- DeepL / API 사용 X  → 비용 없음
- 캐시 사용: 같은 문구는 한 번만 번역
- resume 옵션: 이전에 번역한 출력 파일이 있으면 그 이후 줄부터 이어서 처리

Usage:
  python GND_translate_marian.py \
    --input cleaned_data/gnd_separate.jsonl \
    --output cleaned_data/gnd_translated_marian.jsonl \
    --batch-size 16 \
    --protect-acronyms \
    --resume
"""

import os
import re
import sys
import json
import time
import random
import argparse
import unicodedata
from typing import Any, Dict, Iterable, List, Tuple

import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

MODEL_NAME = "Helsinki-NLP/opus-mt-de-en"

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

# ---------------------- MarianMT Translate --------------------

def load_marian_model():
    """
    Load tokenizer + model once, move model to CPU/GPU.
    """
    print(f"[Info] Loading model {MODEL_NAME} ...", file=sys.stderr)
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"[Info] Model loaded on device: {device}", file=sys.stderr)
    return tokenizer, model, device

def marian_translate_batch(
    texts: List[str],
    tokenizer: MarianTokenizer,
    model: MarianMTModel,
    device: torch.device,
    max_length: int = 512,
) -> List[str]:
    """
    Translate a batch of texts using MarianMT (DE->EN).
    """
    if not texts:
        return []
    # tokenizer가 빈 문자열을 싫어해서 최소한 공백 제거
    clean_texts = [t if t.strip() else " " for t in texts]

    enc = tokenizer(
        clean_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        generated = model.generate(
            **enc,
            max_length=max_length,
            num_beams=4,
        )

    out = tokenizer.batch_decode(generated, skip_special_tokens=True)
    # 앞뒤 공백 정리 + NFC 정규화
    return [nfc(t) for t in out]

# ------------------- Record Translation -----------------------

def translate_record(
    rec: Dict[str, Any],
    tokenizer: MarianTokenizer,
    model: MarianMTModel,
    device: torch.device,
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

    # Call Marian in batches
    if queue:
        if protect:
            prot, maps = protect_acronyms(queue)
            translated: List[str] = []
            for i in range(0, len(prot), batch_size):
                chunk = prot[i:i+batch_size]
                tr = marian_translate_batch(chunk, tokenizer, model, device)
                translated.extend(tr)
            translated = restore_acronyms(translated, maps)
        else:
            translated = []
            for i in range(0, len(queue), batch_size):
                chunk = queue[i:i+batch_size]
                tr = marian_translate_batch(chunk, tokenizer, model, device)
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
    ap = argparse.ArgumentParser(description="Translate GND (6 fields) to EN via Helsinki-NLP/opus-mt-de-en")
    ap.add_argument("--input", required=True, help="Input JSONL path")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    ap.add_argument("--batch-size", type=int, default=16, help="Batch size per model call (texts)")
    ap.add_argument("--protect-acronyms", action="store_true", help="Protect ALL-CAPS 2–4 letter acronyms")
    ap.add_argument("--resume", action="store_true", help="Skip already written lines and append")
    args = ap.parse_args()

    # 모델 로드
    tokenizer, model, device = load_marian_model()

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
        for idx, line in enumerate(tqdm(lines, total=total_lines, desc="Translating (Marian DE→EN)"), start=1):
            processed += 1
            if args.resume and idx <= start_skip:
                continue
            try:
                rec = json.loads(line)
                out_rec = translate_record(
                    rec=rec,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
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