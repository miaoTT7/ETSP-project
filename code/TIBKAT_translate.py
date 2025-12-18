#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import shutil
from pathlib import Path
from typing import List

import torch
from transformers import MarianMTModel, MarianTokenizer

# ====== 경로 설정 ======
ROOT_IN = Path("cleaned_data") / "TIBKAT"
ROOT_OUT = Path("cleaned_data") / "TIBKAT_translated"

DOC_TYPES = ["Article", "Book", "Conference", "Report", "Thesis"]
SPLITS = ["train", "test"]

MODEL_NAME = "Helsinki-NLP/opus-mt-de-en"


print("Loading translation model...")
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME)

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")

model.to(device)
model.eval()


def translate_batch(texts: List[str], max_length: int = 512) -> List[str]:
    """텍스트 리스트를 한 번에 번역."""
    if not texts:
        return []
    enc = tokenizer(
        texts,
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
            num_beams=1,
        )
    return [tokenizer.decode(t, skip_special_tokens=True) for t in generated]


def translate_jsonl_text_and_title(in_path: Path, out_path: Path, batch_size: int = 16):
    """
    독일어 JSONL 파일을 읽어서:
      - content.text
      - content.title_en
    두 필드만 번역해서 덮어쓴다.
    나머지 구조는 그대로 유지.
    """
    print(f"[DE→EN(text,title)] {in_path}  ->  {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:

        batch_objs = []
        batch_pairs = []  # (text, title)

        def flush():
            nonlocal batch_objs, batch_pairs
            if not batch_objs:
                return

            # text, title 번역을 한 번에 처리하기 위해  [text1, title1, text2, title2, ...] 로 합침
            flat_src = []
            for txt, tit in batch_pairs:
                flat_src.append(txt if txt is not None else "")
                flat_src.append(tit if tit is not None else "")

            flat_trans = translate_batch(flat_src)
            assert len(flat_trans) == len(flat_src)

            # 다시 text/title 쌍으로 복원
            it = iter(flat_trans)
            for obj in batch_objs:
                txt_en = next(it)
                tit_en = next(it)

                content = obj.get("content", {})
                # 원래 값이 있던 경우에만 덮어쓰기 (없으면 그냥 비워둠)
                if content.get("text") is not None:
                    content["text"] = txt_en
                if content.get("title_en") is not None:
                    content["title_en"] = tit_en
                obj["content"] = content

                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

            batch_objs = []
            batch_pairs = []

        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            content = obj.get("content", {})
            src_text = content.get("text")
            src_title = content.get("title_en")

            # 둘 다 없으면 번역하지 않고 그대로 씀
            if src_text is None and src_title is None:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue

            batch_objs.append(obj)
            batch_pairs.append((src_text, src_title))

            if len(batch_objs) >= batch_size:
                flush()

        # 마지막 남은 것 처리
        flush()


def copy_jsonl(in_path: Path, out_path: Path):
    """영어 파일은 그대로 복사."""
    print(f"[COPY EN] {in_path}  ->  {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(in_path, out_path)


def main():
    for split in SPLITS:
        in_dir = ROOT_IN / split
        out_dir = ROOT_OUT / split

        for doc_type in DOC_TYPES:
            # 독일어 파일: text + title_en 번역
            de_in = in_dir / f"{doc_type}_de.jsonl"
            if de_in.exists():
                de_out = out_dir / f"{doc_type}_de.jsonl"
                translate_jsonl_text_and_title(de_in, de_out, batch_size=4)
            else:
                print(f"[WARN] Missing file: {de_in}")

            # 영어 파일: 그대로 복사
            en_in = in_dir / f"{doc_type}_en.jsonl"
            if en_in.exists():
                en_out = out_dir / f"{doc_type}_en.jsonl"
                copy_jsonl(en_in, en_out)
            else:
                print(f"[WARN] Missing file: {en_in}")

    print("All done.")


if __name__ == "__main__":
    main()