#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path("processed_data") / "TIBKAT" / "translating"
OUT_DIR = Path("processed_data") / "TIBKAT" / "embedding"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_texts(jsonl_path: Path):
    paper_ids = []
    texts = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            pid = obj.get("paper_id")
            content = obj.get("content", {})
            text = content.get("text") or content.get("title_en")

            if not pid or not text:
                continue

            paper_ids.append(pid)
            texts.append(text)

    return paper_ids, texts


def embed_and_save(split: str, model: SentenceTransformer, batch_size: int = 64):
    in_path = ROOT / f"{split}_all.jsonl"
    print(f"[{split}] loading from {in_path}")

    paper_ids, texts = load_texts(in_path)
    print(f"[{split}] {len(texts)} texts to encode")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    emb_path = OUT_DIR / f"tibkat_{split}_embeddings.npy"
    ids_path = OUT_DIR / f"tibkat_{split}_ids.json"

    np.save(emb_path, embeddings)
    with ids_path.open("w", encoding="utf-8") as f:
        json.dump(paper_ids, f, ensure_ascii=False, indent=2)

    print(f"[{split}] embeddings saved to {emb_path}")
    print(f"[{split}] ids saved to {ids_path}\n")


def main():
    print("Loading MiniLM model...")
    model = SentenceTransformer(MODEL_NAME)

    for split in ["train", "test"]:
        embed_and_save(split, model, batch_size=64)

    print("All done.")


if __name__ == "__main__":
    main()