import json
import os
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

INPUT_JSONL = "processed_data/GND/translating/translated_6_GND.jsonl"
OUT_DIR = "processed_data/GND/embedding"
SUBJ_EMB_PATH = os.path.join(OUT_DIR, "subject_embeddings.npy")
SUBJ_IDS_PATH = os.path.join(OUT_DIR, "subject_ids.json")
SUBJ_TEXTS_PATH = os.path.join(OUT_DIR, "subject_texts.json")

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading sentence-transformer model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

gnd_ids = []
texts = []

print("Reading GND jsonl and preparing texts...")

with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        obj = json.loads(line)

        code = obj.get("Code")
        label = obj.get("Classification Name", "") or ""
        definition = obj.get("Definition", "") or ""

        if not code or not label:
            continue

        if definition:
            text = f"{label}. {definition}"
        else:
            text = label

        gnd_ids.append(code)
        texts.append(text)

print(f"Total subjects to embed: {len(texts)}")

print("Encoding texts into embeddings...")
embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

print("Embeddings shape:", embeddings.shape)  # (num_subjects, embedding_dim)

np.save(SUBJ_EMB_PATH, embeddings)

with open(SUBJ_IDS_PATH, "w", encoding="utf-8") as f:
    json.dump(gnd_ids, f, ensure_ascii=False, indent=2)

with open(SUBJ_TEXTS_PATH, "w", encoding="utf-8") as f:
    json.dump(texts, f, ensure_ascii=False, indent=2)

print("Saved:")
print(" -", SUBJ_EMB_PATH)
print(" -", SUBJ_IDS_PATH)
print(" -", SUBJ_TEXTS_PATH)
