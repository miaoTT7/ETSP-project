# GND_embedding_test_nofaiss.py

import json
import numpy as np

EMB_PATH = "cleaned_data/GND/subject_embeddings.npy"
IDS_PATH = "cleaned_data/GND/subject_ids.json"
TEXTS_PATH = "cleaned_data/GND/subject_texts.json"

# 1) 저장된 것들 로드
emb = np.load(EMB_PATH).astype("float32")
with open(IDS_PATH, "r", encoding="utf-8") as f:
    ids = json.load(f)
with open(TEXTS_PATH, "r", encoding="utf-8") as f:
    texts = json.load(f)

print("Embeddings shape:", emb.shape)

# 2) 코사인 유사도 계산을 위한 정규화
norms = np.linalg.norm(emb, axis=1, keepdims=True)
emb_norm = emb / (norms + 1e-12)

# 3) 임의의 subject 하나를 골라서, 비슷한 subject top-5를 찾아보기
query_idx = 0  # 0번째 subject
query_vec = emb_norm[query_idx : query_idx + 1]  # shape (1, dim)

# 전체와의 코사인 유사도 계산
sims = (emb_norm @ query_vec.T).reshape(-1)  # (N,)

# 자기 자신 포함 상위 5개 인덱스
topk = 5
top_indices = np.argsort(-sims)[:topk]

print("\n[Query subject]")
print("ID:", ids[query_idx])
print("TEXT:", texts[query_idx])

print("\n[Top-5 similar subjects (by cosine similarity)]")
for rank, idx in enumerate(top_indices, start=1):
    print(f"{rank}. ID={ids[idx]}  sim={sims[idx]:.4f}")
    print("   ", texts[idx])
    print()