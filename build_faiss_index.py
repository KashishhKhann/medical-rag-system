"""
Build FAISS index from MIMIC.processed_chunks embeddings.

- Index type: cosine similarity (L2-normalized embeddings + IndexFlatIP)
- Outputs:
    data/faiss_biobert.index
    data/faiss_biobert_mapping.json
"""

import os
import json
import numpy as np
from pymongo import MongoClient
from tqdm import tqdm
import faiss

from config import (
    MONGO_URI, DB_NAME, CHUNKS_COLLECTION,
    FAISS_INDEX_PATH, FAISS_MAP_PATH,
    FIELD_CHUNK_ID, FIELD_EMBEDDING
)

os.makedirs("data", exist_ok=True)

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
col = db[CHUNKS_COLLECTION]

print("\n------------------------")
print("Building FAISS index from processed_chunks.embeddings")
print("------------------------")

cursor = col.find({FIELD_EMBEDDING: {"$exists": True}}, {FIELD_CHUNK_ID: 1, FIELD_EMBEDDING: 1})
docs = list(cursor)
print(f"Chunks with embeddings: {len(docs)}")

if not docs:
    print("No embeddings found. Run add_embeddings.py first.")
    client.close()
    raise SystemExit

try:
    chunk_ids = []
    emb_list  = []

    for d in tqdm(docs, desc="Collecting vectors"):
        emb = d.get(FIELD_EMBEDDING)
        if not emb:
            continue
        emb_np = np.asarray(emb, dtype="float32")
        if emb_np.ndim != 1:
            continue
        chunk_ids.append(d[FIELD_CHUNK_ID])
        emb_list.append(emb_np)

    X = np.stack(emb_list, axis=0)
    print(f"Matrix shape: {X.shape}")

    faiss.normalize_L2(X)

    d = X.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(X)
    print(f"FAISS index built with {index.ntotal} vectors, dim={d}")

    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"Saved index -> {FAISS_INDEX_PATH}")

    mapping = {i: cid for i, cid in enumerate(chunk_ids)}
    with open(FAISS_MAP_PATH, "w") as f:
        json.dump(mapping, f)

    print(f"Saved mapping -> {FAISS_MAP_PATH}")
    print("------------------------\n")

finally:
    client.close()
