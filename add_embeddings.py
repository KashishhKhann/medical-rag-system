"""
Add BioClinicalBERT embeddings to chunks in MIMIC.processed_chunks.

Quick fix:
- Forces Hugging Face checkpoint + CLS pooling (NO mean pooling).
- Default is idempotent (embeds only missing embeddings).
- Set FORCE_REEMBED=1 to overwrite embeddings for all chunks (no clearing needed).
"""

import os
from pymongo import MongoClient, UpdateOne
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, models

from config import (
    MONGO_URI, DB_NAME, CHUNKS_COLLECTION, BERT_MODEL,
    FIELD_EMBEDDING, FIELD_FULL_TEXT, FIELD_TEXT
)

DOC_BATCH_SIZE = int(os.getenv("DOC_BATCH_SIZE", "256"))
ENCODE_BATCH_SIZE = int(os.getenv("ENCODE_BATCH_SIZE", "32"))
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "256"))
FORCE_REEMBED = os.getenv("FORCE_REEMBED", "0") == "1"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
col = db[CHUNKS_COLLECTION]

print("------------------------")
print("Adding BioClinicalBERT embeddings to processed_chunks")
print("------------------------")
print(f"MongoDB: {MONGO_URI}")
print(f"Database: {DB_NAME}")
print(f"Collection: {CHUNKS_COLLECTION}")
print(f"Model (HF checkpoint): {BERT_MODEL}")
print(f"FORCE_REEMBED: {FORCE_REEMBED}")
print(f"MAX_SEQ_LENGTH: {MAX_SEQ_LENGTH}")
print("------------------------------------------------------------")

word = models.Transformer(BERT_MODEL, max_seq_length=MAX_SEQ_LENGTH)
pool = models.Pooling(
    word.get_word_embedding_dimension(),
    pooling_mode_cls_token=True,
    pooling_mode_mean_tokens=False,
    pooling_mode_max_tokens=False,
)
embedder = SentenceTransformer(modules=[word, pool])

dim = embedder.get_sentence_embedding_dimension()
print(f"Pooling: CLS={pool.pooling_mode_cls_token} MEAN={pool.pooling_mode_mean_tokens} MAX={pool.pooling_mode_max_tokens}")
print(f"Embedding dimension: {dim}")

query = {} if FORCE_REEMBED else {FIELD_EMBEDDING: {"$exists": False}}
total = col.count_documents(query)
print(f"Chunks to embed: {total}")

if total == 0:
    print("No work to do.")
    client.close()
    raise SystemExit(0)

cursor = col.find(
    query,
    {"_id": 1, FIELD_FULL_TEXT: 1, FIELD_TEXT: 1}
).batch_size(DOC_BATCH_SIZE)

ops = []
texts = []
ids = []

with tqdm(total=total, desc="Embedding chunks") as pbar:
    for doc in cursor:
        text = (doc.get(FIELD_FULL_TEXT) or doc.get(FIELD_TEXT) or "").strip()
        if not text:
            pbar.update(1)
            continue

        ids.append(doc["_id"])
        texts.append(text)

        if len(texts) >= DOC_BATCH_SIZE:
            vecs = embedder.encode(
                texts,
                batch_size=ENCODE_BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True
            )

            ops = [
                UpdateOne({"_id": _id}, {"$set": {FIELD_EMBEDDING: vec.tolist()}})
                for _id, vec in zip(ids, vecs)
            ]
            col.bulk_write(ops, ordered=False)

            pbar.update(len(texts))
            texts, ids, ops = [], [], []

    if texts:
        vecs = embedder.encode(
            texts,
            batch_size=ENCODE_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        ops = [
            UpdateOne({"_id": _id}, {"$set": {FIELD_EMBEDDING: vec.tolist()}})
            for _id, vec in zip(ids, vecs)
        ]
        col.bulk_write(ops, ordered=False)
        pbar.update(len(texts))

client.close()
print("Done")
