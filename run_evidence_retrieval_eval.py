import argparse
import csv
import json
import os
import time
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from pymongo import MongoClient
import faiss
from sentence_transformers import SentenceTransformer, models

from config import (
    MONGO_URI, DB_NAME, CHUNKS_COLLECTION,
    FAISS_INDEX_PATH, FAISS_MAP_PATH,
    BERT_MODEL,
)

def build_embedder(model_name: str, max_seq_length: int = 256) -> SentenceTransformer:
    word = models.Transformer(model_name, max_seq_length=max_seq_length)
    pool = models.Pooling(
        word.get_word_embedding_dimension(),
        pooling_mode_cls_token=True,
        pooling_mode_mean_tokens=False,
        pooling_mode_max_tokens=False,
    )
    return SentenceTransformer(modules=[word, pool])

def embed_query(embedder: SentenceTransformer, text: str) -> np.ndarray:
    v = embedder.encode(text)
    q = np.asarray(v, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(q)
    return q

def admission_scoped_candidates(
    coll,
    q_vec: np.ndarray,
    subject_id: int,
    hadm_id: int,
    pool_size: int,
) -> List[Dict[str, Any]]:
    docs = list(coll.find(
        {"subject_id": subject_id, "hadm_id": hadm_id, "embedding": {"$exists": True}},
        {"_id": 0, "chunk_id": 1, "section": 1, "embedding": 1}
    ))
    docs = [d for d in docs if isinstance(d.get("embedding"), list) and len(d["embedding"]) > 0]
    if not docs:
        return []

    X = np.asarray([d["embedding"] for d in docs], dtype="float32")
    faiss.normalize_L2(X)
    scores = (X @ q_vec.T).reshape(-1)

    idx = np.argsort(-scores)[:pool_size]
    out = []
    for i in idx:
        out.append({
            "chunk_id": docs[i]["chunk_id"],
            "score": float(scores[i]),
            "section": docs[i].get("section", "unknown"),
        })
    return out

def init_faiss() -> Tuple[faiss.Index, Dict[int, str]]:
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"Missing FAISS index at {FAISS_INDEX_PATH}")
    if not os.path.exists(FAISS_MAP_PATH):
        raise FileNotFoundError(f"Missing FAISS mapping at {FAISS_MAP_PATH}")

    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_MAP_PATH, "r", encoding="utf-8") as f:
        mapping_raw = json.load(f)
    mapping = {int(k): v for k, v in mapping_raw.items()}
    return index, mapping

def global_faiss_candidates(
    index: faiss.Index,
    mapping: Dict[int, str],
    q_vec: np.ndarray,
    oversample: int,
) -> List[Dict[str, Any]]:
    D, I = index.search(q_vec, oversample)
    hits = []
    for idx, dist in zip(I[0], D[0]):
        if idx == -1:
            continue
        cid = mapping.get(int(idx))
        if not cid:
            continue
        hits.append({"chunk_id": cid, "score": float(dist)})
    return hits

def attach_sections_from_mongo(coll, chunk_ids: List[str]) -> Dict[str, str]:
    if not chunk_ids:
        return {}
    docs = coll.find({"chunk_id": {"$in": chunk_ids}}, {"_id": 0, "chunk_id": 1, "section": 1})
    return {d["chunk_id"]: d.get("section", "unknown") for d in docs}

def compute_metrics(retrieved_sections: List[str], target_sections: set, total_relevant: int) -> Dict[str, float]:
    if not retrieved_sections:
        return {"hit_at_k": 0.0, "precision_at_k": 0.0, "coverage_at_k": 0.0}

    hits = sum(1 for s in retrieved_sections if s in target_sections)
    hit_at_k = 1.0 if hits > 0 else 0.0
    precision_at_k = hits / len(retrieved_sections)

    if total_relevant > 0:
        coverage_at_k = hits / total_relevant
    else:
        coverage_at_k = 1.0 if hits > 0 else 0.0

    return {
        "hit_at_k": float(hit_at_k),
        "precision_at_k": float(precision_at_k),
        "coverage_at_k": float(coverage_at_k),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="eval_tasks_50.csv")
    ap.add_argument("--out", required=True, help="output CSV")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--pool_mult", type=int, default=10, help="within-admission pool multiplier")
    ap.add_argument("--oversample", type=int, default=400, help="global FAISS oversample")
    ap.add_argument("--max_seq_length", type=int, default=256)
    args = ap.parse_args()

    mongo = MongoClient(MONGO_URI)
    coll = mongo[DB_NAME][CHUNKS_COLLECTION]

    embedder = build_embedder(BERT_MODEL, max_seq_length=args.max_seq_length)
    faiss_index, faiss_map = init_faiss()

    tasks = []
    with open(args.tasks, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tasks.append(r)

    rows_out = []

    for t in tasks:
        case_id = t.get("case_id", "")
        task_name = t.get("task_name", "")
        subject_id = int(t["subject_id"])
        hadm_id = int(t["hadm_id"])
        question = t["question"]
        target_sections = set((t.get("target_sections") or "").split("|"))
        top_k = int(t.get("top_k") or args.top_k)

        total_relevant = coll.count_documents({
            "subject_id": subject_id, "hadm_id": hadm_id,
            "section": {"$in": list(target_sections)}
        })

        t0 = time.perf_counter()
        q_vec = embed_query(embedder, question)
        t1 = time.perf_counter()

        pool_size = max(top_k * args.pool_mult, 30)
        t2 = time.perf_counter()
        within = admission_scoped_candidates(coll, q_vec, subject_id, hadm_id, pool_size=pool_size)
        within_top = within[:top_k]
        t3 = time.perf_counter()

        within_sections = [x["section"] for x in within_top]
        m_within = compute_metrics(within_sections, target_sections, total_relevant)

        rows_out.append({
            "mode": "within_admission",
            "case_id": case_id,
            "task_name": task_name,
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "top_k": top_k,
            "targets": "|".join(sorted(target_sections)),
            "total_relevant_in_admission": int(total_relevant),
            "retrieved_sections": "|".join(within_sections),
            "retrieved_chunk_ids": "|".join([x["chunk_id"] for x in within_top]),
            **{k: round(v, 4) for k, v in m_within.items()},
            "embed_time_sec": round(t1 - t0, 4),
            "retrieval_time_sec": round(t3 - t2, 4),
        })

        t4 = time.perf_counter()
        global_hits = global_faiss_candidates(faiss_index, faiss_map, q_vec, oversample=args.oversample)
        global_top = global_hits[:top_k]
        t5 = time.perf_counter()

        ids = [x["chunk_id"] for x in global_top]
        section_map = attach_sections_from_mongo(coll, ids)
        global_sections = [section_map.get(cid, "unknown") for cid in ids]

        m_global = compute_metrics(global_sections, target_sections, total_relevant=0)

        rows_out.append({
            "mode": "global_no_filter",
            "case_id": case_id,
            "task_name": task_name,
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "top_k": top_k,
            "targets": "|".join(sorted(target_sections)),
            "total_relevant_in_admission": int(total_relevant),
            "retrieved_sections": "|".join(global_sections),
            "retrieved_chunk_ids": "|".join(ids),
            **{k: round(v, 4) for k, v in m_global.items()},
            "embed_time_sec": round(t1 - t0, 4),
            "retrieval_time_sec": round(t5 - t4, 4),
        })

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    print(f"Wrote {args.out} rows={len(rows_out)}")
    mongo.close()

if __name__ == "__main__":
    main()
