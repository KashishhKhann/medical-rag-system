import argparse
import csv
import json
import math
import re
import time
from difflib import SequenceMatcher
from typing import List, Dict, Any, Tuple

import numpy as np
from pymongo import MongoClient
from neo4j import GraphDatabase
import requests
import faiss
from sentence_transformers import SentenceTransformer, models
import spacy
from transformers import AutoTokenizer

from config import (
    MONGO_URI, DB_NAME, CHUNKS_COLLECTION,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    OLLAMA_URL, OLLAMA_MODEL,
    BERT_MODEL,
    USE_SPACY_SENT_SPLIT, SENTENCE_SPLIT_MODEL,
    USE_CHILD_SPANS, CHILD_SPAN_TARGET_TOKENS, CHILD_SPAN_MAX_TOKENS,
    CHILD_SPAN_MIN_TOKENS, CHILD_SPAN_OVERLAP_TOKENS, CHILD_SPAN_TOP_K,
    CHILD_SPAN_MAX_PER_PARENT,
    SECTION_BONUS_WEIGHT,
    USE_BM25_RERANK, BM25_WEIGHT, BM25_K1, BM25_B,
    OLLAMA_TIMEOUT_SEC,
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

def admission_scoped_candidates(coll, q_vec: np.ndarray, subject_id: int, hadm_id: int, pool_size: int):
    docs = list(coll.find(
        {"subject_id": subject_id, "hadm_id": hadm_id, "embedding": {"$exists": True}},
        {"_id": 0, "chunk_id": 1, "section": 1, "text": 1, "full_text": 1, "metadata": 1, "chunk_index": 1, "embedding": 1}
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
        d = docs[i]
        out.append({"chunk_id": d["chunk_id"], "faiss_score": float(scores[i]), "doc": d})
    return out

STOP_CONCEPTS = {
    "admission", "discharge", "patient", "hospital", "history",
    "mg", "mcg", "po", "iv", "tid", "bid", "qhs", "daily"
}

SECTION_HINTS = {
    "chief_complaint": [
        "chief complaint", "reason for admission", "reason for visit", "cc",
    ],
    "hpi": ["history of present illness", "hpi", "present illness"],
    "pmh": ["past medical history", "pmh"],
    "physical_exam": ["physical exam", "physical examination", "pe"],
    "hospital_course": ["hospital course", "brief hospital course"],
    "meds_discharge": [
        "discharge medications", "discharge meds", "medications on discharge",
        "meds on discharge", "discharge diagnosis", "discharge disposition",
    ],
    "header": ["allergies", "service admitted", "service"],
}

_SENT_SPLITTER = None
_TOKENIZER = None

def get_sentence_splitter():
    global _SENT_SPLITTER
    if _SENT_SPLITTER is not None:
        return _SENT_SPLITTER
    if not USE_SPACY_SENT_SPLIT:
        _SENT_SPLITTER = None
        return _SENT_SPLITTER
    try:
        nlp = spacy.load(SENTENCE_SPLIT_MODEL)
    except Exception:
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            _SENT_SPLITTER = None
            return _SENT_SPLITTER
    if not nlp.has_pipe("parser") and not nlp.has_pipe("senter"):
        nlp.add_pipe("sentencizer")
    _SENT_SPLITTER = nlp
    return _SENT_SPLITTER

def split_sentences(text: str) -> List[str]:
    nlp = get_sentence_splitter()
    if nlp is not None:
        doc = nlp(text)
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        if sents:
            return sents
    return [s.strip() for s in re.split(r"(?<=[\.!?])\s+", text) if s.strip()]

def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            _TOKENIZER = AutoTokenizer.from_pretrained(BERT_MODEL, use_fast=True)
        except Exception:
            _TOKENIZER = None
    return _TOKENIZER

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    tok = get_tokenizer()
    if tok is not None:
        return len(tok.encode(text, add_special_tokens=False))
    return len(re.findall(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]", text))

def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def tokenize_text(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text) if text else []

def max_fuzzy_ratio(concept_norm: str, q_tokens: List[str]) -> float:
    if not concept_norm or not q_tokens:
        return 0.0
    c_tokens = concept_norm.split()
    if not c_tokens:
        return 0.0
    n = len(c_tokens)
    if len(q_tokens) < n:
        span = " ".join(q_tokens)
        return SequenceMatcher(None, concept_norm, span).ratio() if span else 0.0
    best = 0.0
    for i in range(len(q_tokens) - n + 1):
        span = " ".join(q_tokens[i:i + n])
        ratio = SequenceMatcher(None, concept_norm, span).ratio()
        if ratio > best:
            best = ratio
    return best

def build_concept_freq(concepts_map: Dict[str, List[str]]) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for concepts in concepts_map.values():
        seen = set()
        for c in concepts:
            key = normalize_text(c)
            if not key or key in seen:
                continue
            seen.add(key)
            freq[key] = freq.get(key, 0) + 1
    return freq

def prepare_question(question: str) -> Tuple[str, List[str], set]:
    q_norm = normalize_text(question)
    q_tokens = tokenize_text(q_norm)
    q_token_set = {t for t in q_tokens if len(t) >= 3}
    return q_norm, q_tokens, q_token_set

def concept_match_score(concept_norm: str, q_norm: str, q_tokens: List[str], q_token_set: set) -> float:
    if not concept_norm or concept_norm in STOP_CONCEPTS:
        return 0.0
    if re.search(rf"\\b{re.escape(concept_norm)}\\b", q_norm):
        return 1.0
    c_tokens = [t for t in concept_norm.split() if len(t) >= 3]
    if not c_tokens:
        return 0.0
    overlap_ratio = len(set(c_tokens) & q_token_set) / len(set(c_tokens))
    score = 0.0
    if overlap_ratio >= 0.6:
        score = max(score, overlap_ratio)
    fuzzy_ratio = max_fuzzy_ratio(concept_norm, q_tokens)
    if fuzzy_ratio >= 0.82:
        score = max(score, fuzzy_ratio)
    return score

def tokenize_for_bm25(text: str) -> List[str]:
    norm = normalize_text(text)
    return [t for t in norm.split() if len(t) >= 2]

def bm25_scores(
    query_tokens: List[str],
    docs_tokens: List[List[str]],
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> List[float]:
    if not query_tokens or not docs_tokens:
        return [0.0 for _ in docs_tokens]

    df: Dict[str, int] = {}
    for tokens in docs_tokens:
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1

    N = len(docs_tokens)
    avgdl = sum(len(t) for t in docs_tokens) / max(1, N)

    scores: List[float] = []
    for tokens in docs_tokens:
        freqs: Dict[str, int] = {}
        for t in tokens:
            freqs[t] = freqs.get(t, 0) + 1
        dl = len(tokens)
        score = 0.0
        for t in query_tokens:
            f = freqs.get(t)
            if not f:
                continue
            idf = math.log(1.0 + (N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))
            denom = f + k1 * (1.0 - b + b * (dl / max(1.0, avgdl)))
            score += idf * (f * (k1 + 1.0)) / denom
        scores.append(score)
    return scores

def infer_section_bias(question: str) -> set:
    q_norm = normalize_text(question)
    if not q_norm:
        return set()
    matches = set()
    for section, terms in SECTION_HINTS.items():
        for term in terms:
            if " " in term:
                if term in q_norm:
                    matches.add(section)
                    break
            else:
                if re.search(rf"\\b{re.escape(term)}\\b", q_norm):
                    matches.add(section)
                    break
    return matches

def init_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def get_concepts_for_chunks(driver, chunk_ids: List[str]) -> Dict[str, List[str]]:
    if not chunk_ids:
        return {}
    q = """
    MATCH (c:Chunk)-[:MENTIONS_CONCEPT]->(e:Concept)
    WHERE c.chunk_id IN $cids
    RETURN c.chunk_id AS cid, collect(DISTINCT e.name) AS concepts
    """
    out = {cid: [] for cid in chunk_ids}
    with driver.session() as s:
        for r in s.run(q, cids=chunk_ids):
            cid = r.get("cid")
            concepts = r.get("concepts") or []
            out[cid] = [c for c in concepts if c]
    return out

def kg_overlap_stats(
    concepts: List[str],
    q_norm: str,
    q_tokens: List[str],
    q_token_set: set,
    concept_freq: Dict[str, int],
) -> Tuple[int, float]:
    if not concepts or not q_norm:
        return 0, 0.0

    matched = 0
    score = 0.0
    seen = set()

    for c in concepts:
        key = normalize_text(c)
        if not key or key in seen:
            continue
        seen.add(key)

        match_score = concept_match_score(key, q_norm, q_tokens, q_token_set)
        if match_score <= 0:
            continue

        freq = concept_freq.get(key, 1)
        penalty = 1.0 / (1.0 + math.log1p(freq))

        matched += 1
        score += match_score * penalty

    return matched, float(math.log1p(score))

def minmax(values: List[float]) -> List[float]:
    if not values:
        return []
    mn, mx = min(values), max(values)
    if mx == mn:
        return [0.0] * len(values)
    return [(v - mn) / (mx - mn + 1e-9) for v in values]

def rank_sim_only(cands: List[Dict[str, Any]], top_k: int, question: str):
    if not cands:
        return []
    section_bias = infer_section_bias(question)
    sims = [c["faiss_score"] for c in cands]
    sims_n = minmax(sims)
    bm25_q = tokenize_for_bm25(question)
    bm25_docs = [
        tokenize_for_bm25((c["doc"].get("text") or c["doc"].get("full_text") or ""))
        for c in cands
    ]
    bm25_vals = bm25_scores(bm25_q, bm25_docs) if USE_BM25_RERANK else [0.0] * len(cands)
    bm25_n = minmax(bm25_vals)

    enriched = []
    for c, sn, bn in zip(cands, sims_n, bm25_n):
        bonus = 1.0 if c["doc"].get("section") in section_bias else 0.0
        c_score = sn + (SECTION_BONUS_WEIGHT * bonus) + (BM25_WEIGHT * bn)
        enriched.append({**c, "hybrid_score": c_score})

    enriched.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return enriched[:top_k]

def rank_sim_plus_kg(cands: List[Dict[str, Any]], concepts_map: Dict[str, List[str]], question: str,
                     top_k: int, alpha: float, beta: float):
    section_bias = infer_section_bias(question)
    concept_freq = build_concept_freq(concepts_map)
    q_norm, q_tokens, q_token_set = prepare_question(question)
    bm25_q = tokenize_for_bm25(question)
    bm25_docs = [
        tokenize_for_bm25((c["doc"].get("text") or c["doc"].get("full_text") or ""))
        for c in cands
    ]
    bm25_vals = bm25_scores(bm25_q, bm25_docs) if USE_BM25_RERANK else [0.0] * len(cands)
    enriched = []
    for c in cands:
        cid = c["chunk_id"]
        concepts = concepts_map.get(cid, [])
        matched, kg_score = kg_overlap_stats(
            concepts, q_norm, q_tokens, q_token_set, concept_freq
        )
        section_bonus = 1.0 if c["doc"].get("section") in section_bias else 0.0
        enriched.append({
            **c,
            "concepts": concepts,
            "kg_score": kg_score,
            "kg_matched": matched,
            "section_bonus": section_bonus,
        })

    sims = [e["faiss_score"] for e in enriched]
    kgs = [e["kg_score"] for e in enriched]
    sims_n = minmax(sims)
    kgs_n = minmax(kgs)
    bm25_n = minmax(bm25_vals)

    for e, sn, kn, bn in zip(enriched, sims_n, kgs_n, bm25_n):
        e["hybrid_score"] = (
            alpha * sn
            + beta * kn
            + (SECTION_BONUS_WEIGHT * e["section_bonus"])
            + (BM25_WEIGHT * bn)
        )

    enriched.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return enriched[:top_k]

def build_child_spans(
    text: str,
    target_tokens: int,
    max_tokens: int,
    min_tokens: int,
    overlap_tokens: int,
    max_spans: int,
) -> List[Dict[str, Any]]:
    sentences = split_sentences(text or "")
    if not sentences:
        return []

    spans: List[Dict[str, Any]] = []
    i = 0
    while i < len(sentences) and len(spans) < max_spans:
        buf: List[str] = []
        buf_tokens = 0
        j = i

        while j < len(sentences):
            sent = sentences[j]
            st = estimate_tokens(sent)
            if buf and buf_tokens + st > max_tokens:
                break
            buf.append(sent)
            buf_tokens += st
            j += 1
            if buf_tokens >= target_tokens:
                break

        if not buf:
            i += 1
            continue

        if buf_tokens < min_tokens and j < len(sentences):
            sent = sentences[j]
            st = estimate_tokens(sent)
            if buf_tokens + st <= max_tokens:
                buf.append(sent)
                buf_tokens += st
                j += 1

        span_text = " ".join(buf).strip()
        if span_text:
            spans.append(
                {"text": span_text, "start": i, "end": j, "tokens": buf_tokens}
            )

        if j >= len(sentences):
            break

        if overlap_tokens > 0:
            total = buf_tokens
            consumed = 0
            new_i = i
            for k in range(i, j):
                consumed += estimate_tokens(sentences[k])
                if total - consumed <= overlap_tokens:
                    new_i = k
                    break
            if new_i <= i:
                new_i = min(i + 1, j)
            i = new_i
        else:
            i = j

    return spans

def refine_with_child_spans(
    ranked: List[Dict[str, Any]],
    q_vec: np.ndarray,
    embedder: SentenceTransformer,
    child_top_k: int = CHILD_SPAN_TOP_K,
) -> List[Dict[str, Any]]:
    spans: List[Dict[str, Any]] = []

    for cand in ranked:
        doc = cand["doc"]
        text = (doc.get("text") or doc.get("full_text") or "").strip()
        if not text:
            continue
        for idx, span in enumerate(
            build_child_spans(
                text,
                target_tokens=CHILD_SPAN_TARGET_TOKENS,
                max_tokens=CHILD_SPAN_MAX_TOKENS,
                min_tokens=CHILD_SPAN_MIN_TOKENS,
                overlap_tokens=CHILD_SPAN_OVERLAP_TOKENS,
                max_spans=CHILD_SPAN_MAX_PER_PARENT,
            )
        ):
            spans.append({**cand, "span": {**span, "idx": idx}, "span_text": span["text"]})

    if not spans:
        return ranked

    texts = [s["span_text"] for s in spans]
    vecs = embedder.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    vecs = np.asarray(vecs, dtype="float32")
    faiss.normalize_L2(vecs)
    scores = (vecs @ q_vec.T).reshape(-1)

    for s, sc in zip(spans, scores):
        s["span_score"] = float(sc)

    spans.sort(key=lambda x: x["span_score"], reverse=True)
    top_spans = spans[: max(1, child_top_k)]

    out: List[Dict[str, Any]] = []
    for s in top_spans:
        doc = dict(s["doc"])
        doc["text"] = s["span_text"]
        doc["full_text"] = s["span_text"]
        meta = dict(doc.get("metadata", {}) or {})
        meta["span_start"] = s["span"]["start"]
        meta["span_end"] = s["span"]["end"]
        meta["span_tokens"] = s["span"]["tokens"]
        meta["span_idx"] = s["span"]["idx"]
        doc["metadata"] = meta
        out.append({**s, "doc": doc})

    return out

def short_text(doc: Dict[str, Any], width: int = 420) -> str:
    txt = (doc.get("text") or doc.get("full_text") or "").replace("\n", " ").strip()
    return txt if len(txt) <= width else txt[:width] + " ..."

def build_context(ranked: List[Dict[str, Any]], include_concepts: bool = True) -> str:
    blocks = []
    for i, c in enumerate(ranked, start=1):
        d = c["doc"]
        section = d.get("section", "unknown")
        chunk_id = d.get("chunk_id", "")
        meta = d.get("metadata", {}) or {}
        temporal = meta.get("temporal", "unknown")
        snippet = short_text(d, width=420)
        span_idx = meta.get("span_idx")

        lines = [
            f"### [{i}] chunk_id={chunk_id} | section={section} | temporal={temporal}",
        ]
        if span_idx is not None:
            span_start = meta.get("span_start")
            span_end = meta.get("span_end")
            span_tokens = meta.get("span_tokens")
            lines.append(
                f"SPAN: idx={span_idx} | sents={span_start}-{span_end} | tokens={span_tokens}"
            )
        lines.append(f"SNIPPET: {snippet}")
        if include_concepts and c.get("concepts"):
            uniq = sorted(set(c["concepts"]))
            lines.append("CONCEPTS: " + "; ".join(uniq[:30]))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)

def call_ollama(question: str, context: str, subject_id: int, hadm_id: int,
               temperature: float = 0.2, num_predict: int = 256) -> Tuple[str, float]:
    system = """You are a medical question answering system.
Use ONLY the provided context. If information is missing, say so.
Answer in 3-6 sentences, then bullet-point reasoning, then cite chunk_ids used.
"""
    prompt = f"""{system}

QUESTION:
{question}

Patient: {subject_id}
Admission: {hadm_id}

CONTEXT:
{context}

Return format:
1) Short Answer
2) Reasoning (bullets)
3) Evidence (bullets with [chunk:...])
"""
    t0 = time.perf_counter()
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": num_predict},
        },
        timeout=OLLAMA_TIMEOUT_SEC,
    )
    resp.raise_for_status()
    out = resp.json().get("response", "").strip()
    t1 = time.perf_counter()
    return out, (t1 - t0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="eval_tasks_50.csv")
    ap.add_argument("--out", required=True, help="output CSV with answers")
    ap.add_argument("--variant", choices=["sim_only", "sim_plus_kg"], default="sim_plus_kg")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--pool_mult", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--beta", type=float, default=0.15)
    ap.add_argument("--max_seq_length", type=int, default=256)
    ap.add_argument("--num_predict", type=int, default=256)
    ap.add_argument("--use_child_spans", action="store_true", default=USE_CHILD_SPANS)
    ap.add_argument("--no_child_spans", action="store_true")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    use_child_spans = args.use_child_spans and not args.no_child_spans

    mongo = MongoClient(MONGO_URI)
    coll = mongo[DB_NAME][CHUNKS_COLLECTION]
    embedder = build_embedder(BERT_MODEL, max_seq_length=args.max_seq_length)

    neo4j_driver = init_neo4j_driver() if args.variant == "sim_plus_kg" else None

    tasks = []
    with open(args.tasks, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tasks.append(r)

    tasks = tasks[args.start:] if args.start else tasks
    if args.limit and args.limit > 0:
        tasks = tasks[:args.limit]

    out_rows = []

    try:
        for t in tasks:
            case_id = t.get("case_id", "")
            task_name = t.get("task_name", "")
            subject_id = int(t["subject_id"])
            hadm_id = int(t["hadm_id"])
            question = t["question"]
            targets = t.get("target_sections", "")

            top_k = int(t.get("top_k") or args.top_k)

            t0 = time.perf_counter()
            q_vec = embed_query(embedder, question)
            pool_size = max(top_k * args.pool_mult, 30)
            cands = admission_scoped_candidates(coll, q_vec, subject_id, hadm_id, pool_size=pool_size)
            t1 = time.perf_counter()

            if not cands:
                out_rows.append({
                    "case_id": case_id,
                    "task_name": task_name,
                    "subject_id": subject_id,
                    "hadm_id": hadm_id,
                    "question": question,
                    "target_sections": targets,
                    "variant": args.variant,
                    "retrieved_sections": "",
                    "retrieved_chunk_ids": "",
                    "answer": "",
                    "llm_time_sec": "",
                    "embed_plus_retrieval_sec": round(t1 - t0, 4),
                    "neo4j_time_sec": "",
                    "notes": "NO_CANDIDATES_WITHIN_ADMISSION",
                })
                continue

            neo_t = ""
            if args.variant == "sim_only":
                ranked = rank_sim_only(cands, top_k=top_k, question=question)
            else:
                t2 = time.perf_counter()
                chunk_ids = [c["chunk_id"] for c in cands]
                concepts_map = get_concepts_for_chunks(neo4j_driver, chunk_ids)
                ranked = rank_sim_plus_kg(cands, concepts_map, question, top_k=top_k, alpha=args.alpha, beta=args.beta)
                t3 = time.perf_counter()
                neo_t = round(t3 - t2, 4)

            if use_child_spans:
                ranked = refine_with_child_spans(
                    ranked, q_vec, embedder, child_top_k=min(CHILD_SPAN_TOP_K, top_k)
                )

            retrieved_sections = [c["doc"].get("section", "unknown") for c in ranked]
            retrieved_chunk_ids = [c["chunk_id"] for c in ranked]

            context = build_context(ranked, include_concepts=(args.variant == "sim_plus_kg"))
            answer, llm_time = call_ollama(question, context, subject_id, hadm_id, num_predict=args.num_predict)

            out_rows.append({
                "case_id": case_id,
                "task_name": task_name,
                "subject_id": subject_id,
                "hadm_id": hadm_id,
                "question": question,
                "target_sections": targets,
                "variant": args.variant,
                "retrieved_sections": "|".join(retrieved_sections),
                "retrieved_chunk_ids": "|".join(retrieved_chunk_ids),
                "answer": answer,
                "llm_time_sec": round(llm_time, 4),
                "embed_plus_retrieval_sec": round(t1 - t0, 4),
                "neo4j_time_sec": neo_t,
                "notes": "",
            })

        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else [])
            if out_rows:
                w.writeheader()
                w.writerows(out_rows)

        print(f"Wrote {args.out} rows={len(out_rows)}")

    finally:
        if neo4j_driver:
            neo4j_driver.close()
        mongo.close()

if __name__ == "__main__":
    main()
