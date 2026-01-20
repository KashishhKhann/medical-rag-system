"""
Medical RAG Query System

This system answers medical questions by:
1. Finding similar chunks using vector search (FAISS) OR scoped ranking in Mongo (patient/admission)
2. Reranking candidates using knowledge graph concepts (Neo4j) with query-aware KG scoring
3. Asking an LLM (Ollama) to generate an answer

You can filter results by patient ID or admission ID (either one or both).
"""

import os
import json
import textwrap
import re
import math
from difflib import SequenceMatcher
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from pymongo import MongoClient
from neo4j import GraphDatabase
import faiss
import requests
from sentence_transformers import SentenceTransformer, models
import spacy
from transformers import AutoTokenizer

from config import (
    MONGO_URI, DB_NAME, CHUNKS_COLLECTION,
    FAISS_INDEX_PATH, FAISS_MAP_PATH,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    OLLAMA_URL, OLLAMA_MODEL, BERT_MODEL,
    FIELD_CHUNK_ID, FIELD_SUBJECT_ID, FIELD_HADM_ID,
    FIELD_SECTION, FIELD_TEXT, FIELD_METADATA,
    USE_SPACY_SENT_SPLIT, SENTENCE_SPLIT_MODEL,
    USE_CHILD_SPANS, CHILD_SPAN_TARGET_TOKENS, CHILD_SPAN_MAX_TOKENS,
    CHILD_SPAN_MIN_TOKENS, CHILD_SPAN_OVERLAP_TOKENS, CHILD_SPAN_TOP_K,
    CHILD_SPAN_MAX_PER_PARENT,
    SECTION_BONUS_WEIGHT,
    USE_BM25_RERANK, BM25_WEIGHT, BM25_K1, BM25_B,
    OLLAMA_TIMEOUT_SEC,
)

MONGO_DB = DB_NAME
MONGO_COLLECTION = CHUNKS_COLLECTION

def init_mongo():
    client = MongoClient(MONGO_URI)
    return client, client[MONGO_DB][MONGO_COLLECTION]


def init_faiss() -> Tuple[faiss.Index, Dict[int, str]]:
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"Missing FAISS index at {FAISS_INDEX_PATH}")
    if not os.path.exists(FAISS_MAP_PATH):
        raise FileNotFoundError(f"Missing FAISS mapping at {FAISS_MAP_PATH}")

    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_MAP_PATH, "r", encoding="utf-8") as f:
        mapping_raw = json.load(f)
    mapping = {int(k): v for k, v in mapping_raw.items()}

    print(f"FAISS Loaded | dim={index.d} | entries={len(mapping)}")
    return index, mapping


def init_embedder():
    """
    Force HF checkpoint + CLS pooling ONLY (no mean pooling).
    This avoids SentenceTransformer auto-wrapping with mean pooling.
    """
    print(f"\nLoading embedding model (HF checkpoint): {BERT_MODEL}")

    word = models.Transformer(BERT_MODEL, max_seq_length=256)
    pool = models.Pooling(
        word.get_word_embedding_dimension(),
        pooling_mode_cls_token=True,
        pooling_mode_mean_tokens=False,
        pooling_mode_max_tokens=False,
    )
    model = SentenceTransformer(modules=[word, pool])

    print(
        f"Pooling: CLS={pool.pooling_mode_cls_token} "
        f"MEAN={pool.pooling_mode_mean_tokens} "
        f"MAX={pool.pooling_mode_max_tokens}"
    )
    print(f"Embedding dim: {model.get_sentence_embedding_dimension()}")
    return model


def init_neo4j():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def embed_query(model, text: str) -> np.ndarray:
    vec = model.encode(text)
    q = np.asarray(vec, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(q)
    return q


def faiss_candidates(
    index: faiss.Index,
    mapping: Dict[int, str],
    q_vec: np.ndarray,
    oversample: int = 60,
) -> List[Dict[str, Any]]:
    """
    Global FAISS search across all chunks.
    """
    D, I = index.search(q_vec, oversample)
    hits: List[Dict[str, Any]] = []
    for idx, dist in zip(I[0], D[0]):
        if idx == -1:
            continue
        cid = mapping.get(int(idx))
        if not cid:
            continue
        hits.append({"faiss_id": int(idx), "chunk_id": cid, "faiss_score": float(dist)})
    return hits


def attach_docs_and_filter(
    hits: List[Dict[str, Any]],
    coll,
    top_k: int,
    subject_id: Optional[int],
    hadm_id: Optional[int],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for h in hits:
        cid = h["chunk_id"]
        doc = coll.find_one({"chunk_id": cid})
        if not doc:
            continue

        if subject_id is not None and doc.get("subject_id") != subject_id:
            continue
        if hadm_id is not None and doc.get("hadm_id") != hadm_id:
            continue

        out.append({"chunk_id": cid, "faiss_score": h["faiss_score"], "doc": doc})

        if len(out) >= top_k * 5:
            break

    if not out and subject_id is not None and hadm_id is not None:
        print("No exact matches for that patient+admission. Trying patient-only fallback...\n")
        for h in hits:
            cid = h["chunk_id"]
            doc = coll.find_one({"chunk_id": cid})
            if not doc:
                continue
            if doc.get("subject_id") != subject_id:
                continue
            out.append({"chunk_id": cid, "faiss_score": h["faiss_score"], "doc": doc})
            if len(out) >= top_k * 5:
                break

    return out


def scoped_mongo_candidates(
    coll,
    q_vec: np.ndarray,
    subject_id: Optional[int],
    hadm_id: Optional[int],
    pool_size: int,
    max_docs: Optional[int] = None,
) -> List[Dict[str, Any]]:
    query: Dict[str, Any] = {"embedding": {"$exists": True}}
    if subject_id is not None:
        query["subject_id"] = subject_id
    if hadm_id is not None:
        query["hadm_id"] = hadm_id

    proj = {
        "_id": 0,
        "chunk_id": 1,
        "embedding": 1,
        "section": 1,
        "text": 1,
        "metadata": 1,
        "subject_id": 1,
        "hadm_id": 1,
    }

    cursor = coll.find(query, proj)
    if max_docs is not None:
        cursor = cursor.limit(int(max_docs))

    docs: List[Dict[str, Any]] = []
    embs: List[List[float]] = []

    for d in cursor:
        emb = d.get("embedding")
        if isinstance(emb, list) and len(emb) > 0:
            docs.append(d)
            embs.append(emb)

    if not docs:
        return []

    X = np.asarray(embs, dtype="float32")
    faiss.normalize_L2(X)

    tmp = faiss.IndexFlatIP(X.shape[1])
    tmp.add(X)

    k = min(pool_size, len(docs))
    D, I = tmp.search(q_vec, k)

    out: List[Dict[str, Any]] = []
    for j, i in enumerate(I[0]):
        out.append({
            "chunk_id": docs[i]["chunk_id"],
            "faiss_score": float(D[0][j]),
            "doc": docs[i],
        })
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

def get_concepts_for_chunks(driver, chunk_ids: List[str]) -> Dict[str, List[str]]:
    """
    Batch fetch concepts for many chunks in a single Neo4j query.
    """
    if not chunk_ids:
        return {}

    q = """
    MATCH (c:Chunk)-[:MENTIONS_CONCEPT]->(e:Concept)
    WHERE c.chunk_id IN $cids
    RETURN c.chunk_id AS cid, collect(DISTINCT e.name) AS concepts
    """

    out: Dict[str, List[str]] = {cid: [] for cid in chunk_ids}
    with driver.session() as s:
        for r in s.run(q, cids=chunk_ids):
            cid = r.get("cid")
            concepts = r.get("concepts") or []
            out[cid] = [c for c in concepts if c]
    return out


def compute_kg_score(
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


def hybrid_rank(
    candidates: List[Dict[str, Any]],
    driver,
    question: str,
    top_k: int,
    alpha: float = 0.85,
    beta: float = 0.15,
) -> List[Dict[str, Any]]:
    """
    Combine FAISS similarity with query-aware KG overlap score.
    """
    if not candidates:
        return []

    chunk_ids = [c["chunk_id"] for c in candidates]
    concepts_map = get_concepts_for_chunks(driver, chunk_ids)
    concept_freq = build_concept_freq(concepts_map)
    section_bias = infer_section_bias(question)
    q_norm, q_tokens, q_token_set = prepare_question(question)
    bm25_q = tokenize_for_bm25(question)
    bm25_docs = [
        tokenize_for_bm25((c["doc"].get("text") or c["doc"].get("full_text") or ""))
        for c in candidates
    ]
    bm25_vals = bm25_scores(bm25_q, bm25_docs) if USE_BM25_RERANK else [0.0] * len(candidates)

    enriched: List[Dict[str, Any]] = []
    for c in candidates:
        cid = c["chunk_id"]
        concepts = concepts_map.get(cid, [])
        kg_matched, kg_score = compute_kg_score(
            concepts, q_norm, q_tokens, q_token_set, concept_freq
        )
        section_bonus = 1.0 if c["doc"].get("section") in section_bias else 0.0
        enriched.append(
            {
                **c,
                "concepts": concepts,
                "kg_score": kg_score,
                "kg_matched": kg_matched,
                "section_bonus": section_bonus,
            }
        )

    faiss_vals = [e["faiss_score"] for e in enriched]
    kg_vals = [e["kg_score"] for e in enriched]

    def minmax(values: List[float]) -> List[float]:
        if not values:
            return []
        mn, mx = min(values), max(values)
        if mx == mn:
            return [0.0] * len(values)
        return [(v - mn) / (mx - mn + 1e-9) for v in values]

    faiss_norm = minmax(faiss_vals)
    kg_norm = minmax(kg_vals)
    bm25_norm = minmax(bm25_vals)

    for e, fn, kn, bn in zip(enriched, faiss_norm, kg_norm, bm25_norm):
        e["hybrid_score"] = (
            alpha * fn
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


def build_context_block(rank_idx: int, cand: Dict[str, Any]) -> str:
    doc = cand["doc"]
    concepts = cand.get("concepts", [])

    chunk_id = doc.get("chunk_id")
    section = doc.get("section", "unknown")
    meta = doc.get("metadata", {}) or {}
    temporal = meta.get("temporal", "unknown")

    subject_id = doc.get("subject_id")
    hadm_id = doc.get("hadm_id")

    text = doc.get("text", "") or ""
    snippet = textwrap.shorten(text.replace("\n", " "), width=420, placeholder=" ...")
    span_meta = meta.get("span_idx")

    lines: List[str] = []
    lines.append(
        f"### [{rank_idx}] chunk_id={chunk_id} "
        f"| patient={subject_id} | hadm={hadm_id} "
        f"| section={section} | temporal={temporal}"
    )
    if span_meta is not None:
        span_start = meta.get("span_start")
        span_end = meta.get("span_end")
        span_tokens = meta.get("span_tokens")
        lines.append(
            f"SPAN: idx={span_meta} | sents={span_start}-{span_end} | tokens={span_tokens}"
        )
    lines.append(f"TEXT SNIPPET:\n{snippet}\n")

    if concepts:
        lines.append("CONCEPTS: " + "; ".join(sorted(set(concepts))))

    return "\n".join(lines)


def build_full_context(cands: List[Dict[str, Any]]) -> str:
    return "\n\n".join(build_context_block(i, c) for i, c in enumerate(cands, start=1))


def call_llm(question: str, context: str, subject_id=None, hadm_id=None) -> str:
    system = """
You are a medical question answering system.

You will be given:
- Medical note chunks with patient info
- Medical concepts from those notes

Your job:
1. Answer the question in 3-6 sentences
2. Explain your reasoning with bullet points
3. List the chunks you used as evidence

DO NOT make up information.
If you're not sure, say so.
"""

    prompt = f"""{system}

------------------------------------------------
CLINICAL QUESTION:
{question}

Patient filter:   {subject_id if subject_id is not None else "all"}
Admission filter: {hadm_id if hadm_id is not None else "all"}
------------------------------------------------

RETRIEVED CONTEXT:
{context}
------------------------------------------------

Now answer in this structure:

1. Short Answer
2. Reasoning (bullet points)
3. Evidence Summary (bullet list with [chunk:chunk_id])

ANSWER:
"""

    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 512},
        },
        timeout=OLLAMA_TIMEOUT_SEC,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()


def kg_rag_query(
    question: str,
    top_k: int = 5,
    subject_id: Optional[int] = None,
    hadm_id: Optional[int] = None,
):
    print("\n---")
    print("Medical Question Answering System")
    print("---")
    print(f"MongoDB:   {MONGO_URI}  DB={MONGO_DB}  Coll={MONGO_COLLECTION}")
    print(f"Neo4j:     {NEO4J_URI}")
    print(f"FAISS:     {FAISS_INDEX_PATH}")
    print(f"LLM:       {OLLAMA_MODEL}")
    print("---")

    mongo_client, coll = init_mongo()
    index, mapping = init_faiss()
    embedder = init_embedder()
    driver = init_neo4j()

    try:
        print(f"\nYour question: {question}")
        if subject_id is not None or hadm_id is not None:
            print(f"   Filter: patient={subject_id} | admission={hadm_id}")
        print()

        q_vec = embed_query(embedder, question)

        if subject_id is not None or hadm_id is not None:
            pool_size = max(top_k * 10, 50)

            candidates = scoped_mongo_candidates(
                coll,
                q_vec,
                subject_id=subject_id,
                hadm_id=hadm_id,
                pool_size=pool_size,
            )

            if not candidates:
                hits_raw = faiss_candidates(index, mapping, q_vec, oversample=max(top_k * 50, 200))
                if not hits_raw:
                    print("No similar chunks found.")
                    return

                candidates = attach_docs_and_filter(
                    hits_raw, coll, top_k=top_k, subject_id=subject_id, hadm_id=hadm_id
                )

                if not candidates:
                    print("No matching chunks after filtering.")
                    print(f"Total hits found (global): {len(hits_raw)}")
                    print("Try: provide a patient OR an admission filter, or remove filters.")
                    return

        else:
            hits_raw = faiss_candidates(index, mapping, q_vec, oversample=max(top_k * 50, 200))
            if not hits_raw:
                print("No similar chunks found.")
                return

            candidates = attach_docs_and_filter(hits_raw, coll, top_k=top_k, subject_id=None, hadm_id=None)
            if not candidates:
                print("No matching chunks after filtering.")
                return

        ranked = hybrid_rank(candidates, driver, question, top_k=top_k)
        if USE_CHILD_SPANS:
            ranked = refine_with_child_spans(
                ranked, q_vec, embedder, child_top_k=min(CHILD_SPAN_TOP_K, top_k)
            )
        print(f"Using {len(ranked)} best chunks.\n")

        context = build_full_context(ranked)
        answer = call_llm(question, context, subject_id, hadm_id)

        print("\n--- ANSWER ---\n")
        print(answer)
        print("\n---\n")

    finally:
        mongo_client.close()
        driver.close()


if __name__ == "__main__":
    print("\n--- Medical Question Answering System ---\n")

    try:
        q = input("\nWhat's your medical question?\n> ").strip()
        if not q:
            print("No question entered.")
            raise SystemExit

        pid_str = input("\nPatient ID (optional, leave blank for all): ").strip()
        hadm_str = input("Admission ID (optional, leave blank for all): ").strip()

        subject_id = int(pid_str) if pid_str.isdigit() else None
        hadm_id = int(hadm_str) if hadm_str.isdigit() else None

        kg_rag_query(q, top_k=25, subject_id=subject_id, hadm_id=hadm_id)

    except KeyboardInterrupt:
        print("\nCancelled.")
