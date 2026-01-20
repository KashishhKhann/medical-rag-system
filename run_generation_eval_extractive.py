import argparse
import csv
import time
from typing import List, Dict, Any, Optional, Tuple

from pymongo import MongoClient
import numpy as np
import requests

from rouge_score import rouge_scorer
from bert_score import score as bertscore

from config import (
    MONGO_URI, DB_NAME, CHUNKS_COLLECTION,
    OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT_SEC
)

EXTRACTIVE_TASKS_ALLOWLIST = {
    "meds_discharge_list",
    "physical_exam_results",
    "chief_complaint_reason",
    "pmh_list",
}

def fetch_reference_text(
    coll,
    subject_id: int,
    hadm_id: int,
    target_sections: List[str],
    max_chars: int = 6000,
) -> str:
    """
    Build a gold reference by concatenating text of all chunks in the target sections
    for this admission. This is a defensible extractive reference for your tasks.
    """
    docs = list(coll.find(
        {"subject_id": subject_id, "hadm_id": hadm_id, "section": {"$in": target_sections}},
        {"_id": 0, "section": 1, "text": 1, "full_text": 1, "chunk_index": 1}
    ))

    docs.sort(key=lambda d: d.get("chunk_index", 10**9))

    parts = []
    for d in docs:
        txt = d.get("text") or d.get("full_text") or ""
        txt = txt.strip()
        if not txt:
            continue
        parts.append(txt)

        if sum(len(p) for p in parts) > max_chars:
            break

    ref = "\n".join(parts).strip()
    return ref

def call_ollama(question: str, context: str) -> str:
    """
    Generate answer conditioned on the reference context (extractive-style).
    This isolates generation quality rather than retrieval quality.
    """
    prompt = f"""You are a medical QA assistant.
Use ONLY the provided context. If information is missing, say so.

QUESTION:
{question}

CONTEXT:
{context}

Answer concisely and accurately:
"""
    t0 = time.perf_counter()
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 256},
        },
        timeout=OLLAMA_TIMEOUT_SEC,
    )
    resp.raise_for_status()
    out = resp.json().get("response", "").strip()
    t1 = time.perf_counter()
    return out, (t1 - t0)

def rouge1_f1(pred: str, ref: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    s = scorer.score(ref, pred)
    return float(s["rouge1"].fmeasure)

def bertscore_f1(pred: str, ref: str, model_type: str = "microsoft/deberta-xlarge-mnli") -> float:
    """
    BERTScore default model is often 'roberta-large' internally.
    You can set model_type for stability.
    """
    P, R, F1 = bertscore([pred], [ref], lang="en", model_type=model_type, verbose=False)
    return float(F1[0].item())

def try_bleurt(pred: str, ref: str, bleurt_checkpoint: str) -> Optional[float]:
    """
    Optional BLEURT scoring (requires: pip install bleurt and checkpoint download).
    If not available, return None.
    """
    try:
        from bleurt import score as bleurt_score
    except Exception:
        return None

    scorer = bleurt_score.BleurtScorer(bleurt_checkpoint)
    scores = scorer.score(references=[ref], candidates=[pred])
    return float(scores[0])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="eval_tasks_50.csv")
    ap.add_argument("--out", required=True, help="output CSV")
    ap.add_argument("--with_llm", action="store_true", help="Call Ollama to generate answers")
    ap.add_argument("--bertscore_model", default="microsoft/deberta-xlarge-mnli")
    ap.add_argument("--bleurt_checkpoint", default="", help="Path to BLEURT checkpoint (optional)")
    ap.add_argument("--max_ref_chars", type=int, default=6000)
    args = ap.parse_args()

    mongo = MongoClient(MONGO_URI)
    coll = mongo[DB_NAME][CHUNKS_COLLECTION]

    rows = []
    with open(args.tasks, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    out_rows = []

    for r in rows:
        task_name = r.get("task_name", "")
        if task_name not in EXTRACTIVE_TASKS_ALLOWLIST:
            continue

        subject_id = int(r["subject_id"])
        hadm_id = int(r["hadm_id"])
        question = r["question"]
        target_sections = (r.get("target_sections") or "").split("|")
        target_sections = [s for s in target_sections if s]

        ref = fetch_reference_text(
            coll, subject_id, hadm_id, target_sections,
            max_chars=args.max_ref_chars
        )

        if not ref.strip():
            out_rows.append({
                "task_name": task_name,
                "subject_id": subject_id,
                "hadm_id": hadm_id,
                "targets": "|".join(target_sections),
                "ref_len": 0,
                "pred_len": 0,
                "rouge1_f1": "",
                "bertscore_f1": "",
                "bleurt": "",
                "aggregate": "",
                "llm_time_sec": "",
                "note": "no_reference_text",
            })
            continue

        if args.with_llm:
            pred, llm_time = call_ollama(question, ref)
        else:
            pred, llm_time = "", 0.0

        if not pred.strip() and args.with_llm:
            note = "empty_prediction"
        else:
            note = ""

        if pred.strip():
            r1 = rouge1_f1(pred, ref)
            bs = bertscore_f1(pred, ref, model_type=args.bertscore_model)

            bl = None
            if args.bleurt_checkpoint:
                bl = try_bleurt(pred, ref, args.bleurt_checkpoint)

            metrics = [r1, bs] + ([bl] if bl is not None else [])
            agg = float(sum(metrics) / len(metrics)) if metrics else ""

        else:
            r1, bs, bl, agg = "", "", "", ""

        out_rows.append({
            "task_name": task_name,
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "targets": "|".join(target_sections),
            "ref_len": len(ref),
            "pred_len": len(pred),
            "rouge1_f1": round(r1, 4) if isinstance(r1, float) else r1,
            "bertscore_f1": round(bs, 4) if isinstance(bs, float) else bs,
            "bleurt": round(bl, 4) if isinstance(bl, float) else ("" if bl is None else bl),
            "aggregate": round(agg, 4) if isinstance(agg, float) else agg,
            "llm_time_sec": round(llm_time, 4) if args.with_llm else "",
            "note": note,
        })

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else [])
        if out_rows:
            w.writeheader()
            w.writerows(out_rows)

    print(f"Wrote {args.out} rows={len(out_rows)}")
    mongo.close()

if __name__ == "__main__":
    main()
