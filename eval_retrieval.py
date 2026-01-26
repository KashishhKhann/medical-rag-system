import json
import json
import csv
import re
import math
import sys

EVAL_CSV = "eval_tasks_50.csv"
CHUNKS_JSON = "MIMIC.processed_chunks.json"


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    return re.sub(r"\\s+", " ", text).strip()


def tokenize_for_bm25(text: str):
    norm = normalize_text(text)
    return [t for t in norm.split() if len(t) >= 2]


def bm25_scores(query_tokens, docs_tokens, k1=1.2, b=0.75):
    if not query_tokens or not docs_tokens:
        return [0.0 for _ in docs_tokens]

    df = {}
    for tokens in docs_tokens:
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1

    N = len(docs_tokens)
    avgdl = sum(len(t) for t in docs_tokens) / max(1, N)

    scores = []
    for tokens in docs_tokens:
        freqs = {}
        for t in tokens:
            freqs[t] = freqs.get(t, 0) + 1
        dl = len(tokens)
        score = 0.0
        for t in query_tokens:
            f = freqs.get(t, 0)
            if not f:
                continue
            idf = math.log(1.0 + (N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))
            denom = f + k1 * (1.0 - b + b * (dl / max(1.0, avgdl)))
            score += idf * (f * (k1 + 1.0)) / denom
        scores.append(score)
    return scores


def read_eval_tasks(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for r in reader:
            rows.append(r)
    return rows


def main(task_idx=0):
    rows = read_eval_tasks(EVAL_CSV)
    if task_idx < 0 or task_idx >= len(rows):
        print(f"task_idx out of range: {task_idx}")
        sys.exit(1)

    task = rows[task_idx]
    subject_id = int(task["subject_id"])
    hadm_id = int(task["hadm_id"])
    question = str(task["question"])[:400]
    gold_section = str(task.get("gold_section", ""))
    top_k = int(task.get("top_k", 5))

    with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    docs = [c for c in chunks if int(c.get("subject_id", -1)) == subject_id and int(c.get("hadm_id", -1)) == hadm_id]
    if not docs:
        print("No chunks found for this patient/admission.")
        sys.exit(1)

    docs_tokens = [tokenize_for_bm25(c.get("text", "")) for c in docs]
    q_tokens = tokenize_for_bm25(question)
    scores = bm25_scores(q_tokens, docs_tokens)

    ranked = sorted(list(zip(docs, scores)), key=lambda x: x[1], reverse=True)
    top = ranked[:top_k]

    print(f"Task #{task_idx} | case={task['case_id']} | question={question}")
    print(f"Gold section: {gold_section}")
    print("Top retrieved chunks:")
    for i, (doc, sc) in enumerate(top, start=1):
        print(f"[{i}] chunk_id={doc.get('chunk_id')} | section={doc.get('section')} | score={sc:.4f}")

    top_sections = [doc.get('section') for doc, _ in top]
    top1_match = (top_sections[0] == gold_section) if top_sections else False
    in_topk = gold_section in top_sections

    print(f"\nTop1 matches gold_section: {top1_match}")
    print(f"Gold section in top{top_k}: {in_topk}")


if __name__ == '__main__':
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(idx)
if __name__ == '__main__':
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(idx)
