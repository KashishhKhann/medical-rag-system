# Medical RAG Pipeline Runbook

## Scope
Step-by-step instructions to reproduce the Medical RAG pipeline starting from an existing filtered notes JSON (e.g., `MIMIC.filtered_notes.json`) through import → chunking → embeddings → FAISS → Neo4j KG → answer generation → evaluation, yielding `answers_sim_plus_kg.csv`, `generation_eval_extractive.csv`, and optionally `evidence_retrieval_eval.csv`.

## Prerequisites
- OS: macOS or Linux.
- Installed tooling: Python 3.10, pip, curl, MongoDB client (`mongosh`), Neo4j running locally, Ollama installed.
- Running services (localhost):
  - MongoDB: `localhost:27017`
  - Neo4j: `localhost:7687`
  - Ollama: `localhost:11434`
- Quick checks:
  - `mongosh --eval "db.runCommand({ ping: 1 })"`
  - `curl http://localhost:11434/api/tags`
  - Confirm Neo4j is up (Browser or `cypher-shell`).

## Environment setup (venv310)
```bash
python3.10 -m venv venv310
source venv310/bin/activate
pip install --upgrade pip
pip install -r requirements310.txt
python -m spacy download en_core_sci_sm   # required for kg_extraction.py
# (Optional) python -m spacy download en_core_web_sm  # SciSpaCy if available
```

## Configuration
- `config.py` (and optional `.env`) control URIs and model names. Expected vars if overriding defaults:
  - `MONGODB_URI` (default `mongodb://localhost:27017/`)
  - `MONGODB_DB` (default `MIMIC`)
  - `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
  - `FAISS_INDEX_PATH`, `FAISS_MAP_PATH`
  - `OLLAMA_BASE_URL` (default `http://localhost:11434`), `OLLAMA_MODEL`

## Input file
- Ensure filtered notes JSON is present (e.g., `MIMIC.filtered_notes.json`).
- Formats accepted: JSON array or NDJSON (one JSON object per line).

## Pipeline steps

### A) Import filtered notes into MongoDB
```bash
python import_mimic_notes.py --file MIMIC.filtered_notes.json --drop
```
Expected: DB `MIMIC`, collection `filtered_notes` populated.

### B) Chunk notes
```bash
python process_batch.py
```
Expected: Collection `processed_chunks` with `chunk_id`, `subject_id`, `hadm_id`, `section`, `text/full_text`, `metadata`. Initially `embedding` empty.

### C) Add embeddings
```bash
python add_embeddings.py
```
Expected: `embedding` field (768-dim) populated for all chunks.

### D) Build FAISS index
```bash
python build_faiss_index.py
```
Expected files: `data/faiss_biobert.index`, `data/faiss_biobert_mapping.json`.

### E) Build Neo4j KG
```bash
python kg_extraction.py
```
Expected: Neo4j nodes `Chunk` and `Concept`, relation `MENTIONS_CONCEPT`; Mongo `kg_status` updated.

### F) (Optional) Interactive demo
```bash
python kg_rag_query.py
```
Prompts for question, optional `subject_id`, `hadm_id`.

## Evaluation / reproduction (non-interactive)

### Generate answers (sim_plus_kg variant)
```bash
python run_generation_capture_from_tasks.py --tasks eval_tasks_50.csv --out answers_sim_plus_kg.csv --variant sim_plus_kg --top_k 5
```

### Extractive evaluation (LLM conditioned on gold sections)
```bash
python run_generation_eval_extractive.py --tasks eval_tasks_50.csv --out generation_eval_extractive.csv --with_llm
```

### Optional: Evidence retrieval metrics
```bash
python run_evidence_retrieval_eval.py --tasks eval_tasks_50.csv --out evidence_retrieval_eval.csv --top_k 5
```

## Verification
- Mongo counts:
  ```bash
  python - <<'PY'
from pymongo import MongoClient
client = MongoClient("mongodb://localhost:27017/")
db = client["MIMIC"]
print("filtered_notes:", db.filtered_notes.count_documents({}))
print("processed_chunks:", db.processed_chunks.count_documents({}))
client.close()
PY
  ```
- FAISS files:
  ```bash
  ls -lh data/faiss_biobert.index data/faiss_biobert_mapping.json
  ```
- Neo4j nodes:
  ```bash
  python - <<'PY'
from neo4j import GraphDatabase
driver = GraphDatabase.driver("neo4j://127.0.0.1:7687", auth=("neo4j", "password"))
with driver.session() as s:
    print("Total Neo4j nodes:", s.run("MATCH (n) RETURN count(n) as c").single()[0])
driver.close()
PY
  ```

## Troubleshooting
- spaCy model missing: `python -m spacy download en_core_web_sm`.
- Ollama not running / wrong model tag: ensure `curl http://localhost:11434/api/tags` shows your model; adjust `OLLAMA_MODEL` in `.env`/`config.py`.
- Mongo connection issues: verify `mongosh` ping; check `MONGODB_URI`.
- `add_embeddings.py` reports “No work to do”: embeddings already present (normal).
- Slow runs: LLM generation dominates latency; retrieval/FAISS are fast.

## Reset / Clean start (optional)
```bash
echo "YES" | python reset_all.py
```
Clears Mongo collections (`filtered_notes`, `processed_chunks`), FAISS files, and Neo4j graph.

## Minimal files to keep for reproducibility
- Inputs: `MIMIC.filtered_notes.json`, `eval_tasks_50.csv`
- Core scripts: `import_mimic_notes.py`, `process_batch.py`, `add_embeddings.py`, `build_faiss_index.py`, `kg_extraction.py`, `kg_rag_query.py`
- Eval scripts: `run_generation_capture_from_tasks.py`, `run_generation_eval_extractive.py`, `run_evidence_retrieval_eval.py`
- Config/deps: `config.py`, optional `.env`, `requirements310.txt`
- Outputs: `answers_sim_plus_kg.csv`, `generation_eval_extractive.csv`, (optional) `evidence_retrieval_eval.csv`
