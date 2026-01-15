# Medical RAG System

End-to-end medical RAG pipeline that ingests clinical notes, chunks them, builds embeddings and a FAISS index, constructs a Neo4j knowledge graph, and supports retrieval + generation with evaluation scripts.

## What this repo does
- Import filtered notes into MongoDB
- Chunk notes into retrievable passages
- Create embeddings and build a FAISS index
- Build a Neo4j knowledge graph from chunks
- Run retrieval + generation and evaluate results

## Requirements
- Python 3.10
- MongoDB running locally (`mongodb://localhost:27017/`)
- Neo4j running locally (`bolt://localhost:7687`)
- Ollama running locally (`http://localhost:11434`)

## Quick start
```bash
python3.10 -m venv venv310
source venv310/bin/activate
pip install --upgrade pip
pip install -r requirements310.txt
python -m spacy download en_core_sci_sm
```

## Configuration
The pipeline reads defaults from `config.py` and optional overrides from `.env`.

Key env vars:
- `MONGODB_URI`, `MONGODB_DB`
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
- `FAISS_INDEX_PATH`, `FAISS_MAP_PATH`
- `OLLAMA_BASE_URL`, `OLLAMA_MODEL`
- `BIOCLINICALBERT_MODEL` (embedding model override)

By default, the embedding model is:
```
sentence-transformers/all-mpnet-base-v2
```
To use BioClinicalBERT:
```
BIOCLINICALBERT_MODEL=emilyalsentzer/Bio_ClinicalBERT
```

## Pipeline (end-to-end)
```bash
# A) Import filtered notes
python import_mimic_notes.py --file MIMIC.filtered_notes.json --drop

# B) Chunk notes
python process_batch.py

# C) Add embeddings
python add_embeddings.py

# D) Build FAISS index
python build_faiss_index.py

# E) Build Neo4j KG
python kg_extraction.py

# F) Interactive query
python kg_rag_query.py
```

## Evaluation
```bash
python run_generation_capture_from_tasks.py --tasks eval_tasks_50.csv --out answers_sim_plus_kg.csv --variant sim_plus_kg --top_k 5
python run_generation_eval_extractive.py --tasks eval_tasks_50.csv --out generation_eval_extractive.csv --with_llm
python run_evidence_retrieval_eval.py --tasks eval_tasks_50.csv --out evidence_retrieval_eval.csv --top_k 5
```

## Project files
- Core scripts: `import_mimic_notes.py`, `process_batch.py`, `add_embeddings.py`, `build_faiss_index.py`, `kg_extraction.py`, `kg_rag_query.py`
- Evaluation: `run_generation_capture_from_tasks.py`, `run_generation_eval_extractive.py`, `run_evidence_retrieval_eval.py`
- Config: `config.py`, optional `.env`, `requirements310.txt`

## Notes
- Large input/output files are stored in the repo root (e.g., `MIMIC.filtered_notes.json`, `answers_sim_plus_kg.csv`).
- See `RUNBOOK.md` for a detailed, step-by-step reproduction guide and troubleshooting tips.
# medical-rag-system
