"""
Centralized configuration for the Medical RAG pipeline.

All scripts should import from this module to ensure consistency.
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("MONGODB_DB", "MIMIC")

NOTES_COLLECTION = os.getenv("MONGODB_NOTES_COLLECTION", "filtered_notes")
CHUNKS_COLLECTION = os.getenv("MONGODB_CHUNKS_COLLECTION", "processed_chunks")
LOG_COLLECTION = "processing_log"

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss_biobert.index")
FAISS_MAP_PATH = os.getenv("FAISS_MAP_PATH", "data/faiss_biobert_mapping.json")

BERT_MODEL = os.getenv("BIOCLINICALBERT_MODEL", "emilyalsentzer/Bio_ClinicalBERT")

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2:latest")

USE_HF_TOKENIZER = True
USE_SPACY_SENT_SPLIT = True
SENTENCE_SPLIT_MODEL = "en_core_sci_sm"

CHUNK_NARR_TARGET_TOKENS = 240
CHUNK_NARR_MAX_TOKENS = 320
CHUNK_NARR_MIN_TOKENS = 140
CHUNK_NARR_OVERLAP_TOKENS = 40

CHUNK_LIST_TARGET_TOKENS = 420
CHUNK_LIST_MAX_TOKENS = 520
CHUNK_LIST_MIN_TOKENS = 240
CHUNK_LIST_OVERLAP_TOKENS = 40

CHUNK_HEADER_TARGET_TOKENS = 160
CHUNK_HEADER_MAX_TOKENS = 220
CHUNK_HEADER_MIN_TOKENS = 80
CHUNK_HEADER_OVERLAP_TOKENS = 0

CHUNK_OVERLAP_SENTENCES = 0

USE_CHILD_SPANS = True
CHILD_SPAN_TARGET_TOKENS = 120
CHILD_SPAN_MAX_TOKENS = 160
CHILD_SPAN_MIN_TOKENS = 60
CHILD_SPAN_OVERLAP_TOKENS = 40
CHILD_SPAN_TOP_K = 6
CHILD_SPAN_MAX_PER_PARENT = 12

FIELD_TEXT = "text"
FIELD_FULL_TEXT = "full_text"
FIELD_CHUNK_ID = "chunk_id"
FIELD_NOTE_ID = "note_id"
FIELD_SUBJECT_ID = "subject_id"
FIELD_HADM_ID = "hadm_id"
FIELD_SECTION = "section"
FIELD_EMBEDDING = "embedding"
FIELD_KG_STATUS = "kg_status"
FIELD_METADATA = "metadata"

DATA_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "data_config.json")

def load_data_config():
    """Load data_config.json if it exists, for backward compatibility."""
    if os.path.exists(DATA_CONFIG_PATH):
        with open(DATA_CONFIG_PATH, "r") as f:
            return json.load(f)
    return {}

_data_config = load_data_config()
if _data_config.get("database"):
    DB_NAME = _data_config["database"]
if _data_config.get("collection"):
    NOTES_COLLECTION = _data_config["collection"]
if _data_config.get("text_field"):
    FIELD_TEXT = _data_config["text_field"]

def validate_neo4j_config():
    """Check that Neo4j configuration is present."""
    if not NEO4J_URI:
        raise ValueError("NEO4J_URI environment variable is not set")
    if not NEO4J_USER:
        raise ValueError("NEO4J_USER environment variable is not set")
    if not NEO4J_PASSWORD:
        raise ValueError("NEO4J_PASSWORD environment variable is not set")

def validate_mongo_config():
    """Check that MongoDB configuration is present."""
    if not MONGO_URI:
        raise ValueError("MONGO_URI environment variable is not set")
    if not DB_NAME:
        raise ValueError("DB_NAME is not set")
