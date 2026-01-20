"""
KG Extraction from processed_chunks -> Neo4j

For each chunk in MIMIC.processed_chunks:
- run spaCy NER
- create Concept nodes in Neo4j
- connect Chunk -> Concept with MENTIONS_CONCEPT

This is a lightweight, robust KG just for signal & reasoning.
"""

import os
import argparse
from typing import List

from pymongo import MongoClient
from neo4j import GraphDatabase
from tqdm import tqdm

import spacy

from config import (
    MONGO_URI, DB_NAME, CHUNKS_COLLECTION,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    FIELD_CHUNK_ID, FIELD_TEXT, FIELD_KG_STATUS,
    validate_neo4j_config
)

def load_nlp():
    try:
        return spacy.load("en_core_sci_sm")
    except Exception:
        try:
            return spacy.load("en_core_web_sm")
        except Exception as e:
            msg = (
                "Could not load a spaCy model. Install SciSpacy model:\n"
                "  pip install -r requirements310.txt\n"
                "or:\n"
                "  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/"
                "v0.5.1/en_core_sci_sm-0.5.1.tar.gz\n"
            )
            raise RuntimeError(msg) from e

nlp = load_nlp()

validate_neo4j_config()

mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
chunks_col = db[CHUNKS_COLLECTION]

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def get_chunks_to_process(limit: int | None = None):
    query = {FIELD_KG_STATUS: {"$ne": "done"}}
    cursor = chunks_col.find(query, {FIELD_CHUNK_ID: 1, FIELD_TEXT: 1})
    if limit:
        cursor = cursor.limit(limit)
    return list(cursor)

def process_chunk(tx, chunk_id: str, concepts: List[str]):
    tx.run(
        """
        MERGE (c:Chunk {chunk_id: $chunk_id})
        """,
        chunk_id=chunk_id,
    )

    for name in concepts:
        if not name:
            continue
        tx.run(
            """
            MATCH (c:Chunk {chunk_id: $chunk_id})
            MERGE (e:Concept {name: $name})
            MERGE (c)-[:MENTIONS_CONCEPT {source:'spacy'}]->(e)
            """,
            chunk_id=chunk_id,
            name=name,
        )

def extract_concepts(text: str) -> List[str]:
    doc = nlp(text)
    ents = [ent.text.strip() for ent in doc.ents if ent.text.strip()]
    seen = set()
    out = []
    for e in ents:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out

def main(limit: int | None = None):
    print("\n------------------------")
    print("Running KG Extraction over processed_chunks -> Neo4j")
    print("------------------------\n")

    chunks = get_chunks_to_process(limit)
    total = len(chunks)
    print(f"Chunks to process (kg_status != 'done'): {total}")

    if total == 0:
        print("Nothing to do. All chunks already processed.")
        return

    count_done = 0
    count_failed = 0

    for doc in tqdm(chunks, desc="KG extracting"):
        cid = doc["chunk_id"]
        text = doc.get("text") or ""
        if not text.strip():
            chunks_col.update_one(
                {"_id": doc["_id"]}, {"$set": {"kg_status": "skipped_empty"}}
            )
            continue
        try:
            concepts = extract_concepts(text)
            try:
                with driver.session() as session:
                    session.execute_write(process_chunk, cid, concepts)
            except Exception as e:
                msg = str(e)
                print(f"\nError writing to Neo4j for chunk {cid}: {msg}")
                # Detect common auth / rate-limit issues and abort so user can fix credentials/service
                if "AuthenticationRateLimit" in msg or "Unauthorized" in msg or "authentication" in msg.lower():
                    print("Detected Neo4j authentication/rate-limit error. Aborting run. Check credentials and restart Neo4j.")
                    chunks_col.update_one(
                        {"_id": doc["_id"]},
                        {"$set": {"kg_status": "error_auth", "kg_error": msg}},
                    )
                    count_failed += 1
                    break
                else:
                    chunks_col.update_one(
                        {"_id": doc["_id"]},
                        {"$set": {"kg_status": "error", "kg_error": msg}},
                    )
                    count_failed += 1
                    continue

            chunks_col.update_one(
                {"_id": doc["_id"]},
                {"$set": {"kg_status": "done", "kg_concepts_count": len(concepts)}},
            )
            count_done += 1

        except Exception as e:
            print(f"\nError on chunk {cid}: {e}")
            chunks_col.update_one(
                {"_id": doc["_id"]},
                {"$set": {"kg_status": "error", "kg_error": str(e)}},
            )
            count_failed += 1

    print("\n---------------- KG EXTRACTION COMPLETE ----------------")
    print(f"  Processed chunks : {count_done}")
    print(f"  Failed chunks    : {count_failed}")
    print("--------------------------------------------------------\n")

    mongo_client.close()
    driver.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="KG extraction from processed_chunks -> Neo4j")
    p.add_argument("--limit", type=int, default=None, help="Process at most N chunks (for smoke tests)")
    args = p.parse_args()
    main(limit=args.limit)
