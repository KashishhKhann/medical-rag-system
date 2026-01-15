#!/usr/bin/env python3
"""
Reset script: Clear all MongoDB collections, FAISS indexes, and Neo4j graphs.
Useful for starting fresh or clearing bad data.

Usage:
    python reset_all.py
"""

import os
import sys
from pathlib import Path

from pymongo import MongoClient
from neo4j import GraphDatabase

from config import (
    MONGO_URI, DB_NAME, NOTES_COLLECTION, CHUNKS_COLLECTION,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    FAISS_INDEX_PATH, FAISS_MAP_PATH
)


def reset_mongodb():
    """Drop MongoDB collections."""
    print("\n--- MongoDB Reset ---")
    
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    
    try:
        for coll_name in [NOTES_COLLECTION, CHUNKS_COLLECTION]:
            if coll_name in db.list_collection_names():
                db[coll_name].drop()
                print(f"Dropped collection: {coll_name}")
            else:
                print(f"Collection not found: {coll_name}")
        
        print("MongoDB reset complete")
    finally:
        client.close()


def reset_faiss():
    """Delete FAISS index and mapping files."""
    print("\n--- FAISS Reset ---")
    
    files_to_remove = [FAISS_INDEX_PATH, FAISS_MAP_PATH]
    
    for fpath in files_to_remove:
        p = Path(fpath)
        if p.exists():
            p.unlink()
            print(f"Deleted: {fpath}")
        else:
            print(f"File not found: {fpath}")
    
    print("FAISS reset complete")


def reset_neo4j():
    """Clear all Neo4j nodes and relationships."""
    print("\n--- Neo4j Reset ---")
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Cleared all Neo4j nodes and relationships")
    except Exception as e:
        print(f"Neo4j error: {e}")
    finally:
        driver.close()
    
    print("Neo4j reset complete")


def main():
    print("\n--- MEDICAL RAG SYSTEM RESET ---\n")
    print("\nThis will DELETE:")
    print(f"  - MongoDB collections: {NOTES_COLLECTION}, {CHUNKS_COLLECTION}")
    print(f"  - FAISS index: {FAISS_INDEX_PATH}")
    print(f"  - FAISS mapping: {FAISS_MAP_PATH}")
    print(f"  - Neo4j graph: ALL nodes and relationships")
    print("\nThis action CANNOT be undone!")
    
    response = input("\nAre you sure? Type 'YES' to confirm: ").strip()
    
    if response != "YES":
        print("Cancelled. No changes made.")
        sys.exit(0)
    
    print("\nStarting reset...\n")
    
    try:
        reset_mongodb()
        reset_faiss()
        reset_neo4j()
        
        print("\n--- FULL RESET COMPLETE ---")
        print("\nSystem is now ready for fresh data import!")
        print("\nNext steps:")
        print("1. python import_mimic_notes.py --file MIMIC.filtered_notes.json")
        print("2. python process_batch.py")
        print("3. python add_embeddings.py")
        print("4. python build_faiss_index.py")
        print("5. python kg_extraction.py")
        print("6. python kg_rag_query.py")
        print("-"*60 + "\n")
        
    except Exception as e:
        print(f"\nReset failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
