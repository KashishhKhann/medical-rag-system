#!/usr/bin/env python3
"""
Import MIMIC.filtered_notes.json into MongoDB.

This script loads the JSON file containing filtered clinical notes and imports
them into the MongoDB collection that serves as the source for the Medical RAG pipeline.

Usage:
    python import_mimic_notes.py [--file PATH] [--drop]

Options:
    --file PATH    Path to the JSON file (default: ./MIMIC.filtered_notes.json)
    --drop         Drop the existing collection before importing
"""

import argparse
import json
import sys
from pathlib import Path

from pymongo import MongoClient
from pymongo.errors import BulkWriteError
from tqdm import tqdm

from config import (
    MONGO_URI, DB_NAME, NOTES_COLLECTION,
    FIELD_SUBJECT_ID, FIELD_HADM_ID, FIELD_TEXT
)


def detect_json_format(file_path: Path) -> str:
    """
    Detect whether the JSON file is:
    - 'array': A JSON array of objects [...]
    - 'ndjson': Newline-delimited JSON (one object per line)

    Returns:
        'array' or 'ndjson'
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        first_char = f.read(1).strip()
        if first_char == '[':
            return 'array'
        elif first_char == '{':
            return 'ndjson'
        else:
            raise ValueError(f"Unknown JSON format. File starts with: {first_char!r}")


def load_json_array(file_path: Path) -> list:
    """Load a JSON array file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_ndjson(file_path: Path) -> list:
    """Load a newline-delimited JSON file."""
    docs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
    return docs


def validate_document(doc: dict) -> dict:
    """
    Validate and normalize a document before insertion.

    Returns the normalized document, or None if invalid.
    """
    text = doc.get(FIELD_TEXT) or doc.get('TEXT') or doc.get('note_text')
    if not text or not str(text).strip():
        return None

    normalized = {
        FIELD_TEXT: str(text).strip(),
    }

    subject_id = (
        doc.get(FIELD_SUBJECT_ID) or
        doc.get('SUBJECT_ID') or
        doc.get('patient_id')
    )
    if subject_id is not None:
        try:
            normalized[FIELD_SUBJECT_ID] = int(subject_id)
        except (ValueError, TypeError):
            normalized[FIELD_SUBJECT_ID] = subject_id

    hadm_id = (
        doc.get(FIELD_HADM_ID) or
        doc.get('HADM_ID') or
        doc.get('admission_id')
    )
    if hadm_id is not None:
        try:
            normalized[FIELD_HADM_ID] = int(hadm_id)
        except (ValueError, TypeError):
            normalized[FIELD_HADM_ID] = hadm_id

    optional_fields = [
        'charttime', 'CHARTTIME', 'chartdate', 'CHARTDATE',
        'category', 'CATEGORY', 'description', 'DESCRIPTION',
        'row_id', 'ROW_ID', 'note_id'
    ]
    for field in optional_fields:
        if field in doc and doc[field] is not None:
            normalized[field.lower()] = doc[field]

    return normalized


def create_indexes(collection):
    """Create indexes on the collection."""
    print("Creating indexes...")

    try:
        collection.create_index(FIELD_SUBJECT_ID)
        print(f"  - Created index on {FIELD_SUBJECT_ID}")

        collection.create_index(FIELD_HADM_ID)
        print(f"  - Created index on {FIELD_HADM_ID}")

        collection.create_index([(FIELD_SUBJECT_ID, 1), (FIELD_HADM_ID, 1)])
        print(f"  - Created compound index on ({FIELD_SUBJECT_ID}, {FIELD_HADM_ID})")
    except Exception as e:
        print(f"  - Warning: Index creation failed: {e}")


def import_notes(file_path: Path, drop_existing: bool = False):
    """
    Import notes from JSON file into MongoDB.

    Args:
        file_path: Path to the JSON file
        drop_existing: Whether to drop the existing collection first
    """
    print("\n------------------------")
    print("MIMIC Notes Import Script")
    print("------------------------")
    print(f"File:       {file_path}")
    print(f"MongoDB:    {MONGO_URI}")
    print(f"Database:   {DB_NAME}")
    print(f"Collection: {NOTES_COLLECTION}")
    print("------------------------\n")

    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        sys.exit(1)

    client = MongoClient(MONGO_URI)
    try:
        db = client[DB_NAME]
        collection = db[NOTES_COLLECTION]

        if drop_existing:
            print("Dropping existing collection...")
            collection.drop()

        existing_count = collection.count_documents({})
        if existing_count > 0:
            print(f"Warning: Collection already has {existing_count} documents.")
            response = input("Continue and add more? (y/N): ").strip().lower()
            if response != 'y':
                print("Aborted.")
                return

        print(f"Detecting file format...")
        file_format = detect_json_format(file_path)
        print(f"Format: {file_format}")

        print("Loading documents from file...")
        if file_format == 'array':
            raw_docs = load_json_array(file_path)
        else:
            raw_docs = load_ndjson(file_path)

        print(f"Loaded {len(raw_docs)} documents from file.")

        print("Validating and normalizing documents...")
        valid_docs = []
        invalid_count = 0

        for doc in tqdm(raw_docs, desc="Validating"):
            normalized = validate_document(doc)
            if normalized:
                valid_docs.append(normalized)
            else:
                invalid_count += 1

        print(f"Valid documents: {len(valid_docs)}")
        if invalid_count > 0:
            print(f"Skipped invalid documents: {invalid_count}")

        if not valid_docs:
            print("ERROR: No valid documents to import.")
            return

        print("\nInserting documents into MongoDB...")
        batch_size = 1000
        inserted_count = 0

        for i in tqdm(range(0, len(valid_docs), batch_size), desc="Inserting batches"):
            batch = valid_docs[i:i + batch_size]
            try:
                result = collection.insert_many(batch, ordered=False)
                inserted_count += len(result.inserted_ids)
            except BulkWriteError as e:
                inserted_count += e.details.get('nInserted', 0)
                print(f"Warning: Batch had {len(e.details.get('writeErrors', []))} errors")

        create_indexes(collection)

        final_count = collection.count_documents({})
        unique_patients = len(collection.distinct(FIELD_SUBJECT_ID))
        unique_admissions = len(collection.distinct(FIELD_HADM_ID))

        print("\n------------------------")
        print("IMPORT COMPLETE")
        print("------------------------")
        print(f"Documents inserted: {inserted_count}")
        print(f"Total in collection: {final_count}")
        print(f"Unique patients: {unique_patients}")
        print(f"Unique admissions: {unique_admissions}")
        print("------------------------")
        print("\nNext steps:")
        print("1. python process_batch.py")
        print("2. python add_embeddings.py")
        print("3. python build_faiss_index.py")
        print("4. python kg_extraction.py")
        print("5. python kg_rag_query.py")
        print("------------------------\n")
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(
        description="Import MIMIC filtered notes into MongoDB"
    )
    parser.add_argument(
        '--file',
        type=str,
        default='./MIMIC.filtered_notes.json',
        help='Path to the JSON file (default: ./MIMIC.filtered_notes.json)'
    )
    parser.add_argument(
        '--drop',
        action='store_true',
        help='Drop the existing collection before importing'
    )

    args = parser.parse_args()
    file_path = Path(args.file)

    import_notes(file_path, drop_existing=args.drop)


if __name__ == "__main__":
    main()
