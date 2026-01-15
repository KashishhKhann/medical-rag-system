from pymongo import MongoClient
from bson import json_util

client = MongoClient("mongodb://localhost:27017")
db = client["MIMIC"]

docs = list(db["processed_chunks"].aggregate([
    {"$sample": {"size": 50}}
]))

with open("processed_chunks_random_50.json", "w", encoding="utf-8") as f:
    f.write(json_util.dumps(docs, indent=2))

print(f"Wrote {len(docs)} docs to processed_chunks_random_50.json")
