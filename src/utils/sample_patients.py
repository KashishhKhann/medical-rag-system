from pymongo import MongoClient
import random

MONGO = "mongodb://localhost:27017/"
DB = "MIMIC"
RAW_COLLECTION = "notes"
NEW_COLLECTION = "filtered_notes"

client = MongoClient(MONGO)
db = client[DB]
notes_col = db[RAW_COLLECTION]
filtered_col = db[NEW_COLLECTION]

all_patients = notes_col.distinct("subject_id")
random.shuffle(all_patients)

PATIENT_COUNT = 1000
selected_patients = all_patients[:PATIENT_COUNT]

print(f"Selected {len(selected_patients)} random patients")
print("\nExporting records for selected patients...\n")

batch_size = 2000
count = 0

for pid in selected_patients:
    docs = list(notes_col.find({"subject_id": pid}))
    if docs:
        filtered_col.insert_many(docs)
        count += len(docs)

print("\n" + "---" * 20)
print(f"Export COMPLETE")
print(f"Patients exported : {len(selected_patients)}")
print(f"Notes copied      : {count}")
print(f"New collection    : {NEW_COLLECTION}")
print("---" * 20)

client.close()
