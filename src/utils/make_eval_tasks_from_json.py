"""
Build an evaluation task sheet from a JSON export of processed_chunks.

Input:  eval_json (a JSON array of chunk documents; e.g., MIMIC.processed_chunks.json)
Output: eval_tasks_50.csv   (one row per (case,task))
        eval_cases_50.json  (unique admissions to auto-run)

This script does NOT need Mongo/Neo4j/FAISS. It only uses the JSON file.
"""

import json
import pandas as pd

SECTION_QUESTIONS = {
    "chief_complaint": ("chief_complaint_reason", "What was the chief complaint / reason for admission?", ["chief_complaint","hpi"]),
    "hpi": ("hpi_summary", "Summarize the history of present illness.", ["hpi"]),
    "pmh": ("pmh_list", "What past medical history is documented?", ["pmh"]),
    "physical_exam": ("physical_exam_results", "Summarize the key physical exam findings and pertinent results (including notable abnormal labs if present).", ["physical_exam"]),
    "hospital_course": ("hospital_course_summary", "Summarize the hospital course and management during the admission.", ["hospital_course"]),
    "meds_discharge": ("meds_discharge_list", "List the discharge medications and any discharge diagnoses/disposition stated.", ["meds_discharge"]),
    "header": ("header_service_allergies", "What service was the patient admitted under and what allergies are recorded?", ["header"]),
}

def oid_to_str(x):
    if isinstance(x, dict) and "$oid" in x:
        return x["$oid"]
    return str(x)

def date_to_str(x):
    if isinstance(x, dict) and "$date" in x:
        return x["$date"]
    return str(x)

def main(eval_json="MIMIC.processed_chunks.json",
         out_csv="eval_tasks_50.csv",
         out_cases="eval_cases_50.json",
         top_k=5):

    with open(eval_json, "r", encoding="utf-8") as f:
        docs = json.load(f)

    tasks = []
    for d in docs:
        sec = d.get("section", "unknown")
        task_name, question, targets = SECTION_QUESTIONS.get(
            sec, ("unknown", f"Summarize the {sec} section.", [sec])
        )

        case_id = f"{int(d['subject_id'])}_{int(d['hadm_id'])}"

        text = (d.get("text") or "").replace("\n", " ").strip()
        gold_snippet = text[:350] + ("..." if len(text) > 350 else "")

        tasks.append({
            "case_id": case_id,
            "subject_id": int(d["subject_id"]),
            "hadm_id": int(d["hadm_id"]),
            "note_id": d.get("note_id", ""),
            "gold_chunk_id": d.get("chunk_id", ""),
            "gold_section": sec,
            "task_name": task_name,
            "question": question,
            "target_sections": "|".join(targets),
            "top_k": int(top_k),
            "gold_snippet": gold_snippet,
            "json_oid": oid_to_str(d.get("_id", "")),
            "created_at": date_to_str(d.get("created_at", "")),
        })

    for d in docs:
        if d.get("section") != "meds_discharge":
            continue
        text = d.get("text", "") or ""
        case_id = f"{int(d['subject_id'])}_{int(d['hadm_id'])}"
        if "Discharge Diagnosis:" in text:
            tasks.append({
                "case_id": case_id,
                "subject_id": int(d["subject_id"]),
                "hadm_id": int(d["hadm_id"]),
                "note_id": d.get("note_id", ""),
                "gold_chunk_id": d.get("chunk_id", ""),
                "gold_section": "meds_discharge",
                "task_name": "discharge_diagnosis",
                "question": "What is the discharge diagnosis stated in the discharge section?",
                "target_sections": "meds_discharge",
                "top_k": int(top_k),
                "gold_snippet": text.replace("\n"," ").strip()[:350] + ("..." if len(text) > 350 else ""),
                "json_oid": oid_to_str(d.get("_id", "")),
                "created_at": date_to_str(d.get("created_at", "")),
            })
        if "Discharge Disposition:" in text:
            tasks.append({
                "case_id": case_id,
                "subject_id": int(d["subject_id"]),
                "hadm_id": int(d["hadm_id"]),
                "note_id": d.get("note_id", ""),
                "gold_chunk_id": d.get("chunk_id", ""),
                "gold_section": "meds_discharge",
                "task_name": "discharge_disposition",
                "question": "What is the discharge disposition stated in the discharge section?",
                "target_sections": "meds_discharge",
                "top_k": int(top_k),
                "gold_snippet": text.replace("\n"," ").strip()[:350] + ("..." if len(text) > 350 else ""),
                "json_oid": oid_to_str(d.get("_id", "")),
                "created_at": date_to_str(d.get("created_at", "")),
            })

    df = pd.DataFrame(tasks)
    df.to_csv(out_csv, index=False)

    seen = set()
    cases = []
    for t in tasks:
        key = (t["subject_id"], t["hadm_id"])
        if key in seen:
            continue
        seen.add(key)
        cases.append({"case_id": t["case_id"], "subject_id": t["subject_id"], "hadm_id": t["hadm_id"]})

    with open(out_cases, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)

    print(f"Wrote: {out_csv}  rows={len(df)}")
    print(f"Wrote: {out_cases} admissions={len(cases)}")

if __name__ == "__main__":
    main()
