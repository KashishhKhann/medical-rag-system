"""
Process a batch of notes -> sections -> smart chunks -> store in MongoDB + Neo4j.

Pipeline order after running this:
1. python add_embeddings.py
2. python build_faiss_index.py
3. python kg_extraction.py   (optional but recommended)
4. python kg_rag_query.py    (interactive querying)
"""

import json
import re
from datetime import datetime
from typing import Dict, List, Optional

from pymongo import MongoClient
from neo4j import GraphDatabase
from tqdm import tqdm
import spacy
from transformers import AutoTokenizer

from config import (
    MONGO_URI, DB_NAME, NOTES_COLLECTION, CHUNKS_COLLECTION,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    FIELD_TEXT, FIELD_FULL_TEXT, FIELD_CHUNK_ID,
    FIELD_NOTE_ID, FIELD_SUBJECT_ID, FIELD_HADM_ID,
    FIELD_SECTION, FIELD_METADATA,
    BERT_MODEL,
    USE_HF_TOKENIZER,
    USE_SPACY_SENT_SPLIT, SENTENCE_SPLIT_MODEL,
    CHUNK_NARR_TARGET_TOKENS, CHUNK_NARR_MAX_TOKENS, CHUNK_NARR_MIN_TOKENS,
    CHUNK_NARR_OVERLAP_TOKENS,
    CHUNK_LIST_TARGET_TOKENS, CHUNK_LIST_MAX_TOKENS, CHUNK_LIST_MIN_TOKENS,
    CHUNK_LIST_OVERLAP_TOKENS,
    CHUNK_HEADER_TARGET_TOKENS, CHUNK_HEADER_MAX_TOKENS, CHUNK_HEADER_MIN_TOKENS,
    CHUNK_HEADER_OVERLAP_TOKENS,
    CHUNK_OVERLAP_SENTENCES,
)

TEXT_FIELD = FIELD_TEXT


class BatchProcessor:
    SECTION_ALIASES = {
        "chief complaint": "chief_complaint",
        "cc": "chief_complaint",
        "reason for admission": "chief_complaint",
        "history of present illness": "hpi",
        "hpi": "hpi",
        "present illness": "hpi",
        "past medical history": "pmh",
        "pmh": "pmh",
        "physical exam": "physical_exam",
        "physical examination": "physical_exam",
        "hospital course": "hospital_course",
        "brief hospital course": "hospital_course",
        "discharge medications": "meds_discharge",
        "discharge meds": "meds_discharge",
        "medications on discharge": "meds_discharge",
        "meds on discharge": "meds_discharge",
        "discharge diagnosis": "meds_discharge",
        "discharge disposition": "meds_discharge",
    }

    def __init__(
        self,
        mongo_uri: str = MONGO_URI,
        db_name: str = DB_NAME,
        notes_collection: str = NOTES_COLLECTION,
        chunks_collection: str = CHUNKS_COLLECTION,
    ):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.raw_notes = self.db[notes_collection]
        self.chunks_col = self.db[chunks_collection]
        self.log_col = self.db["processing_log"]

        self.neo4j = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        print("\n--- Batch Pipeline Ready ---")
        print(f"MongoDB       - {mongo_uri}")
        print(f"Database      - {db_name}")
        print(f"Source Notes  - {notes_collection}")
        print(f"Output Chunks - {chunks_collection}")
        print(f"Log Collection - processing_log")
        print(f"Neo4j         - {NEO4J_URI}")
        print("------------------------\n")
        self.tokenizer = self._load_tokenizer()
        self.sent_splitter = self._load_sentence_splitter()

    def _normalize_heading(self, text: str) -> str:
        text = (text or "").lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _match_heading_label(self, label: str) -> Optional[str]:
        norm = self._normalize_heading(label)
        if not norm:
            return None
        for phrase, sec_key in self.SECTION_ALIASES.items():
            if norm == phrase or phrase in norm:
                return sec_key
        return None

    def _looks_like_heading(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped or len(stripped) > 80:
            return False
        if stripped.isupper():
            return True
        if stripped.endswith(":"):
            return True
        return bool(re.match(r"^[A-Za-z][A-Za-z /,-]{2,}$", stripped))

    def _load_tokenizer(self):
        if not USE_HF_TOKENIZER:
            return None
        try:
            return AutoTokenizer.from_pretrained(BERT_MODEL, use_fast=True)
        except Exception as e:
            print(f"Tokenizer load failed for {BERT_MODEL}; using regex token counts. ({e})")
            return None

    def _load_sentence_splitter(self):
        if not USE_SPACY_SENT_SPLIT:
            return None
        try:
            nlp = spacy.load(SENTENCE_SPLIT_MODEL)
        except Exception:
            try:
                nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                print(f"spaCy model not available for sentence splitting: {e}")
                return None
        if not nlp.has_pipe("parser") and not nlp.has_pipe("senter"):
            nlp.add_pipe("sentencizer")
        return nlp

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        return len(re.findall(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]", text))

    def _is_list_line(self, line: str) -> bool:
        return bool(re.match(r"^\s*([-*+]|(\d+[\.\)])|([A-Za-z]\))|\([A-Za-z0-9]+\))\s+", line))

    def _is_list_block(self, text: str) -> bool:
        lines = [l for l in text.splitlines() if l.strip()]
        if len(lines) < 3:
            return False
        list_lines = sum(1 for l in lines if self._is_list_line(l))
        return (list_lines / len(lines)) >= 0.5

    def _split_paragraphs(self, text: str) -> List[str]:
        paragraphs: List[str] = []
        buf: List[str] = []
        buf_kind: Optional[str] = None

        def flush():
            nonlocal buf, buf_kind
            if buf:
                paragraphs.append("\n".join(buf).strip())
            buf = []
            buf_kind = None

        for line in text.splitlines():
            if not line.strip():
                flush()
                continue
            line_kind = "list" if self._is_list_line(line) else "text"
            if buf_kind and line_kind != buf_kind:
                flush()
            buf_kind = line_kind
            buf.append(line)

        flush()
        return [p for p in paragraphs if p]

    def _section_chunk_params(self, section: str) -> Dict[str, object]:
        sec = (section or "").lower().strip()

        if sec in {"meds_discharge", "pmh"}:
            return {
                "target_tokens": CHUNK_LIST_TARGET_TOKENS,
                "max_tokens": CHUNK_LIST_MAX_TOKENS,
                "min_tokens": CHUNK_LIST_MIN_TOKENS,
                "overlap_tokens": CHUNK_LIST_OVERLAP_TOKENS,
                "force_list_split": True,
            }

        if sec in {"header"}:
            return {
                "target_tokens": CHUNK_HEADER_TARGET_TOKENS,
                "max_tokens": CHUNK_HEADER_MAX_TOKENS,
                "min_tokens": CHUNK_HEADER_MIN_TOKENS,
                "overlap_tokens": CHUNK_HEADER_OVERLAP_TOKENS,
                "force_list_split": False,
            }

        return {
            "target_tokens": CHUNK_NARR_TARGET_TOKENS,
            "max_tokens": CHUNK_NARR_MAX_TOKENS,
            "min_tokens": CHUNK_NARR_MIN_TOKENS,
            "overlap_tokens": CHUNK_NARR_OVERLAP_TOKENS,
            "force_list_split": False,
        }

    def _split_sentences(self, text: str) -> List[str]:
        if self.sent_splitter is not None:
            doc = self.sent_splitter(text)
            sents = [s.text.strip() for s in doc.sents if s.text.strip()]
            if sents:
                return sents
        return [s.strip() for s in re.split(r"(?<=[\.!?])\s+", text) if s.strip()]

    def _split_by_lines(
        self,
        text: str,
        target_tokens: int,
        max_tokens: int,
        min_tokens: int,
    ) -> List[str]:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines:
            return [text]

        chunks: List[str] = []
        buf: List[str] = []
        buf_tokens = 0

        def flush_buf():
            nonlocal buf, buf_tokens
            if buf:
                ch = "\n".join(buf).strip()
                if ch:
                    chunks.append(ch)
            buf = []
            buf_tokens = 0

        for line in lines:
            lt = self._estimate_tokens(line)
            if lt > max_tokens:
                flush_buf()
                chunks.append(line)
                continue
            if buf_tokens + lt <= target_tokens or buf_tokens < min_tokens:
                buf.append(line)
                buf_tokens += lt + 1
            else:
                flush_buf()
                buf.append(line)
                buf_tokens = lt + 1

        flush_buf()
        return chunks

    def _build_overlap_text(
        self,
        prev_chunk: str,
        overlap_tokens: int,
        overlap_sentences: int,
    ) -> str:
        if overlap_tokens > 0:
            sents = self._split_sentences(prev_chunk)
            if not sents:
                return ""
            buf: List[str] = []
            tok_count = 0
            for sent in reversed(sents):
                st = self._estimate_tokens(sent)
                if buf and tok_count + st > overlap_tokens:
                    break
                buf.insert(0, sent)
                tok_count += st
                if tok_count >= overlap_tokens:
                    break
            return " ".join(buf).strip()

        if overlap_sentences > 0:
            sents = self._split_sentences(prev_chunk)
            if not sents:
                return ""
            return " ".join(sents[-overlap_sentences:]).strip()

        return ""

    def _apply_overlap(
        self,
        chunks: List[str],
        overlap_tokens: int,
        max_tokens: int,
        overlap_sentences: int = 0,
    ) -> List[str]:
        if (overlap_tokens <= 0 and overlap_sentences <= 0) or len(chunks) <= 1:
            return chunks

        out = [chunks[0]]
        prev = chunks[0]

        for ch in chunks[1:]:
            overlap = self._build_overlap_text(prev, overlap_tokens, overlap_sentences)
            candidate = f"{overlap} {ch}".strip() if overlap else ch
            if overlap and self._estimate_tokens(candidate) > max_tokens:
                candidate = ch
            out.append(candidate)
            prev = ch

        return out

    def process_batch(self, batch_size: int = 10):
        """Process multiple notes into smart chunks."""

        total = self.raw_notes.count_documents({})
        print(f"Total notes available: {total}")

        processed_ids = {
            log["note_id"] for log in self.log_col.find({"status": "completed"})
        }
        print(f"Already processed: {len(processed_ids)}\n")

        cursor = self.raw_notes.find(
            {},
            {
                "_id": 1,
                TEXT_FIELD: 1,
                "subject_id": 1,
                "hadm_id": 1,
            },
        )

        notes_to_process = []
        for doc in cursor:
            note_id = str(doc["_id"])
            if note_id not in processed_ids:
                notes_to_process.append(doc)
                if len(notes_to_process) >= batch_size:
                    break

        if not notes_to_process:
            print("All notes already processed.")
            return

        print(f"Starting batch of {len(notes_to_process)} notes...\n")

        stats = {"success": 0, "failed": 0, "chunks": 0}

        for note in tqdm(notes_to_process, desc="Processing"):
            note_id = str(note["_id"])
            try:
                chunks = self._process_single(note)
                self.log_col.insert_one(
                    {
                        "note_id": note_id,
                        "status": "completed",
                        "chunks_created": len(chunks),
                        "processed_at": datetime.utcnow(),
                    }
                )
                stats["success"] += 1
                stats["chunks"] += len(chunks)

            except Exception as e:
                self.log_col.insert_one(
                    {
                        "note_id": note_id,
                        "status": "failed",
                        "error": str(e),
                        "processed_at": datetime.utcnow(),
                    }
                )
                stats["failed"] += 1

        print("\n--- BATCH COMPLETE ---")
        print(f"  Success     - {stats['success']}")
        print(f"  Failed      - {stats['failed']}")
        print(f"  Chunks Made - {stats['chunks']}")
        print("------------------------\n")

        if stats["success"] > 0:
            print("Next steps:")
            print("1. python add_embeddings.py")
            print("2. python build_faiss_index.py")
            print("3. python kg_extraction.py (optional)")
            print("4. python kg_rag_query.py\n")

        return stats

    def _process_single(self, note: Dict) -> List[Dict]:
        """Parse sections -> smart-chunk within each section -> store -> Neo4j."""
        note_id = str(note["_id"])
        text = note.get(TEXT_FIELD) or ""

        if not text.strip():
            text = str(note)

        existing = self.chunks_col.count_documents({"note_id": note_id})
        if existing > 0:
            print(f"SKIPPED: chunks for note_id={note_id} already exist ({existing})")
            return []

        sections = self._parse_sections(text)

        all_chunks: List[Dict] = []
        global_idx = 0

        for section_name, section_text in sections.items():
            if not section_text or len(section_text.strip()) < 30:
                continue

            smart_chunks = self._smart_chunk_section(section_text, section_name)

            for _local_idx, chunk_text in enumerate(smart_chunks):
                if len(chunk_text.strip()) < 30:
                    continue

                metadata = self._extract_metadata(chunk_text, section_name)

                chunk_doc = {
                    "chunk_id": f"{note_id}_{section_name}_{global_idx}",
                    "note_id": note_id,
                    "subject_id": note.get("subject_id"),
                    "hadm_id": note.get("hadm_id"),

                    "section": section_name,
                    "text": chunk_text,
                    "full_text": (
                        f"[Section: {section_name} | Temporal: {metadata['temporal']}]\n"
                        f"---\n{chunk_text}"
                    ),
                    "metadata": metadata,
                    "chunk_index": global_idx,
                    "created_at": datetime.utcnow(),
                }

                all_chunks.append(chunk_doc)
                global_idx += 1

        if all_chunks:
            self.chunks_col.insert_many(all_chunks)
            self._push_to_neo4j(note_id, all_chunks)

        return all_chunks

    def _parse_sections(self, text: str) -> Dict[str, str]:
        """
        Parse note into sections using marker phrases, case-insensitive.
        If no markers are found, entire note becomes one 'body' section.
        """
        sections: Dict[str, str] = {}
        current = "header"
        buffer: List[str] = []

        for line in text.split("\n"):
            raw = line.rstrip()
            stripped = raw.strip()
            heading_key = None
            heading_line = None

            if stripped:
                m = re.match(r"^([A-Za-z][A-Za-z0-9 /,-]*)\s*:\s*(.*)$", stripped)
                if m:
                    label, rest = m.group(1), m.group(2)
                    heading_key = self._match_heading_label(label)
                    if heading_key:
                        heading_line = f"{label}:{(' ' + rest) if rest else ''}".strip()

                if not heading_key and self._looks_like_heading(stripped):
                    heading_key = self._match_heading_label(stripped.rstrip(":"))
                    if heading_key:
                        heading_line = stripped if stripped.endswith(":") else f"{stripped}:"

            if heading_key:
                if buffer:
                    sections[current] = "\n".join(buffer).strip()
                current = heading_key
                buffer = []
                if heading_line:
                    buffer.append(heading_line)
            else:
                buffer.append(raw)

        if buffer:
            sections[current] = "\n".join(buffer).strip()

        if len(sections) == 1 and "header" in sections:
            sections = {"body": sections["header"]}

        return sections

    def _smart_chunk_section(
        self,
        text: str,
        section: str,
    ) -> List[str]:

        section_key = (section or "").lower().strip()
        text = text.strip()
        params = self._section_chunk_params(section_key)
        target_tokens = int(params["target_tokens"])
        max_tokens = int(params["max_tokens"])
        min_tokens = int(params["min_tokens"])
        overlap_tokens = int(params["overlap_tokens"])
        force_list_split = bool(params["force_list_split"])
        overlap_sentences = CHUNK_OVERLAP_SENTENCES

        if min_tokens > target_tokens:
            min_tokens = target_tokens
        if max_tokens < target_tokens:
            max_tokens = target_tokens

        if section_key == "header" and self._estimate_tokens(text) <= max_tokens:
            return [text]

        if self._estimate_tokens(text) <= max_tokens:
            return [text]

        raw_paragraphs = self._split_paragraphs(text)
        if not raw_paragraphs:
            raw_paragraphs = [text]

        chunks: List[str] = []
        current_buf: List[str] = []
        current_tokens = 0

        def flush_current():
            nonlocal current_buf, current_tokens
            if current_buf:
                chunk_text = "\n\n".join(current_buf).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                current_buf = []
                current_tokens = 0

        for para in raw_paragraphs:
            plen = self._estimate_tokens(para)

            if plen > max_tokens:
                flush_current()
                chunks.extend(
                    self._split_by_sentences(
                        para, target_tokens, max_tokens, min_tokens, force_list_split
                    )
                )
                continue

            if current_tokens + plen <= target_tokens or current_tokens < min_tokens:
                current_buf.append(para)
                current_tokens += plen + 2
            else:
                flush_current()
                current_buf.append(para)
                current_tokens = plen

        flush_current()

        final_chunks: List[str] = []
        for ch in chunks:
            if self._estimate_tokens(ch) > max_tokens:
                final_chunks.extend(
                    self._split_by_sentences(
                        ch, target_tokens, max_tokens, min_tokens, force_list_split
                    )
                )
            else:
                final_chunks.append(ch)

        return self._apply_overlap(
            final_chunks,
            overlap_tokens=overlap_tokens,
            max_tokens=max_tokens,
            overlap_sentences=overlap_sentences,
        )

    def _split_by_sentences(
        self,
        text: str,
        target_tokens: int,
        max_tokens: int,
        min_tokens: int,
        force_list_split: bool,
    ) -> List[str]:

        if force_list_split or self._is_list_block(text):
            return self._split_by_lines(text, target_tokens, max_tokens, min_tokens)

        sentences = self._split_sentences(text)
        if not sentences:
            return [text]

        chunks: List[str] = []
        buf: List[str] = []
        buf_tokens = 0

        def flush_buf():
            nonlocal buf, buf_tokens
            if buf:
                ch = " ".join(buf).strip()
                if ch:
                    chunks.append(ch)
                buf = []
                buf_tokens = 0

        for sent in sentences:
            slen = self._estimate_tokens(sent)

            if slen > max_tokens:
                flush_buf()
                chunks.append(sent)
                continue

            if buf_tokens + slen <= target_tokens or buf_tokens < min_tokens:
                buf.append(sent)
                buf_tokens += slen + 1
            else:
                flush_buf()
                buf.append(sent)
                buf_tokens = slen + 1

        flush_buf()
        return chunks

    def _extract_metadata(self, text: str, section: str) -> Dict:
        lo = text.lower()
        if any(w in lo for w in ["admission", "admitted", "presented"]):
            temporal = "admission"
        elif "discharge" in lo or "discharged" in lo:
            temporal = "discharge"
        else:
            temporal = "during_stay"

        return {
            "section": section,
            "temporal": temporal,
            "has_medications": bool(re.search(r"\b\d+\s*mg\b", text)),
            "has_labs": bool(re.search(r"\b(Na|K|Cr)\s*[:=]?\s*\d", text)),
        }

    def _push_to_neo4j(self, note_id: str, chunks: List[Dict]):
        with self.neo4j.session() as session:
            session.run(
                """
                MERGE (n:Note {note_id: $note_id})
                ON CREATE SET n.created_at = datetime()
                """,
                note_id=note_id,
            )

            for c in chunks:
                session.run(
                    """
                    MATCH (n:Note {note_id: $note_id})
                    MERGE (ch:Chunk {chunk_id: $chunk_id})
                    SET ch.section = $section
                    MERGE (n)-[:HAS_CHUNK]->(ch)
                    """,
                    note_id=note_id,
                    chunk_id=c["chunk_id"],
                    section=c["section"],
                )

    def cleanup(self):
        self.client.close()
        self.neo4j.close()

def main():
    try:
        bs = input("\nBatch size? (default 10): ").strip()
        batch_size = int(bs) if bs.isdigit() else 10

        processor = BatchProcessor()
        processor.process_batch(batch_size)

    except KeyboardInterrupt:
        print("\nCancelled by user")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            if 'processor' in locals():
                processor.cleanup()
        except Exception:
            pass


if __name__ == "__main__":
    main()
