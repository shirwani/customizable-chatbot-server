"""Create a ChromaDB collection from the client's FAQ text file.

This script reads the FAQ file for the configured CLIENT_SITE and stores
its contents in a persistent ChromaDB collection located under that
client's `chroma_db` directory.

The collection is called "faq".

Usage (from project root):

    python -m code.create_chromadb_faq_collection

or

    python code/create_chromadb_faq_collection.py

The script is idempotent with respect to document IDs: it will skip
FAQ entries whose IDs are already present in the collection, so you can
safely re-run it after updating the FAQ file.
"""

from __future__ import annotations

import os
import re
from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer

from utils import read_from_text_file

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Collection name requested by the user
COLLECTION_NAME = "faq"

# Base path for the current client site (from .env via utils)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIENT_SITE = os.environ.get("CLIENT_SITE", "")

if not CLIENT_SITE:
    raise RuntimeError(
        "CLIENT_SITE is not set in the environment/.env; cannot locate FAQ file."
    )

CLIENT_SITE_ROOT = os.path.join(PROJECT_ROOT, "client_sites", CLIENT_SITE)
FAQ_FILE_PATH = os.path.join(CLIENT_SITE_ROOT, "faq", "faq.txt")
CLIENT_CHROMA_DB_PATH = os.path.join(CLIENT_SITE_ROOT, "chroma_db")

# Reuse the same model as products for consistency / speed
FAQ_EMBEDDER_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_faq_file_exists() -> None:
    """Validate that the FAQ file exists."""
    if not os.path.isfile(FAQ_FILE_PATH):
        raise FileNotFoundError(
            f"FAQ file not found at {FAQ_FILE_PATH}. "
            "Expected path: client_sites/CLIENT_SITE/faq/faq.txt"
        )


def _ensure_collection():
    """Create or get the ChromaDB collection used for FAQ content."""
    # Ensure the target directory exists
    os.makedirs(CLIENT_CHROMA_DB_PATH, exist_ok=True)

    client = chromadb.PersistentClient(path=CLIENT_CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return collection


def _get_existing_ids(collection) -> set:
    """Fetch all existing IDs in the collection to support idempotent indexing."""
    existing_ids: set = set()
    res = collection.get(include=["metadatas"], limit=100000)
    for _id in res.get("ids", []) or []:
        existing_ids.add(str(_id))
    return existing_ids


def _parse_faq_entries(text: str) -> List[Dict[str, str]]:
    """Parse the FAQ text into structured entries.

    Expected format for each entry:

        Q: <question>\n
        Keywords: kw1, kw2, ...\n
        A: <answer>\n
    """
    entries: List[Dict[str, str]] = []

    # Split on blank lines between blocks
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]

    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if len(lines) < 3:
            continue

        q_line = next((ln for ln in lines if ln.lower().startswith("q:")), None)
        k_line = next((ln for ln in lines if ln.lower().startswith("keywords:")), None)
        a_line = next((ln for ln in lines if ln.lower().startswith("a:")), None)

        if not (q_line and k_line and a_line):
            continue

        question = q_line.split(":", 1)[1].strip()
        keywords_raw = k_line.split(":", 1)[1].strip()
        answer = a_line.split(":", 1)[1].strip()

        keywords = [kw.strip() for kw in keywords_raw.split(",") if kw.strip()]

        entry = {
            "question": question,
            "keywords": keywords,
            "answer": answer,
        }
        entries.append(entry)

    return entries


def _build_text_for_entry(entry: Dict[str, str]) -> str:
    """Build a text representation for embedding using both question and answer.

    This helps the semantic search match queries against either the
    question wording or key phrases contained only in the answer.
    """
    question = (entry.get("question") or "").strip()
    answer = (entry.get("answer") or "").strip()
    if question and answer:
        return f"Q: {question} \nA: {answer}"
    return question or answer


# ---------------------------------------------------------------------------
# Main indexing logic
# ---------------------------------------------------------------------------

def index_faq_to_chroma() -> None:
    """Index the client's FAQ entries into the Chroma `faq` collection.

    - One Chroma document per Q/A entry
    - Document text = question (for fast semantic search)
    - Metadata = question, answer, keywords, client_site
    """

    _ensure_faq_file_exists()

    collection = _ensure_collection()
    existing_ids = _get_existing_ids(collection)

    faq_text: str = read_from_text_file(FAQ_FILE_PATH)
    if not faq_text.strip():
        print("FAQ file is empty; nothing to index.")
        return

    entries = _parse_faq_entries(faq_text)
    if not entries:
        print("No FAQ entries parsed; check FAQ format.")
        return

    embedder = SentenceTransformer(FAQ_EMBEDDER_MODEL_NAME)

    docs: List[str] = []
    metadatas: List[Dict[str, object]] = []
    ids: List[str] = []

    for idx, entry in enumerate(entries, start=1):
        doc_id = f"faq-{idx}"
        if doc_id in existing_ids:
            continue

        text = _build_text_for_entry(entry)
        docs.append(text)
        metadatas.append(
            {
                "question": entry.get("question", ""),
                "answer": entry.get("answer", ""),
                "keywords": entry.get("keywords", []),
                "client_site": CLIENT_SITE,
                "source": "faq.txt",
            }
        )
        ids.append(doc_id)

    if not ids:
        print("All FAQ entries already indexed; nothing new to add.")
        return

    embeddings = embedder.encode(docs, convert_to_numpy=True)
    collection.add(documents=docs, metadatas=metadatas, ids=ids, embeddings=embeddings)

    print(
        f"Indexed {len(ids)} FAQ entries for client '{CLIENT_SITE}' into Chroma collection "
        f"'{COLLECTION_NAME}' at '{CLIENT_CHROMA_DB_PATH}'."
    )


if __name__ == "__main__":
    index_faq_to_chroma()
