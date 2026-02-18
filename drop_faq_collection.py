"""Drop the ChromaDB `faq` collection for the current CLIENT_SITE.

Usage (from project root):

    source venv/bin/activate
    python code/drop_faq_collection.py

This script reads CLIENT_SITE from the environment (as set by your .env
and utils.py), computes that client's chroma_db path, and deletes the
`faq` collection if it exists.
"""

from __future__ import annotations
from llm_utils import get_chroma_db_client, get_faq_collection_name

def drop_faq_collection() -> None:
    """Delete the `faq` collection for the configured client site if present."""
    try:
        get_chroma_db_client().delete_collection(get_faq_collection_name())
        print(f"Deleted '{get_faq_collection_name()}' collection.")
    except Exception as e:
        print(f"Could not delete '{get_faq_collection_name()}' collection (may not exist yet): {e}")


if __name__ == "__main__":
    drop_faq_collection()
