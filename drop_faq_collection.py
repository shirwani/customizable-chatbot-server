"""Drop the ChromaDB `faq` collection for the current CLIENT_SITE.

Usage (from project root):

    source venv/bin/activate
    python code/drop_faq_collection.py

This script reads CLIENT_SITE from the environment (as set by your .env
and utils.py), computes that client's chroma_db path, and deletes the
`faq` collection if it exists.
"""

from __future__ import annotations

import os
import chromadb
from dotenv import load_dotenv

load_dotenv()


def drop_faq_collection() -> None:
    """Delete the `faq` collection for the configured client site if present."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    client_site = os.environ.get("CLIENT_SITE", "")

    if not client_site:
        raise RuntimeError(
            "CLIENT_SITE is not set in the environment/.env; cannot locate chroma_db path."
        )

    client_site_root = os.path.join(project_root, "client_sites", client_site)
    client_chroma_db_path = os.path.join(client_site_root, "chroma_db")

    print(f"CLIENT_SITE: {client_site}")
    print(f"Chroma DB path: {client_chroma_db_path}")

    client = chromadb.PersistentClient(path=client_chroma_db_path)

    try:
        client.delete_collection("faq")
        print("Deleted 'faq' collection.")
    except Exception as e:
        print(f"Could not delete 'faq' collection (may not exist yet): {e}")


if __name__ == "__main__":
    drop_faq_collection()
