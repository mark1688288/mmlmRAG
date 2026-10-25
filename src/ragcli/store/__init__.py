"""Local corpus storage.

SQLite is the authority for document text. Chroma is the vector index for one
embedding model and width.
"""

from ragcli.store.chroma import ChromaIndex
from ragcli.store.sqlite import SqliteStore

__all__ = ["ChromaIndex", "SqliteStore"]
