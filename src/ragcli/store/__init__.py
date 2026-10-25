"""Local corpus storage.

SQLite is the authority for document text. Chroma is added by a later change.
"""

from ragcli.store.sqlite import SqliteStore

__all__ = ["SqliteStore"]
