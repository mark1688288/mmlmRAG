"""Process exit codes.

The numbers are the contract shared with the design: 0 success, 1 unclassified,
2 usage, 3 unsupported suffix, 4 missing or unreadable, 5 document rejected,
6 Ollama unreachable, 7 model missing, 8 no chunks, 9 question or context,
10 disk, 11 no collection, 12 index mismatch, 13 resource gate.
"""

EXIT_OK = 0
EXIT_UNCLASSIFIED = 1
EXIT_USAGE = 2
EXIT_UNSUPPORTED_TYPE = 3
EXIT_NOT_FOUND = 4
EXIT_DOCUMENT = 5
EXIT_OLLAMA_UNREACHABLE = 6
EXIT_MODEL_MISSING = 7
EXIT_NO_CHUNKS = 8
EXIT_QUESTION = 9
EXIT_DISK = 10
EXIT_NO_COLLECTION = 11
EXIT_MISMATCH = 12
EXIT_RESOURCES = 13


class RagError(Exception):
    """A failure that already has a process exit code and a stderr message."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
