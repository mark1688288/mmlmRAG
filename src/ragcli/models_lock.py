"""Serialize every stage that talks to Ollama.

The lock file is ``data/locks/ollama.lock``. Waiters print a notice every
15 seconds. A ``timeout`` of ``None`` waits until the lock is free or the
process is interrupted. KeyboardInterrupt still releases the lock.

``exclusive_model`` is the stage boundary: one flock, unload anything that
is not the target, then ``keep_alive`` 0 on the way out.
"""

from __future__ import annotations

import fcntl
import os
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from ragcli.errors import EXIT_RESOURCES, EXIT_UNCLASSIFIED, EXIT_USAGE, RagError
from ragcli.i18n import t

LOCK_NOTICE_SECONDS = 15.0
_POLL_SECONDS = 0.2
_PRIVATE_DIR = 0o700
_PRIVATE_FILE = 0o600


@contextmanager
def ollama_lock(
    data_dir: Path,
    *,
    timeout: float | None = None,
    stderr=None,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    poll: float = _POLL_SECONDS,
) -> Iterator[Path]:
    """Hold ``data/locks/ollama.lock`` for one Ollama stage."""
    if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float))):
        raise RagError(EXIT_USAGE, t("lock.timeout_type"))
    if timeout is not None and timeout < 0:
        raise RagError(EXIT_USAGE, t("config.not_negative", label="--lock-timeout"))
    if isinstance(poll, bool) or not isinstance(poll, (int, float)) or poll <= 0:
        raise RagError(EXIT_USAGE, t("lock.poll_positive"))
    root = _locks_dir(data_dir)
    path = root / "ollama.lock"
    fd = _open_lock(path)
    acquired = False
    try:
        _acquire(
            fd,
            timeout=timeout,
            now=now,
            sleep=sleep,
            stderr=stderr or sys.stderr,
            poll=poll,
        )
        acquired = True
        yield path
    finally:
        if acquired:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


@contextmanager
def exclusive_model(
    data_dir: Path,
    client,
    model: str,
    num_ctx: int,
    *,
    timeout: float | None = None,
    stderr=None,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    poll: float = _POLL_SECONDS,
) -> Iterator[None]:
    """Lock, prepare one model, and unload it when the stage ends.

    ``client`` provides ``prepare_stage(model, num_ctx)`` and
    ``release_stage(model)``. The lock is released if either call fails.
    """
    with ollama_lock(
        data_dir,
        timeout=timeout,
        stderr=stderr,
        now=now,
        sleep=sleep,
        poll=poll,
    ):
        client.prepare_stage(model, num_ctx)
        try:
            yield
        finally:
            client.release_stage(model)


def _acquire(fd: int, *, timeout, now, sleep, stderr, poll) -> None:
    started = now()
    notices = 0
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            pass
        current = now()
        due = int((current - started) // LOCK_NOTICE_SECONDS)
        while notices < due:
            print(t("lock.wait"), file=stderr, flush=True)
            notices += 1
        if timeout is not None and current >= started + timeout:
            raise RagError(
                EXIT_RESOURCES,
                t("lock.timeout", seconds=_format_seconds(timeout)),
            )
        remaining = None if timeout is None else (started + timeout) - current
        delay = poll if remaining is None else min(poll, remaining)
        if delay > 0:
            sleep(delay)


def _format_seconds(timeout: float) -> str:
    if isinstance(timeout, float) and timeout.is_integer():
        return str(int(timeout))
    return f"{timeout:g}"


def _locks_dir(data_dir: Path) -> Path:
    if not isinstance(data_dir, Path):
        raise RagError(EXIT_UNCLASSIFIED, t("lock.data_dir_type"))
    if data_dir.is_symlink():
        raise RagError(EXIT_UNCLASSIFIED, t("io.data_dir_symlink"))
    root = data_dir.expanduser().resolve()
    _ensure_private_dir(root)
    locks = root / "locks"
    if locks.is_symlink():
        raise RagError(EXIT_UNCLASSIFIED, t("io.data_dir_symlink"))
    _ensure_private_dir(locks)
    return locks


def _ensure_private_dir(path: Path) -> None:
    """Create ``path`` as 0700. An existing directory keeps its mode."""
    if path.is_symlink():
        raise RagError(EXIT_UNCLASSIFIED, t("io.data_dir_symlink"))
    if path.exists():
        if not path.is_dir():
            raise RagError(
                EXIT_UNCLASSIFIED,
                t("io.data_path_not_directory", name=path.name),
            )
        return
    try:
        path.mkdir(mode=_PRIVATE_DIR)
    except FileExistsError:
        if path.is_symlink() or not path.is_dir():
            raise RagError(EXIT_UNCLASSIFIED, t("io.data_dir_symlink")) from None
        return
    os.chmod(path, _PRIVATE_DIR)


def _open_lock(path: Path) -> int:
    if path.is_symlink():
        raise RagError(EXIT_UNCLASSIFIED, t("io.data_dir_symlink"))
    flags = os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        fd = os.open(path, flags | os.O_CREAT | os.O_EXCL, _PRIVATE_FILE)
    except FileExistsError:
        if path.is_symlink():
            raise RagError(EXIT_UNCLASSIFIED, t("io.data_dir_symlink")) from None
        fd = os.open(path, flags)
        return fd
    except OSError as exc:
        raise RagError(EXIT_UNCLASSIFIED, t("lock.create_failed")) from exc
    os.chmod(path, _PRIVATE_FILE)
    return fd
