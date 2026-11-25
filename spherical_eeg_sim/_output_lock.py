"""Serialize generation and recovery for each output directory."""

from contextlib import contextmanager
import errno
import os
from pathlib import Path
import sys
import threading
from typing import Iterator


# POSIX record locks belong to the process. Guard concurrent calls from its
# own threads before opening another descriptor for the same lock file.
_active: set[Path] = set()
_guard = threading.Lock()


def _after_fork() -> None:
    global _guard
    _guard = threading.Lock()
    _active.clear()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_after_fork)


@contextmanager
def lock_output(target: Path) -> Iterator[None]:
    """Hold a nonblocking OS lock across a complete publication transaction.

    Keep the sibling file after releasing the lock. Removing it would let a
    later caller lock a different file while another still holds the old one.
    The operating system releases the byte-range lock on close or process exit.
    """
    message = f"Another dataset generation is active for {target}"
    with _guard:
        if target in _active:
            raise BlockingIOError(errno.EAGAIN, message)
        _active.add(target)
    try:
        path = target.parent / f'.{target.name}.generation.lock'
        with path.open('a+b') as handle:
            handle.seek(0)
            try:
                if sys.platform == 'win32':
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.lockf(handle, fcntl.LOCK_EX | fcntl.LOCK_NB, 1)
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                raise BlockingIOError(errno.EAGAIN, message) from error
            yield
    finally:
        with _guard:
            _active.discard(target)
