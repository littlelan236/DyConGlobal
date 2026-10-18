"""Process locks; the sidecar file itself is never treated as proof of ownership."""

from contextlib import contextmanager
from functools import wraps
from pathlib import Path


@contextmanager
def file_lock(path, *, nonblocking=False):
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as f:
        flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0)
        fcntl.flock(f, flags)
        try:
            yield f
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


@contextmanager
def directory_changes(path):
    """Native Linux filesystem notifications for cross-process admission wakes."""
    import ctypes
    import os
    import select

    library = ctypes.CDLL(None, use_errno=True)
    fd = library.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "inotify_init1")
    try:
        watch = library.inotify_add_watch(fd, os.fsencode(path), 0x80 | 0x08 | 0x100)
        if watch < 0:
            raise OSError(ctypes.get_errno(), "inotify_add_watch")

        def wait_for_change(timeout, extra_fds=()):
            ready = select.select([fd, *extra_fds], [], [], timeout)[0]
            if fd in ready:
                os.read(fd, 65536)
            return ready

        yield wait_for_change
    finally:
        os.close(fd)


def exclusive_run(function):
    @wraps(function)
    def wrapped(cfg, run, *args, **kwargs):
        from core.io import identifier

        name = identifier(Path(run).name)
        with file_lock(Path(cfg["project_root"]) / "state/run-locks" / (name + ".lock"), nonblocking=True):
            return function(cfg, run, *args, **kwargs)

    return wrapped


def exclusive_artifact(owner, parameter):
    def decorate(function):
        import inspect

        signature = inspect.signature(function)

        @wraps(function)
        def wrapped(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            from core.io import identifier

            name = identifier(bound.arguments[parameter])
            cfg = bound.arguments["cfg"]
            with file_lock(
                Path(cfg["project_root"]) / "state/artifact-locks" / owner / (name + ".lock"),
                nonblocking=True,
            ):
                return function(*args, **kwargs)

        return wrapped

    return decorate
