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
