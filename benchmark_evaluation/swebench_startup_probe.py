"""One bounded stack snapshot, installed before pytest/plugin imports."""
import io, json, os, time

pid = os.getpid()
value = {"kind": "phase", "binding": BINDING, "pid": pid,
         "phase": "python_start", "at": time.time()}
path = os.path.join(RECEIPT_DIR, "startup-%s.json" % pid)
with io.open(path + ".tmp", "w", encoding="utf-8") as stream:
    text = json.dumps(value, ensure_ascii=True)
    if not isinstance(text, type(u"")):
        text = text.decode("ascii")
    stream.write(text)
os.rename(path + ".tmp", path)
try:
    import faulthandler
    _trace = open(os.path.join(RECEIPT_DIR, "stack-%s.txt" % pid), "w")
    faulthandler.enable(file=_trace, all_threads=True)
    faulthandler.dump_traceback_later(STACK_SECONDS, repeat=False, file=_trace)
except (ImportError, AttributeError, RuntimeError):
    pass
