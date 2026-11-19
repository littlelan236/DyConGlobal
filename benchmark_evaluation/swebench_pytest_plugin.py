"""Python 2.7+ pytest phase evidence; no external runtime dependencies."""
import io, json, os, sys, time

_reports = []
_collection = []
_errors = []
_started = time.time()

def _write(name, value):
    path = os.path.join(RECEIPT_DIR, name)
    temporary = path + ".tmp"
    with io.open(temporary, "w", encoding="utf-8") as stream:
        text = json.dumps(value, sort_keys=True, ensure_ascii=True)
        if not isinstance(text, type(u"")):
            text = text.decode("ascii")
        stream.write(text)
    os.rename(temporary, path)

def _phase(phase, nodeid=None):
    _write("phase-%s.json" % os.getpid(), dict(kind="phase", binding=BINDING,
           pid=os.getpid(), phase=phase, nodeid=nodeid, at=time.time()))

_phase("plugin_import")

def pytest_sessionstart(session):
    _phase("session_start")

def pytest_collectreport(report):
    if report.failed:
        _errors.append({"nodeid": report.nodeid, "longrepr": type(u"")(report.longrepr)})

def pytest_collection_finish(session):
    _collection.extend(item.nodeid for item in session.items)
    _phase("collection_finish")

def pytest_runtest_logstart(nodeid, location):
    _phase("test_start", nodeid)

def pytest_runtest_logreport(report):
    row = {"nodeid": report.nodeid, "when": report.when, "outcome": report.outcome}
    if report.failed:
        row["longrepr"] = type(u"")(report.longrepr)
        crash = getattr(report.longrepr, "reprcrash", None)
        row["crash"] = {"path": getattr(crash, "path", None),
                        "lineno": getattr(crash, "lineno", None),
                        "message": getattr(crash, "message", None)}
    _reports.append(row)
    _phase(report.when, report.nodeid)

def pytest_sessionfinish(session, exitstatus):
    chain = []
    pid = os.getpid()
    while pid > 1 and len(chain) < 64:
        try:
            with io.open("/proc/%s/status" % pid, encoding="utf-8") as stream:
                status = stream.read()
            parent = int(next(x.split()[1] for x in status.splitlines() if x.startswith("PPid:")))
            chain.append({"pid": pid, "ppid": parent})
            pid = parent
        except (OSError, IOError, StopIteration, ValueError):
            break
    value = {"schema_version": 1, "binding": BINDING, "pid": os.getpid(), "ppid": os.getppid(),
             "ancestors": chain, "executable": sys.executable, "argv": sys.argv,
             "cwd": os.getcwd(), "started_at": _started, "exitstatus": int(exitstatus),
             "collected": _collection, "collection_errors": _errors, "reports": _reports}
    _write(str(os.getpid()) + ".json", value)
    _phase("session_finish")
    try:
        import faulthandler
        faulthandler.cancel_dump_traceback_later()
    except (ImportError, AttributeError, RuntimeError):
        pass
