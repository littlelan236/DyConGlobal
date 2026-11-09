import json
from types import SimpleNamespace
import numpy as np
import pytest
import requests
from hidden.backends.vllm.online import CaptureClient, capture_request_metadata
from runtime.serving.capture import AsyncExportQueue, CaptureRequest, _ExportItem, _capture_stage_summary
from runtime.serving.diagnostics import safe_response_summary, sanitize_error_text

def test_response_summary_is_bounded_and_drops_sensitive_fields():
    body=json.dumps({"error":"capture failed","Authorization":"Bearer secret",
                     "prompt":"do not persist","error_type":"RuntimeError"}).encode()
    summary=safe_response_summary(body)
    assert len(summary.encode())<=512
    assert "capture failed" in summary and "RuntimeError" in summary
    assert "secret" not in summary and "do not persist" not in summary
    assert sanitize_error_text("token_ids=[1,2]")=="[redacted diagnostic text]"

def test_capture_client_attaches_structured_http_diagnostic(monkeypatch):
    response=requests.Response(); response.status_code=500
    response._content=b'{"error":"server failed","Authorization":"Bearer secret","error_type":"RuntimeError"}'
    response.headers["X-DyCon-Correlation-ID"]="corr-123"
    def request(*args,**kwargs):
        assert kwargs["headers"]["X-DyCon-Correlation-ID"]
        return response
    monkeypatch.setattr(requests,"request",request)
    with pytest.raises(requests.HTTPError) as raised:
        CaptureClient("http://capture")._request("POST","/stop",{})
    diagnostic=raised.value.capture_diagnostic
    assert diagnostic["status_code"]==500 and diagnostic["reason_code"]=="http_5xx"
    assert diagnostic["correlation_id"]=="corr-123"
    assert "secret" not in diagnostic["response_summary"]

def test_async_queue_status_is_session_scoped(tmp_path):
    queue=AsyncExportQueue(tmp_path,8)
    try:
        for sid in ("session-a","session-b"):
            assert queue.submit(_ExportItem(sid,{"x":np.ones(2)},{"session_id":sid}))
        queue.drain_session("session-a",5); queue.drain_session("session-b",5)
        a=queue.status("session-a"); b=queue.status("session-b")
        assert a["session_metrics"]["persisted"]==b["session_metrics"]["persisted"]==1
        assert list(a["pending_by_session"])==["session-a"]
        assert "session-b" not in a["pending_by_session"]
    finally: queue.close()

def test_missing_point_is_attributed_to_first_absent_stage(tmp_path):
    request=CaptureRequest.from_mapping(capture_request_metadata(
        "s","task","run","rollout",0,["think_close"],close_token_ids=[9]))
    session=SimpleNamespace(
        session_id="s",default_request=request,
        diagnostic_identity={"sample_key":"sample","attempt_id":"physical","execution_attempt_id":"exec"},
        stage_diagnostics={"native":{"selector_observed":True,"selector_seen":{"think_close":1},"think_close_points":[{"position":12,"stages":["selector_seen","capture_row","submit_accepted"]}]}},
        stage_diagnostics_truncated_count=0,
        samples=[{"request_id":"native","absolute_position":12,"anchor":"think_close",
                  "sample_id":"sample-row","status":"queued"}])
    (tmp_path/"index").mkdir()
    (tmp_path/"index"/"sample-row.json").write_text(json.dumps({"status":"persisted","sha256":"abc"}),encoding="utf-8")
    expected={("native",12,"think_close"),("native",13,"think_close")}
    summary=_capture_stage_summary(session,expected,{("native",12,"think_close")},{"pending_total":1},tmp_path)
    assert summary["stages"]["expected"]==2 and summary["stages"]["selector_seen"]==1
    assert summary["stages"]["capture_row"]==1
    assert summary["missing"][0]["stage"]=="selector_seen"
    assert summary["missing"][0]["reason_code"]=="selector_not_seen"
    assert summary["missing"][0]["execution_attempt_id"]=="exec"


@pytest.mark.parametrize(("row_status", "persisted", "indexed", "expected_stage", "reason"), [
    (None, False, False, "capture_row", "capture_row_missing"),
    ("failed", False, False, "submit", "submit_rejected"),
    ("queued", False, False, "persist", "persist_pending"),
    ("queued", True, False, "indexed", "finalizer_join_missing"),
])
def test_missing_close_reports_first_stage_gap(tmp_path, row_status, persisted, indexed, expected_stage, reason):
    request = CaptureRequest.from_mapping(capture_request_metadata(
        "s", "task", "run", "rollout", 0, ["think_close"], close_token_ids=[9]))
    sample = {"request_id": "native", "absolute_position": 12, "anchor": "think_close",
              "sample_id": "sample-row", "status": row_status}
    session = SimpleNamespace(
        session_id="s", default_request=request, diagnostic_identity={},
        stage_diagnostics={"native": {"selector_seen": {"think_close": 1},
                                     "think_close_points": [{
                                         "position": 12,
                                         "stages": ["selector_seen"]
                                         + (["capture_row"] if row_status is not None else [])
                                         + (["submit_failed"] if row_status == "failed" else
                                            ["submit_accepted"] if row_status == "queued" else []),
                                     }]}},
        stage_diagnostics_truncated_count=0, samples=[] if row_status is None else [sample])
    if persisted:
        (tmp_path / "index").mkdir()
        (tmp_path / "index" / "sample-row.json").write_text(
            json.dumps({"status": "persisted", "sha256": "abc"}), encoding="utf-8")
    point=("native", 12, "think_close")
    observed={point} if indexed else set()
    summary=_capture_stage_summary(session,{point},observed,{"pending_total":1 if expected_stage == "persist" else 0},tmp_path)
    assert summary["missing"][0]["stage"] == expected_stage
    assert summary["missing"][0]["reason_code"] == reason


def test_complete_path_has_consistent_aggregate_counts(tmp_path):
    request=CaptureRequest.from_mapping(capture_request_metadata(
        "s","task","run","rollout",0,["think_close"],close_token_ids=[9]))
    session=SimpleNamespace(
        session_id="s",default_request=request,diagnostic_identity={},
        stage_diagnostics={"native":{"selector_seen":{"think_close":1},
                                     "think_close_points":[{"position":12,"stages":["selector_seen","capture_row","submit_accepted"]}]}},
        stage_diagnostics_truncated_count=0,
        samples=[{"request_id":"native","absolute_position":12,"anchor":"think_close",
                  "sample_id":"sample-row","status":"queued"}])
    (tmp_path/"index").mkdir()
    (tmp_path/"index"/"sample-row.json").write_text(
        json.dumps({"status":"persisted","sha256":"abc"}),encoding="utf-8")
    point=("native",12,"think_close")
    summary=_capture_stage_summary(session,{point},{point},{"pending_total":0},tmp_path)
    assert summary["stages"] == {
        "expected":1,"selector_seen":1,"capture_row":1,"submit":1,"persist":1,"indexed":1}
    assert summary["missing"] == [] and summary["by_request"] == {}


def test_session_diagnostic_report_endpoint_round_trips(tmp_path):
    from hidden.backends.vllm.online import CaptureClient
    from runtime.serving.capture import CaptureControlServer, OnlineCaptureController

    controller=OnlineCaptureController(tmp_path, {"export":{"enabled":False},"run_id":"run"})
    server=CaptureControlServer(controller).start()
    try:
        client=CaptureClient(server.base_url)
        metadata=capture_request_metadata("session","task","run","pending",0,["think_close"])
        client.start(metadata,graph_mode="FULL",
                    diagnostic_identity={"sample_key":"sample","attempt_id":"attempt",
                                         "execution_attempt_id":"execution"})
        client.bind("native-request",metadata)
        reply=client.report_diagnostics("session",{
            "schema_version":1,
            "requests":{"native-request":{
                "selector_seen":{"think_close":1},
                "capture_row":{"think_close":1},
                "submit_accepted":{"think_close":1},
                "think_close_points":[{"position":19,"stages":["selector_seen","capture_row","submit_accepted"]}]}},
            "truncated_count":0})
        assert reply["recorded"] is True
        assert controller.sessions["session"].stage_diagnostics["native-request"]["think_close_points"] == [{"position":19,"stages":["selector_seen","capture_row","submit_accepted"]}]
    finally:
        server.close()


def test_control_http_500_has_correlated_server_side_diagnostic(tmp_path):
    from core.io import read_json
    from hidden.backends.vllm.online import CaptureClient
    from runtime.serving.capture import CaptureControlServer, OnlineCaptureController

    controller=OnlineCaptureController(tmp_path, {"export":{"enabled":False},"run_id":"run"})
    server=CaptureControlServer(controller).start()
    try:
        client=CaptureClient(server.base_url)
        metadata=capture_request_metadata("session","task","run","pending",0,["reasoning_newline"])
        client.start(metadata,graph_mode="FULL",
                    diagnostic_identity={"sample_key":"sample","attempt_id":"attempt",
                                         "execution_attempt_id":"execution"})
        def fail_drain(session_id, timeout):
            raise RuntimeError("synthetic writer failure")
        controller.queue.drain_session=fail_drain
        with pytest.raises(requests.HTTPError) as raised:
            client.stop("session")
        diagnostic=raised.value.capture_diagnostic
        assert diagnostic["status_code"]==500
        assert diagnostic["error_type"]=="HTTPError"
        assert diagnostic["server_error_type"]=="RuntimeError"
        path=controller.export_root/"sessions/session/control-errors"/(diagnostic["correlation_id"]+".json")
        saved=read_json(path)
        assert saved["execution_attempt_id"]=="execution"
        assert saved["queue"]["before"] is not None and saved["queue"]["after"] is not None
        assert saved["error_summary"] == "[redacted diagnostic text]"
    finally:
        controller.queue.drain_session=lambda *_: None
        server.close()


def test_plugin_aggregates_close_stage_positions_once_per_session():
    from runtime.serving.vllm_plugin import DyConCapturePlugin

    plugin=DyConCapturePlugin.__new__(DyConCapturePlugin)
    plugin.request_metadata={"native":{"session_id":"session"}}
    plugin.stage_diagnostics={}
    plugin.stage_diagnostics_points={}
    plugin.stage_diagnostics_point_map={}
    plugin.stage_diagnostics_truncated={}
    plugin._count_diagnostic("native","selector_seen","think_close",42)
    plugin._count_diagnostic("native","capture_row","think_close",42)
    plugin._count_diagnostic("native","submit_accepted","think_close",42)
    snapshot=plugin._session_stage_diagnostics("session")
    record=snapshot["requests"]["native"]
    assert record["selector_seen"]["think_close"]==1
    assert record["capture_row"]["think_close"]==1
    assert record["submit_accepted"]["think_close"]==1
    assert record.get("think_close_points", []) == []


def test_plugin_caps_exception_point_metadata():
    from runtime.serving.vllm_plugin import DyConCapturePlugin

    plugin=DyConCapturePlugin.__new__(DyConCapturePlugin)
    plugin.request_metadata={"native":{"session_id":"session"}}
    plugin.stage_diagnostics={}
    plugin.stage_diagnostics_points={}
    plugin.stage_diagnostics_point_map={}
    plugin.stage_diagnostics_truncated={}
    for position in range(257):
        plugin._count_diagnostic("native","selector_seen","think_close",position)
    snapshot=plugin._session_stage_diagnostics("session")
    assert len(snapshot["requests"]["native"]["think_close_points"])==256
    assert snapshot["truncated_count"]==1


def test_native_selector_observer_uses_host_rows_and_native_exclusions(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from runtime.serving.capture_worker import CaptureWorkerExtension

    worker = CaptureWorkerExtension()
    worker._dycon_close_ids = frozenset({9})
    worker._dycon_selector = {}
    worker.model_runner = SimpleNamespace(input_batch=SimpleNamespace(
        req_id_to_index={"first": 1, "excluded": 0},
        token_ids_cpu=np.array([[0, 0, 0, 9], [0, 0, 0, 9]]),
    ))
    geometry = SimpleNamespace(
        query_start_loc_cpu=np.array([0, 1, 2]),
        num_prompt=torch.tensor([2, 2]), num_computed=torch.tensor([3, 3]),
        num_output=torch.tensor([2, 2]), req_ids=["first", "excluded"],
    )
    session = SimpleNamespace(
        _streams={"tact_states": SimpleNamespace(config=SimpleNamespace(select={"generation_tokens": [9]}))},
        _request_selects={"excluded": {"tact_states": {
            "generation_tokens": [9], "exclude_generation_positions": [1],
        }}},
    )

    def forbidden_cpu_copy(*args, **kwargs):
        raise AssertionError("observer added a device-to-host copy")

    monkeypatch.setattr(torch.Tensor, "cpu", forbidden_cpu_copy)
    worker._observe_close_candidates(geometry, session)
    assert worker._dycon_selector["first"] == {"positions": [3], "count": 1}
    assert worker._dycon_selector["excluded"] == {"positions": [], "count": 0}


def diagnostic_session(*, samples=(), reports=None):
    request = CaptureRequest.from_mapping(capture_request_metadata(
        "s", "task", "run", "rollout", 0, ["think_close"], close_token_ids=[9]))
    return SimpleNamespace(session_id="s", default_request=request, diagnostic_identity={},
        stage_diagnostics=reports or {}, stage_diagnostics_truncated_count=0, samples=list(samples))


def test_queued_sample_is_not_persisted_or_indexed(tmp_path):
    point = ("native", 12, "think_close")
    session = diagnostic_session(samples=[{"request_id": "native", "absolute_position": 12,
        "anchor": "think_close", "sample_id": "row", "status": "queued"}])
    summary = _capture_stage_summary(session, {point}, {point}, {"pending_total": 0}, tmp_path, indexed_available=False)
    assert summary["stages"]["persist"] == 0
    assert summary["stages"]["indexed"] is None
    assert summary["missing"][0]["reason_code"] == "persist_failed"
    assert summary["missing_count"] == 1


def test_observer_failure_stays_unavailable_after_later_report(tmp_path):
    from runtime.serving.capture import OnlineCaptureController
    controller = OnlineCaptureController(tmp_path, {"export": {"enabled": False}, "run_id": "run"})
    try:
        metadata = capture_request_metadata("s", "task", "run", "rollout", 0, ["think_close"])
        controller.start(metadata)
        controller.bind("native", metadata)
        for errors in (1, 0):
            controller.record_stage_diagnostics("s", {"schema_version": 1, "requests": {
                "native": {"selector_observed": True, "observer_errors": errors}}})
        summary = _capture_stage_summary(controller.sessions["s"], {("native", 12, "think_close")},
            set(), {}, tmp_path)
        assert summary["missing"][0]["reason_code"] == "selector_evidence_unavailable"
    finally:
        controller.queue.close()


def test_counts_survive_more_than_256_points_and_requests(tmp_path):
    reports = {f"r{i}": {"selector_observed": True, "selector_seen": {"think_close": 300},
                        "capture_row": {"think_close": 290}, "submit_accepted": {"think_close": 280}}
               for i in range(2)}
    session = diagnostic_session(reports=reports)
    session.stage_diagnostics_truncated_count = 600
    expected = {(rid, pos, "think_close") for rid in reports for pos in range(300)}
    summary = _capture_stage_summary(session, expected, set(), {}, tmp_path)
    assert summary["stages"]["selector_seen"] == 600
    assert summary["stages"]["capture_row"] == 580
    assert summary["stages"]["submit"] == 560
    assert summary["missing_count"] == 600 and summary["truncated_count"] == 344


def test_shared_execution_budget_across_sessions_and_control_errors(tmp_path):
    from runtime.serving.diagnostics import bounded_records
    rows = [{"request_id": "first", "absolute_position": p, "anchor": "think_close"} for p in range(200)]
    assert len(bounded_records(tmp_path, "run", "000", rows)) == 200
    other = [{"request_id": "second", "absolute_position": p, "anchor": "think_close"} for p in range(100)]
    assert len(bounded_records(tmp_path, "run", "000", other)) == 56
    assert bounded_records(tmp_path, "run", "000", [{"correlation_id": "error"}], kind="control") == []
    assert len(bounded_records(tmp_path, "run", "000", rows)) == 200
    assert len(bounded_records(tmp_path, "run", "001", other)) == 100


def test_final_index_diagnostics_distinguishes_join_missing_from_unknown(tmp_path):
    from runtime.serving.diagnostics import write_index_diagnostics
    point = ("native", 12, "think_close")
    requests = {"native": {"session_id": "s", "run_id": "run"}}
    summary = write_index_diagnostics(tmp_path, tmp_path / "out", requests, {point}, set(), {point})
    assert summary["stages"]["persist"] == 1 and summary["stages"]["indexed"] == 0
    assert summary["missing"][0]["reason_code"] == "finalizer_join_missing"
    summary = write_index_diagnostics(tmp_path, tmp_path / "out", requests, {point}, set(), set())
    assert summary["missing"][0]["reason_code"] == "stage_evidence_unavailable"


def test_engine_flush_500_has_correlated_sanitized_stack(tmp_path, monkeypatch):
    import asyncio
    import httpx
    from fastapi import FastAPI
    from runtime.serving.vllm_plugin import DyConCapturePlugin
    plugin = DyConCapturePlugin.__new__(DyConCapturePlugin)
    plugin.native_requests = {}
    plugin.owner_run, plugin.owner_attempt = "run", "000"
    plugin.staging = tmp_path / "staging"
    async def fail():
        raise RuntimeError("prompt=LEAK_SENTINEL Authorization=Bearer secret")
    plugin.drain_once = fail
    monkeypatch.setenv("DYCON_CAPTURE_PROJECT", str(tmp_path))
    app = FastAPI()
    plugin.attach_router(app)
    async def call():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/dycon/capture/engine/flush", json={"session_id": "s"},
                                     headers={"X-DyCon-Correlation-ID": "flush-corr"})
    response = asyncio.run(call())
    assert response.status_code == 500 and response.json()["correlation_id"] == "flush-corr"
    saved = json.loads((tmp_path / "control-errors/flush-corr.json").read_text())
    assert saved["error_type"] == "RuntimeError" and saved["stack"]
    assert "LEAK_SENTINEL" not in json.dumps(saved)


def test_diagnostic_write_failure_does_not_replace_original_control_error(tmp_path, monkeypatch):
    from runtime.serving.capture import CaptureControlServer, OnlineCaptureController
    controller = OnlineCaptureController(tmp_path, {"export": {"enabled": False}, "run_id": "run"})
    server = CaptureControlServer(controller).start()
    try:
        client = CaptureClient(server.base_url)
        client.start(capture_request_metadata("s", "task", "run", "pending", 0, ["reasoning_newline"]))
        def fail(*args, **kwargs):
            raise RuntimeError("original writer failure")
        def diagnostic_fail(*args, **kwargs):
            raise OSError("diagnostic disk failure")
        monkeypatch.setattr(controller.queue, "drain_session", fail)
        monkeypatch.setattr(controller, "record_control_failure", diagnostic_fail)
        with pytest.raises(requests.HTTPError) as raised:
            client.stop("s")
        assert raised.value.capture_diagnostic["server_error_type"] == "RuntimeError"
    finally:
        controller.queue.drain_session = lambda *a: None
        server.close()


def test_skillsbench_deadline_and_stop500_keep_timeout_primary(tmp_path, monkeypatch):
    from runtime.agents.skillsbench import drain_capture, backend_failure_result
    from core.io import write_json
    response = requests.Response()
    response.status_code, response._content = 200, b"{}"
    monkeypatch.setattr(requests, "post", lambda *a, **k: response)
    failure = requests.HTTPError("stop failed")
    failure.capture_diagnostic = {"schema_version": 1, "stage": "control_request", "status_code": 500,
                                  "error_type": "HTTPError", "server_error_type": "RuntimeError"}
    class Client:
        timeout = 1
        def stop(self, sid):
            raise failure
    errors = []
    with pytest.raises(requests.HTTPError):
        drain_capture({"serving": {"base_url": "http://test/v1"}, "capture": {"finalize_seconds": 1}},
                      tmp_path, Client(), "s", errors)
    write_json(tmp_path / "capture-stop-error.json", {"error": "stop500"})
    sample = SimpleNamespace(as_dict=lambda: {"sample_key": "sample"})
    result = backend_failure_result(sample, {}, tmp_path, "now", TimeoutError("attempt deadline"),
                                    capture_session="s", agent_status="TimeoutError")
    assert result["exit_status"] == result["agent_exit_status"] == "TimeoutError"
    assert errors[0]["status_code"] == 500 and result["capture_errors"]


@pytest.mark.parametrize("text", ["LEAK_SENTINEL", "pass123", "Bearer ABC", "prompt text",
    "output text", "tensor data", "sk-123456", "authorization abc", "token_ids 1 2 3"])
def test_unlabelled_and_labelled_secrets_are_never_free_text_diagnostics(text):
    summary = safe_response_summary(json.dumps({"error": text, "message": text, "detail": text,
                                               "prompt": text, "token_ids": [1, 2, 3]}).encode())
    assert text not in summary
    assert len(summary.encode()) <= 512


def test_attempt_fallback_uses_exact_session_and_shared_budget(tmp_path):
    from runtime.serving.diagnostics import write_index_diagnostics
    from core.io import write_json
    run = tmp_path / "runs/run"
    write_json(run / "plan.json", {})
    write_json(run / "resolved-config.json", {"project_root": str(tmp_path)})
    write_json(run / "rollout/tasks/task/repetitions/rep/attempts/physical/capture-diagnostics.json", {
        "run_id": "run", "execution_attempt_id": "000", "session_id": "s", "sample_key": "sample",
        "attempt_id": "physical", "missing": [{"request_id": "native", "absolute_position": 12,
            "anchor": "think_close", "stage": "capture_row", "reason_code": "capture_row_missing",
            "correlation_confidence": "exact"}]})
    summary = write_index_diagnostics(tmp_path / "nas", run / "enrichment/online",
        {"native": {"session_id": "s", "run_id": "run"}}, {("native", 12, "think_close")}, set(), set())
    assert summary["missing"][0]["reason_code"] == "capture_row_missing"
    assert summary["missing"][0]["sample_key"] == "sample"
    assert summary["stages"]["selector_seen"] == 1
    assert summary["stages"]["capture_row"] == 0


@pytest.mark.parametrize("count", [92, 300])
def test_real_finalizer_missing_points_keep_coverage_and_bounded_diagnostics(tmp_path, count):
    from core.io import read_json, write_json, sha256
    from runtime.serving.capture import OnlineCaptureController
    from enrichment.online_trace import finalize_token_events, OnlineCoverageError
    controller = OnlineCaptureController(tmp_path, {"run_id": "run", "open_token_ids": [10],
        "export": {"root": "online"}, "finalize_seconds": 1})
    try:
        metadata = capture_request_metadata("s", "task", "run", "pending", 0, ["think_close"], close_token_ids=[11])
        controller.start(metadata, diagnostic_identity={"execution_attempt_id": "000"})
        messages = []
        reports = {}
        for index in range(count):
            rid = f"native-{index:04d}"
            md = {**metadata, "step_id": index}
            controller.bind(rid, md)
            controller.record_output({"request_id": rid, "prompt_token_ids": [10], "output_token_ids": [11, 20]})
            reports[rid] = {"selector_observed": True, "selector_seen": {"think_close": 0}}
            messages.append({"role": "assistant", "extra": {"response": {
                "prompt_token_ids": [10], "output_token_ids": [11, 20], "dycon_native_request_id": rid}}})
        controller.record_stage_diagnostics("s", {"schema_version": 1, "requests": reports})
        trajectory = tmp_path / "trajectory.json"
        write_json(trajectory, {"messages": messages})
        controller.finalize("s", rollout_id="rollout", trajectory_path=str(trajectory),
                            trajectory_sha256=sha256(trajectory), model_fingerprint="model")
        out = tmp_path / "index-output"
        with pytest.raises(OnlineCoverageError):
            finalize_token_events(controller.export_root, out, open_ids=[10], close_ids=[11])
        coverage = read_json(out / "coverage.json")
        diagnostic = read_json(out / "capture-diagnostics.json")
        assert coverage["expected"] == count and coverage["observed"] == 0
        assert len(coverage["missing"]) == count
        assert diagnostic["missing_count"] == count
        assert len(diagnostic["missing"]) == min(count, 256)
        assert diagnostic["truncated_count"] == max(count - 256, 0)
        assert diagnostic["reason_counts"]["selector_not_seen"] == min(count, 256)
        assert diagnostic["stages"]["indexed"] == 0
    finally:
        controller.close()


@pytest.mark.parametrize("status", [400, 500])
def test_staged_submit_http_failure_has_counts_and_no_sensitive_body(tmp_path, monkeypatch, status):
    from runtime.serving.vllm_plugin import DyConCapturePlugin
    plugin = DyConCapturePlugin.__new__(DyConCapturePlugin)
    plugin.request_metadata = {"native": {"session_id": "s"}}
    plugin.stage_diagnostics, plugin.stage_diagnostics_point_map, plugin.stage_diagnostics_truncated = {}, {}, {}
    plugin.staging = tmp_path / "staging"
    plugin.staging.mkdir()
    plugin.budget, plugin.control_url, plugin.control_seconds = None, "http://test", 1
    plugin.failures, plugin.failure_policy, plugin.rows_published = [], "record_and_continue", 0
    response = requests.Response()
    response.status_code = status
    response._content = b'{"error_type":"RuntimeError","prompt":"LEAK_SENTINEL","error":"pass123"}'
    response.headers["X-DyCon-Correlation-ID"] = "submit-corr"
    monkeypatch.setattr(requests, "post", lambda *a, **kw: response)
    plugin._publish_prepared([("native", 12, 9, "think_close", {"layer_0": np.ones(2)},
                               {"graph_execution": "full", "batch_kind": "decode"})])
    saved = json.loads(next((tmp_path / "failures").glob("*.json")).read_text())
    assert saved["stage"] == "submit" and saved["status_code"] == status
    assert saved["correlation_id"] == "submit-corr"
    assert "LEAK_SENTINEL" not in json.dumps(saved) and "pass123" not in json.dumps(saved)
    assert plugin._session_stage_diagnostics("s")["requests"]["native"]["submit_failed"]["think_close"] == 1
    assert list(plugin.staging.glob("*.npz"))  # Recovery handoff stays intact.


def test_writer_failure_counters_and_session_snapshot_are_isolated(tmp_path, monkeypatch):
    from runtime.serving.capture import OnlineCaptureController
    controller = OnlineCaptureController(tmp_path, {"export": {"root": "online"}})
    original = controller.queue._persist
    def persist(item):
        if item.metadata["session_id"] == "bad":
            raise OSError("prompt=LEAK_SENTINEL")
        return original(item)
    monkeypatch.setattr(controller.queue, "_persist", persist)
    try:
        for sid in ("bad", "good"):
            controller.queue.submit(_ExportItem(sid, {"layer_0": np.ones(2)}, {"session_id": sid}))
            controller.queue.drain_session(sid, 5)
        bad, good = (controller._diagnostic_queue_status(sid) for sid in ("bad", "good"))
        assert bad["session_metrics"]["persist_failed"] == 1
        assert good["session_metrics"]["persisted"] == 1
        assert good["session_metrics"].get("persist_failed", 0) == 0
        assert "LEAK_SENTINEL" not in json.dumps(bad)
        assert "good" not in json.dumps(bad)
    finally:
        controller.close()


def test_budget_write_fault_preserves_counts_without_unbounded_details(tmp_path, monkeypatch):
    import runtime.serving.diagnostics as diagnostics
    def fail(*args, **kwargs):
        raise OSError("diagnostic disk unavailable")
    monkeypatch.setattr(diagnostics, "bounded_records", fail)
    summary = diagnostics.cap_stage_summary(tmp_path, {"run_id": "run", "missing_count": 92,
        "missing": [{"request_id": "r", "absolute_position": 1, "anchor": "think_close"}],
        "stages": {"expected": 92, "selector_seen": 0}})
    assert summary["stages"]["expected"] == 92
    assert summary["missing"] == [] and summary["truncated_count"] == 92
