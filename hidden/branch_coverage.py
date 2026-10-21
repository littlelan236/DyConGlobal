"""Close-event completeness, shared by live capture and artifact repair."""

from enrichment.online_trace import reasoning_targets


def classify_close(receipt, matched, open_ids=(248068,), close_ids=(248069,)):
    if receipt is None:
        return {"status": "partial", "reason": "missing_request_receipt", "availability": []}
    output = receipt["output_token_ids"]
    close_ids = list(receipt.get("close_token_ids", close_ids))
    targets, available = reasoning_targets(receipt, list(open_ids), close_ids)
    has_close = any(output[i:i + len(close_ids)] == close_ids
                    for i in range(len(output) - len(close_ids) + 1))
    if not has_close:
        return {"status": "complete", "reason": "truncated_before_close"
                if receipt.get("finish_reason") == "length" else "no_close_generated",
                "availability": [], "not_required": True}
    expected = {position for position, anchor in targets if anchor == "think_close"}
    observed = {event["absolute_position"] for event in matched}
    complete = bool(expected) and expected <= observed and not any(
        row["status"] == "no_followup_forward" for row in available)
    return {"status": "complete" if complete else "partial",
            "reason": None if complete else "no_followup_forward" if any(
                row["status"] == "no_followup_forward" for row in available) else "missing_close_hidden",
            "availability": available}


def summarize_branches(rows):
    missing, not_required = [], []
    for row in rows:
        state = row["close_capture"]
        item = {"decision_id": row["decision_id"], "branch_id": row["branch_id"], **state}
        if state["status"] != "complete":
            missing.append(item)
        elif state.get("not_required"):
            not_required.append(item)
    expected = len(rows) - len(not_required)
    return {"schema_version": 2, "candidate_count": len(rows), "expected": expected,
            "observed": expected - len(missing), "missing": missing, "not_required": not_required}
