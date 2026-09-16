#!/usr/bin/env python3
"""Fail unless portable smoke logs prove cross-turn physical KV persistence."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    reports = []
    errors = []
    for log in sorted(args.root.glob("**/logs/proxy_*.jsonl")):
        rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        bad = [row for row in rows if row.get("status") != "ok"]
        lifecycle = [row.get("history_kv_lifecycle") for row in rows
                     if isinstance(row.get("history_kv_lifecycle"), dict)]
        sessions = defaultdict(list)
        for event in lifecycle:
            sessions[str(event.get("session_id") or "")].append(event)
            if event.get("history_kv_backend") != "physical_eviction":
                errors.append(f"{log}: non-physical lifecycle event")
            if event.get("persistent_session_enabled") is not True:
                errors.append(f"{log}: persistent_session_enabled is not true")
            if event.get("full_history_reprefill_performed") is not False:
                errors.append(f"{log}: full-history re-prefill was performed/unknown")
        nonempty = {key: value for key, value in sessions.items() if key}
        reusable = any(len(events) >= 2 for events in nonempty.values())
        if bad:
            errors.append(f"{log}: {len(bad)} proxy request errors")
        if not reusable:
            errors.append(
                f"{log}: fewer than two lifecycle events on one session; "
                "cross-turn persistence is not proven")
        chain_checks = 0
        for events in nonempty.values():
            for previous, current in zip(events, events[1:]):
                saved = previous.get("resident_position_summary")
                reused = current.get("previous_resident_position_summary")
                if saved is not None and reused is not None:
                    chain_checks += 1
                    if saved != reused:
                        errors.append(f"{log}: resident-position chain mismatch")
        reports.append({
            "request_log": str(log),
            "requests": len(rows),
            "lifecycle_events": len(lifecycle),
            "session_lengths": {key: len(value) for key, value in nonempty.items()},
            "resident_chain_checks": chain_checks,
            "persistent_reuse_proven": reusable,
        })
    if not reports:
        errors.append(f"{args.root}: no proxy request logs found")
    output = {"ok": not errors, "runs": reports, "errors": errors}
    path = args.root / "persistent_history_validation.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
