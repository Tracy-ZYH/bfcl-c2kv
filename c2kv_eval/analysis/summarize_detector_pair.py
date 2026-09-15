"""Count official held-out scores; Oracle is never used for test selection."""
import argparse
import csv
import json
from pathlib import Path
from c2kv_eval.analysis.select_combined_logistic_v2_thresholds import _score_summary, _read_jsonl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    root = p.parse_args().root
    rows = []
    for variant in ("rule", "trained"):
        if not (root / variant).exists():
            continue
        for method, directory in (("detector", "detector_combined_logistic_v3"),
                                  ("oracle", "detector_oracle"),
                                  ("never_trigger", "detector_never_trigger")):
            correct = total = 0
            ids = []
            segments = []
            for fold in range(5):
                cell = root / variant / directory / f"fold_{fold}"
                _, c, n = _score_summary(cell)
                correct += c
                total += n
                details = _read_jsonl(cell / "logs/details.jsonl")
                ids.extend(str(r["id"]) for r in details)
                segments.extend(s for r in details for s in r.get("repair_segments", []))
            if total != 52 or len(ids) != 52 or len(set(ids)) != 52:
                raise RuntimeError(f"Incomplete held-out results: {variant}/{method}: scored={total}, episodes={len(ids)}")
            rows.append(dict(variant=variant, method=method, correct=correct,
                             num_cases=total, bfcl_accuracy=correct / total,
                             trigger_rate=sum(bool(s.get("repair_triggered")) for s in segments) / len(segments) if segments else None))
    for row in rows:
        oracle = next(r for r in rows if r["variant"] == row["variant"] and r["method"] == "oracle")
        row["oracle_gap_pp"] = 100 * (oracle["bfcl_accuracy"] - row["bfcl_accuracy"])
    with (root / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "summary.md").write_text("| Variant | Method | Correct/52 | BFCL | Oracle gap (pp) |\n|---|---|---:|---:|---:|\n" + "".join(
        f"| {r['variant']} | {r['method']} | {r['correct']}/52 | {r['bfcl_accuracy']:.2%} | {r['oracle_gap_pp']:.2f} |\n" for r in rows))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
