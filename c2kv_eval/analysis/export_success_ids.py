from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _find_first(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern!r} under {root}")
    return matches[0]


def export_success_ids(args: argparse.Namespace) -> None:
    mode_root = Path(args.run_root) / args.mode
    result_path = Path(args.result_path) if args.result_path else _find_first(
        mode_root / "result", f"*_{args.category}_result.json"
    )
    score_path = Path(args.score_path) if args.score_path else _find_first(
        mode_root / "score", f"*_{args.category}_score.json"
    )

    result_rows = _load_jsonl(result_path)
    score_rows = _load_jsonl(score_path)
    if not score_rows:
        raise ValueError(f"Empty score file: {score_path}")

    result_ids = [str(row["id"]) for row in result_rows if "id" in row]
    explicit_valid_ids = {
        str(row["id"])
        for row in score_rows[1:]
        if row.get("id") is not None and row.get("valid") is True
    }
    invalid_ids = {
        str(row["id"])
        for row in score_rows[1:]
        if row.get("id") is not None and row.get("valid") is False
    }

    if explicit_valid_ids:
        success_ids = [test_id for test_id in result_ids if test_id in explicit_valid_ids]
    else:
        success_ids = [test_id for test_id in result_ids if test_id not in invalid_ids]

    output_path = Path(args.output_path) if args.output_path else (
        mode_root / "logs" / f"{args.mode}_correct_ids.txt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(success_ids) + ("\n" if success_ids else ""), encoding="utf-8")

    summary = {
        "run_root": str(Path(args.run_root)),
        "mode": args.mode,
        "category": args.category,
        "result_path": str(result_path),
        "score_path": str(score_path),
        "output_path": str(output_path),
        "result_count": len(result_ids),
        "failed_count": len(invalid_ids),
        "success_count": len(success_ids),
        "score_header": score_rows[0],
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--mode", default="history_full_closed_loop")
    parser.add_argument("--category", default="multi_turn_base")
    parser.add_argument("--result-path", default="")
    parser.add_argument("--score-path", default="")
    parser.add_argument("--output-path", default="")
    return parser.parse_args()


if __name__ == "__main__":
    export_success_ids(parse_args())
