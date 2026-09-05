from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--details-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output, "w", encoding="utf-8") as f:
        for row in _read_jsonl(Path(args.details_path)):
            for segment in row.get("repair_segments") or []:
                if not isinstance(segment, dict):
                    continue
                f.write(json.dumps(segment, ensure_ascii=False) + "\n")
                count += 1
    print(json.dumps({"segments": count, "output_path": str(output)}))


if __name__ == "__main__":
    main()
