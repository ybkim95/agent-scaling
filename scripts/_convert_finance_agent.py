"""Convert upstream Finance-Agent's public test data to our normalized JSON schema.

The upstream vals-ai/finance-agent repository ships a public CSV at
``data/public.csv`` containing 188 questions, expected answers, question
types, expert time estimates, and grading rubrics. The full Vals platform
benchmark requires ``VALS_API_KEY`` access; the public CSV is what we wire
up for unrestricted reproduction here.

Output schema (matching the format consumed by ``Dataset.from_json``):
    {
      "dataset_id": "finance_agent",
      "instances": [
        {
          "task_id": "<string>",
          "instructions": "<natural-language question>",
          "expected_answer": "<ground-truth answer text>",
          "tools": ["python_repl", "submit"],
          "metadata": {
            "source": "finance-agent",
            "question_type": "<e.g. Market Analysis>",
            "expert_time_mins": <int or null>,
            "rubric": <list of grading criteria or null>,
            "upstream_path": "data/public.csv:<row_index>"
          }
        },
        ...
      ]
    }
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _safe_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _parse_rubric(raw: str) -> Optional[List[Dict[str, Any]]]:
    if not raw:
        return None
    try:
        loaded = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return loaded if isinstance(loaded, list) else None


def convert_public_csv(public_csv: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    csv.field_size_limit(sys.maxsize)
    with public_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            question = (row.get("Question") or "").strip()
            answer = (row.get("Answer") or "").strip()
            if not question:
                continue
            records.append(
                {
                    "task_id": f"fa-{i:04d}",
                    "instructions": question,
                    "expected_answer": answer or None,
                    "tools": ["python_repl", "submit"],
                    "metadata": {
                        "source": "finance-agent",
                        "question_type": (row.get("Question Type") or "").strip()
                        or None,
                        "expert_time_mins": _safe_int(row.get("Expert time (mins)")),
                        "rubric": _parse_rubric(row.get("Rubric", "")),
                        "upstream_path": f"data/public.csv:{i}",
                    },
                }
            )
    return records


def convert(upstream_dir: Path, out_json: Path) -> int:
    public_csv = upstream_dir / "data" / "public.csv"
    if not public_csv.exists():
        raise FileNotFoundError(
            f"Expected {public_csv} from upstream vals-ai/finance-agent. "
            "Has the upstream layout changed? Edit scripts/_convert_finance_agent.py."
        )
    records = convert_public_csv(public_csv)
    if not records:
        raise RuntimeError(
            f"No usable records extracted from {public_csv}. Inspect the CSV."
        )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {"dataset_id": "finance_agent", "instances": records}
    out_json.write_text(json.dumps(payload, indent=2))
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    n = convert(args.upstream_dir, args.out)
    print(f"[_convert_finance_agent] wrote {n} tasks -> {args.out}")


if __name__ == "__main__":
    main()
