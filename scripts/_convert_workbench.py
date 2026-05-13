"""Convert upstream WorkBench task definitions to our normalized JSON schema.

The upstream olly-styles/WorkBench repository ships 6 CSV files under
``data/processed/queries_and_answers/`` (one per domain):

  - analytics_queries_and_answers.csv
  - calendar_queries_and_answers.csv
  - customer_relationship_manager_queries_and_answers.csv
  - email_queries_and_answers.csv
  - multi_domain_queries_and_answers.csv
  - project_management_queries_and_answers.csv

Each row has columns: ``query, answer, base_template, chosen_template,
domains``. The ``answer`` column is a Python-literal string list of
function-call strings such as
``['analytics.create_plot.func(time_min="2023-11-21", ...)']``.

Output schema (matching the format consumed by ``Dataset.from_json``):
    {
      "dataset_id": "workbench",
      "instances": [
        {
          "task_id": "<domain>-<row_index>",
          "instructions": "<natural-language query>",
          "expected_actions": [<list of function-call strings>],
          "expected_answer": "<concatenation of function-call strings>",
          "tools": ["send_email", "search_emails", "create_event",
                    "search_events", "submit"],
          "metadata": {
            "source": "workbench",
            "domains": [...],
            "base_template": "...",
            "chosen_template": "...",
            "upstream_path": "data/processed/queries_and_answers/<file>:<row>"
          }
        },
        ...
      ]
    }
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

DOMAIN_FILES = [
    "analytics_queries_and_answers.csv",
    "calendar_queries_and_answers.csv",
    "customer_relationship_manager_queries_and_answers.csv",
    "email_queries_and_answers.csv",
    "multi_domain_queries_and_answers.csv",
    "project_management_queries_and_answers.csv",
]


def _parse_answer_list(raw: str) -> List[str]:
    """The upstream `answer` is a Python-literal list of strings."""
    if not raw:
        return []
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return [raw.strip()]
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    return [str(parsed)]


def _parse_domains(raw: str) -> List[str]:
    if not raw:
        return []
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return [d.strip() for d in raw.split(",") if d.strip()]
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    return [str(parsed)]


def _domain_from_filename(name: str) -> str:
    return name.replace("_queries_and_answers.csv", "")


def convert_domain_csv(csv_path: Path, domain: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    csv.field_size_limit(sys.maxsize)
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            query = (row.get("query") or "").strip()
            if not query:
                continue
            answer_calls = _parse_answer_list(row.get("answer", ""))
            domains = _parse_domains(row.get("domains", "")) or [domain]
            expected_answer = "\n".join(answer_calls) if answer_calls else None
            records.append(
                {
                    "task_id": f"{domain}-{i:04d}",
                    "instructions": query,
                    "expected_actions": [
                        {"call": call} for call in answer_calls
                    ],
                    "expected_answer": expected_answer,
                    "tools": [
                        "send_email",
                        "search_emails",
                        "create_event",
                        "search_events",
                        "submit",
                    ],
                    "metadata": {
                        "source": "workbench",
                        "domains": domains,
                        "base_template": (row.get("base_template") or "").strip()
                        or None,
                        "chosen_template": (row.get("chosen_template") or "").strip()
                        or None,
                        "upstream_path": f"data/processed/queries_and_answers/{csv_path.name}:{i}",
                    },
                }
            )
    return records


def stratified_subset(
    records: List[Dict[str, Any]], sample_size: int, seed: int
) -> List[Dict[str, Any]]:
    """Deterministic per-domain stratified sample.

    Groups records by domain (the first segment of ``task_id``), shuffles each
    group with ``random.Random(seed)``, and takes ``ceil/floor`` of
    ``sample_size / n_domains`` from each so the total is exactly
    ``sample_size``. Domains are processed in alphabetical order so the
    output is stable across machines.
    """
    by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        domain = rec["task_id"].split("-", 1)[0]
        by_domain[domain].append(rec)
    domains = sorted(by_domain.keys())
    if not domains:
        return []
    base = sample_size // len(domains)
    extra = sample_size - base * len(domains)
    rng = random.Random(seed)
    picked: List[Dict[str, Any]] = []
    for i, domain in enumerate(domains):
        n_take = base + (1 if i < extra else 0)
        pool = list(by_domain[domain])
        rng.shuffle(pool)
        picked.extend(pool[:n_take])
    return picked


def convert(
    upstream_dir: Path,
    out_json: Path,
    sample_size: Optional[int] = 100,
    seed: int = 42,
    full_out_json: Optional[Path] = None,
) -> int:
    qa_dir = upstream_dir / "data" / "processed" / "queries_and_answers"
    if not qa_dir.exists():
        raise FileNotFoundError(
            f"Expected {qa_dir} from upstream olly-styles/WorkBench. "
            "Has the upstream layout changed? Edit scripts/_convert_workbench.py."
        )
    records: List[Dict[str, Any]] = []
    for fname in DOMAIN_FILES:
        csv_path = qa_dir / fname
        if not csv_path.exists():
            print(f"[_convert_workbench] WARN: missing {csv_path}; skipping")
            continue
        records.extend(convert_domain_csv(csv_path, _domain_from_filename(fname)))
    if not records:
        raise RuntimeError(
            f"No usable records extracted under {qa_dir}. Inspect the CSV files."
        )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    if full_out_json is not None:
        full_out_json.parent.mkdir(parents=True, exist_ok=True)
        full_payload = {"dataset_id": "workbench", "instances": records}
        full_out_json.write_text(json.dumps(full_payload, indent=2))
        print(
            f"[_convert_workbench] wrote {len(records)} full upstream tasks -> {full_out_json}"
        )

    if sample_size is None or sample_size >= len(records):
        selected = records
    else:
        selected = stratified_subset(records, sample_size, seed)
    payload = {"dataset_id": "workbench", "instances": selected}
    out_json.write_text(json.dumps(payload, indent=2))
    return len(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--full-out",
        type=Path,
        default=None,
        help="Optional path to also write the full upstream JSON (no subsetting).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Stratified subset size (default 100, matches the paper).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for per-domain shuffle (default 42, matches the paper).",
    )
    args = parser.parse_args()
    n = convert(
        args.upstream_dir,
        args.out,
        sample_size=args.sample_size,
        seed=args.seed,
        full_out_json=args.full_out,
    )
    print(f"[_convert_workbench] wrote {n} tasks -> {args.out}")


if __name__ == "__main__":
    main()
