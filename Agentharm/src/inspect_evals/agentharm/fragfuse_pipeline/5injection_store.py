#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
5injection_store.py

Read carrier-query records, extract carrier_query, and append it into
per-task memory bank JSONL files. By default this writes one bank for every
task that has a carrier_query; use --only_allowed to reproduce the older
behavior of keeping only guardrail_decision.allowed == True records.

Default:
  in  = src/inspect_evals/agentharm/fragfuse_output/carrier_query_results.jsonl
  base = src/inspect_evals/agentharm/fragfuse_output/base_bank.jsonl
  bank dir = src/inspect_evals/agentharm/fragfuse_output/bank_records

Output schema (one JSON per line):
{
  "type": "q_carrier",
  "text": "<...carrier_query...>",
  "carrier_query": "<...carrier_query...>",
  "metadata": {
      "task_id": "...",
      "host_id": "...",
      "guardrail_model": "...",
      "source_path": "...",
      "source_line": 123
  }
}

Each output file contains (base_bank lines + 1 q_carrier line).
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
AGENTHARM_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = AGENTHARM_DIR / "fragfuse_output"


# ---------------------------
# JSONL helpers
# ---------------------------
def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input JSONL not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                obj["_source_line"] = line_no
                yield obj


def read_jsonl_raw_lines(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Base bank JSONL not found: {path}")
    lines: List[str] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                lines.append(s)
    return lines


def write_jsonl_lines(path: str, lines: List[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Default behavior: overwrite output.
    with p.open("w", encoding="utf-8") as f:
        for s in lines:
            f.write(s + "\n")


# ---------------------------
# Build the appended carrier row (the +1)
# ---------------------------
def to_carrier_memory_row(
    rec: Dict[str, Any],
    source_path: str,
    *,
    only_allowed: bool,
) -> Optional[Dict[str, Any]]:
    gd = rec.get("guardrail_decision", {})
    allowed = None
    if isinstance(gd, dict):
        allowed = gd.get("allowed", None)
    if only_allowed and allowed is not True:
        return None

    carrier = rec.get("carrier_query", rec.get("q_carrier", None))
    if not isinstance(carrier, str) or not carrier.strip():
        return None

    return {
        "type": "q_carrier",
        "text": carrier.strip(),
        "carrier_query": carrier.strip(),
        "q_carrier": carrier.strip(),
        "metadata": {
            "task_id": rec.get("task_id"),
            "host_id": rec.get("host_id"),
            "guardrail_model": rec.get("guardrail_model"),
            "guardrail_allowed": allowed,
            "source_path": source_path,
            "source_line": rec.get("_source_line"),
        },
    }


# ---------------------------
# CLI
# ---------------------------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_path",
        default=str(OUTPUT_DIR / "carrier_query_results.jsonl"),
        help="Input JSONL containing carrier_query + guardrail_decision",
    )
    ap.add_argument(
        "--base_bank",
        default=str(OUTPUT_DIR / "base_bank.jsonl"),
        help="Base bank JSONL with exactly 32 lines",
    )
    ap.add_argument(
        "--out_dir",
        default=str(OUTPUT_DIR / "bank_records"),
        help="Output directory for per-task bank files",
    )
    ap.add_argument(
        "--expected_base_lines",
        type=int,
        default=32,
        help="Expected number of lines in base bank (default: 32)",
    )
    ap.add_argument(
        "--strict_unique_task",
        action="store_true",
        help="If a task_id appears multiple times, raise error (recommended)",
    )
    ap.add_argument(
        "--only_allowed",
        action="store_true",
        help="Only write banks for records where guardrail_decision.allowed == True (legacy behavior).",
    )
    return ap


def main() -> None:
    args = build_argparser().parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load base bank (as raw JSONL lines) and verify it's exactly 32 lines
    base_lines = read_jsonl_raw_lines(args.base_bank)
    if len(base_lines) != args.expected_base_lines:
        raise RuntimeError(
            f"Base bank line count mismatch: got {len(base_lines)}, expected {args.expected_base_lines}\n"
            f"base_bank={args.base_bank}"
        )

    # 2) Collect carrier queries, one per task_id (or allow duplicates if not strict)
    rows_by_task: Dict[str, Dict[str, Any]] = {}
    dup_task_ids: Dict[str, int] = {}

    total = 0
    included = 0
    allowed = 0

    for rec in read_jsonl(args.in_path):
        total += 1
        gd = rec.get("guardrail_decision", {})
        if isinstance(gd, dict) and gd.get("allowed") is True:
            allowed += 1

        row = to_carrier_memory_row(
            rec,
            source_path=args.in_path,
            only_allowed=args.only_allowed,
        )
        if row is None:
            continue
        included += 1

        task_id = row.get("metadata", {}).get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            # skip weird cases
            continue
        tid = task_id.strip()

        if tid in rows_by_task:
            dup_task_ids[tid] = dup_task_ids.get(tid, 1) + 1
            if args.strict_unique_task:
                raise RuntimeError(f"Duplicate carrier query for task_id={tid}")
            # non-strict: keep the first one, ignore later ones
            continue

        rows_by_task[tid] = row

    # 3) Write one file per task_id: 32 base + 1 carrier query
    written = 0
    for tid, inj_row in sorted(rows_by_task.items(), key=lambda x: x[0]):
        out_path = out_dir / f"bank_{tid}.jsonl"

        # append carrier query as the last line
        appended_lines = list(base_lines)
        appended_lines.append(json.dumps(inj_row, ensure_ascii=False))

        # sanity: 32 + 1
        if len(appended_lines) != args.expected_base_lines + 1:
            raise RuntimeError(f"Unexpected output line count for task_id={tid}: {len(appended_lines)}")

        write_jsonl_lines(str(out_path), appended_lines)
        written += 1

    print("=== per-task bank build summary ===")
    print(f"input:      {args.in_path}")
    print(f"base_bank:  {args.base_bank} (lines={len(base_lines)})")
    print(f"out_dir:    {args.out_dir}")
    print(f"read total records: {total}")
    print(f"allowed carrier query records: {allowed}")
    print(f"included carrier query records: {included}")
    print(f"only_allowed: {args.only_allowed}")
    print(f"unique task_ids written: {written}")
    if dup_task_ids:
        # show a few examples only
        ex = list(dup_task_ids.items())[:10]
        print(f"duplicate allowed task_ids (first 10): {ex}")
        print("Tip: rerun with --strict_unique_task to fail fast on duplicates.")
    print("Done.")


if __name__ == "__main__":
    main()
