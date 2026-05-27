#!/usr/bin/env python3
from __future__ import annotations
"""
Summarize (union) sensitive_fragments across multiple masking records,
removing anything that contains "[MASK_" (case-insensitive).

Run:
  uv run python src/inspect_evals/agentharm/fragfuse_pipeline/3sensitive_fragments_summary.py

"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Set

SCRIPT_DIR = Path(__file__).resolve().parent
AGENTHARM_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = AGENTHARM_DIR / "fragfuse_output"

DEFAULT_FILES = [str(OUTPUT_DIR / "step2_masked_instructions.jsonl")]
DEFAULT_OUT = str(OUTPUT_DIR / "all_sensitive_fragments_union.jsonl")


def _tid_key(t: str):
    try:
        a, b = t.split("-", 1)
        return (int(a), int(b))
    except Exception:
        return (10**9, 10**9)


_MASK_PLAIN_RE = re.compile(r"^\s*MASK_\d+\s*$", re.IGNORECASE)

def _is_mask_fragment(s: str) -> bool:
    ss = s.strip()
    # remove anything containing "[MASK_" (case-insensitive)
    if "[MASK_" in ss.upper():
        return True
    # also remove plain tokens like "MASK_2"
    if _MASK_PLAIN_RE.match(ss):
        return True
    return False

def _clean_fragments(x) -> List[str]:
    if not isinstance(x, list):
        return []
    out: List[str] = []
    seen: Set[str] = set()
    for t in x:
        if not isinstance(t, str):
            continue
        tt = t.strip()
        if not tt:
            continue
        if _is_mask_fragment(tt):
            continue
        if tt in seen:
            continue
        seen.add(tt)
        out.append(tt)
    return out




def read_union(files: List[str]) -> Dict[str, List[str]]:
    union_map: Dict[str, List[str]] = {}
    seen_map: Dict[str, Set[str]] = {}

    for fp in files:
        p = Path(fp)
        if not p.exists():
            print(f"[WARN] missing file, skipped: {fp}")
            continue

        with p.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue

                tid = obj.get("task_id")
                if not isinstance(tid, str) or not tid.strip():
                    continue
                tid = tid.strip()

                fragments = _clean_fragments(obj.get("sensitive_fragments", []))

                if tid not in union_map:
                    union_map[tid] = []
                    seen_map[tid] = set()

                for fragment in fragments:
                    if fragment not in seen_map[tid]:
                        seen_map[tid].add(fragment)
                        union_map[tid].append(fragment)

    return union_map


def write_jsonl(out_path: str, union_map: Dict[str, List[str]]) -> None:
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    # Default behavior: overwrite output.
    outp.write_text("", encoding="utf-8")

    tids = sorted(union_map.keys(), key=lambda x: (_tid_key(x), x))
    with outp.open("a", encoding="utf-8") as f:
        for tid in tids:
            fragments = union_map.get(tid, [])
            rec = {"task_id": tid, "sensitive_fragments": fragments}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", default=DEFAULT_FILES)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    union_map = read_union(args.inputs)

    num_tasks = len(union_map)
    total_fragments = sum(len(v) for v in union_map.values())
    max_fragments = max((len(v) for v in union_map.values()), default=0)

    write_jsonl(args.out, union_map)

    print("\n========== Sensitive Fragments Union Summary ==========")
    print(f"Tasks: {num_tasks}")
    print(f"Total unique fragments (sum over tasks): {total_fragments}")
    print(f"Max fragments in a single task: {max_fragments}")
    print(f"Output: {args.out}")
    print("==================================================\n")


if __name__ == "__main__":
    main()
