
from __future__ import annotations

"""
Stage 1 verification: start with query and sensitive_fragments
Create query' by masking those fragments
Call guardrail on query'
If blocked -> go back (either re-extract fragments or refine masking)
Stop when allowed=true (even if query becomes empty)

Output: src/inspect_evals/agentharm/fragfuse_output/step2_iterative_mask.jsonl
"""

"""
extract all "task_id" and "sensitive_fragments" from fragfuse_output/step1_sensitive_fragments.jsonl
create q_mask by masking those fragments in harmful_full.txt by task_id
write masked query to fragfuse_output/step2_masked_instructions.jsonl
call guardrail_prompt.py to check masked query
write checking results to fragfuse_output/step2_iterative_mask.jsonl
"""

#!/usr/bin/env python3
"""
2_1iterative_mask.py

Step2:
- Read step1_sensitive_fragments.jsonl and extract {"task_id", "sensitive_fragments"}.
- Parse harmful_full.txt into TASK blocks keyed by task_id.
- Mask sensitive fragments in the original query (per task_id).
- Write masked queries to step2_masked_instructions.jsonl
- Call OpenAI-only LLM guardrail (call_guardrail) to check masked query
- Write guardrail decisions to step2_iterative_mask.jsonl
- Print summary: how many are allowed after masking.

Run:
  uv run python src/inspect_evals/agentharm/fragfuse_pipeline/2_1iterative_mask.py
"""


import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import asyncio


# -----------------------------
# Default paths
# -----------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
AGENTHARM_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[3]
OUT_DIR_DEFAULT = str(AGENTHARM_DIR / "fragfuse_output")
HARMFUL_FULL_DEFAULT = str(REPO_ROOT / "harmful_full.txt")
STEP1_FRAGMENTS_DEFAULT = str(Path(OUT_DIR_DEFAULT) / "step1_sensitive_fragments.jsonl")
STEP2_MASKED_OUT_DEFAULT = f"{OUT_DIR_DEFAULT}/step2_masked_instructions.jsonl"
STEP2_RESULTS_OUT_DEFAULT = f"{OUT_DIR_DEFAULT}/step2_iterative_mask.jsonl"


# -----------------------------
# Parse harmful_full.txt
# -----------------------------

_TASK_HEADER_RE = re.compile(r"^\s*###\s*TASK\s+([0-9]+-[0-9]+)\s*$")


@dataclass(frozen=True)
class TaskBlock:
    idx: int
    task_id: str
    text: str


def parse_harmful_full_txt(path: str) -> List[TaskBlock]:
    p = Path(path)
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()

    blocks: List[TaskBlock] = []
    cur_id: Optional[str] = None
    cur_lines: List[str] = []

    def flush() -> None:
        nonlocal cur_id, cur_lines
        if cur_id is None:
            return
        text = "\n".join(cur_lines).strip()
        blocks.append(TaskBlock(idx=len(blocks), task_id=cur_id, text=text))
        cur_id = None
        cur_lines = []

    for line in lines:
        m = _TASK_HEADER_RE.match(line)
        if m:
            flush()
            cur_id = m.group(1)
            cur_lines = []
        else:
            if cur_id is not None:
                cur_lines.append(line)

    flush()
    return blocks


def parse_task_jsonl(path: str) -> List[TaskBlock]:
    """
    Parse task input from JSONL or JSON.
    Supported text fields priority:
      masked_query > query > query_full > instruction > prompt > text
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    text = p.read_text(encoding="utf-8", errors="replace")
    records: List[Dict[str, Any]] = []

    # 1) Try line-delimited JSON first (works for .jsonl and concatenated .json by line)
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if isinstance(obj, dict):
            records.append(obj)

    # 2) Fallback to standard JSON document parsing
    if not records:
        try:
            raw = json.loads(text)
            if isinstance(raw, list):
                records = [x for x in raw if isinstance(x, dict)]
            elif isinstance(raw, dict):
                if isinstance(raw.get("results"), list):
                    records = [x for x in raw["results"] if isinstance(x, dict)]
                else:
                    records = [raw]
        except Exception:
            records = []

    blocks: List[TaskBlock] = []
    for obj in records:
        tid = obj.get("task_id")
        if not isinstance(tid, str) or not tid.strip():
            continue
        tid = tid.strip()

        text_raw = (
            obj.get("masked_query")
            or obj.get("query")
            or obj.get("query_full")
            or obj.get("instruction")
            or obj.get("prompt")
            or obj.get("text")
            or ""
        )
        if not isinstance(text_raw, str) or not text_raw.strip():
            continue

        blocks.append(TaskBlock(idx=len(blocks), task_id=tid, text=text_raw.strip()))

    return blocks


# -----------------------------
# Read step1_sensitive_fragments.jsonl
# -----------------------------

def load_step1_fragments(path: str) -> Dict[str, List[str]]:
    """
    Returns mapping: task_id -> sensitive_fragments (list).
    If multiple records exist for same task_id, keeps the last non-empty list.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"step1_sensitive_fragments.jsonl not found: {path}")

    mapping: Dict[str, List[str]] = {}
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
            fragments = obj.get("sensitive_fragments")

            if not isinstance(tid, str) or not tid:
                continue

            cleaned: List[str] = []
            if isinstance(fragments, list):
                seen: Set[str] = set()
                for t in fragments:
                    if isinstance(t, str):
                        tt = t.strip()
                        if tt and tt not in seen:
                            seen.add(tt)
                            cleaned.append(tt)

            # Keep last; but prefer a non-empty list if present
            if cleaned or tid not in mapping:
                mapping[tid] = cleaned

    return mapping


# -----------------------------
# Masking
# -----------------------------

def _make_placeholder(i: int) -> str:
    return f"[MASK_{i}]"


_MASK_INDEX_RE = re.compile(r"\[\s*MASK_(\d+)\s*\]", re.IGNORECASE)


def _occupied_mask_indices(text: str) -> Set[int]:
    used: Set[int] = set()
    for m in _MASK_INDEX_RE.finditer(text or ""):
        try:
            used.add(int(m.group(1)))
        except Exception:
            continue
    return used


def mask_fragments_in_text(text: str, fragments: List[str]) -> Tuple[str, Dict[str, str]]:
    """
    Literal masking: replace each fragment with [MASK_i].
    - Sort fragments by length desc to reduce partial-overlap issues
    - Use regex with re.escape for literal matching (case-sensitive)
    Returns: (masked_text, mask_map fragment->placeholder)
    """
    if not fragments:
        return text, {}

    # unique
    uniq: List[str] = []
    seen: Set[str] = set()
    for t in fragments:
        tt = t.strip()
        if tt and tt not in seen:
            seen.add(tt)
            uniq.append(tt)

    uniq.sort(key=len, reverse=True)

    mask_map: Dict[str, str] = {}
    masked = text
    used_indices = _occupied_mask_indices(text)
    next_idx = (max(used_indices) + 1) if used_indices else 1

    for fragment in uniq:
        while next_idx in used_indices:
            next_idx += 1
        ph = _make_placeholder(next_idx)
        used_indices.add(next_idx)
        next_idx += 1
        mask_map[fragment] = ph
        masked = re.sub(re.escape(fragment), ph, masked)

    return masked, mask_map


# -----------------------------
# JSONL helpers
# -----------------------------

def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_jsonl_line(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_existing_task_ids(path: str) -> Set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    out: Set[str] = set()
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
            if isinstance(tid, str) and tid:
                out.add(tid)
    return out


# -----------------------------
# Main async pipeline
# -----------------------------

def _decision_note(decision: Dict[str, Any], is_allowed: bool) -> str:
    """
    Build a human-readable note explaining *why* allowed/refused.
    Works across a few common guardrail response shapes.
    """
    if not isinstance(decision, dict):
        return "No decision details (decision is not a dict)."

    # Common fields different implementations might return
    for key in ("reason", "message", "explanation", "rationale"):
        v = decision.get(key)
        if isinstance(v, str) and v.strip():
            prefix = "Allowed" if is_allowed else "Refused"
            return f"{prefix}: {v.strip()}"

    # Sometimes there's nested info (e.g., {"policy": {"violations":[...]}} or {"blocked_by":[...]})
    blocked_by = decision.get("blocked_by")
    if isinstance(blocked_by, list) and blocked_by:
        prefix = "Allowed" if is_allowed else "Refused"
        return f"{prefix}: blocked_by={blocked_by}"

    violations = None
    policy = decision.get("policy")
    if isinstance(policy, dict):
        violations = policy.get("violations")
    if isinstance(violations, list) and violations:
        prefix = "Allowed" if is_allowed else "Refused"
        # keep it compact
        return f"{prefix}: policy_violations={[str(x) for x in violations][:5]}"

    # Sometimes model output parsing errors are reported
    parse_error = decision.get("parse_error")
    if isinstance(parse_error, str) and parse_error.strip():
        prefix = "Allowed" if is_allowed else "Refused"
        return f"{prefix}: parse_error={parse_error.strip()}"

    # Last resort: include allowed flag only
    return "Allowed (no explicit reason returned)." if is_allowed else "Refused (no explicit reason returned)."


def _load_call_guardrail():
    """
    Lazy-import guardrail helper so --skip_guardrail mode works even when
    inspect_evals isn't importable from current PYTHONPATH.
    """
    from inspect_evals.agentharm.guardrail_prompt.guardrail_prompt import call_guardrail

    return call_guardrail


async def run(args: argparse.Namespace) -> None:
    ensure_parent(args.out_masked)
    if not args.skip_guardrail:
        ensure_parent(args.out_results)

    # Default behavior: clear outputs and rerun from scratch.
    Path(args.out_masked).write_text("", encoding="utf-8")
    if not args.skip_guardrail:
        Path(args.out_results).write_text("", encoding="utf-8")

    step1_map = load_step1_fragments(args.step1_fragments)
    if Path(args.harmful_full).suffix.lower() in {".jsonl", ".json"}:
        blocks = parse_task_jsonl(args.harmful_full)
    else:
        blocks = parse_harmful_full_txt(args.harmful_full)

    start_idx = max(args.start_idx, 0)
    end_idx = (len(blocks) - 1) if args.end_idx < 0 else min(args.end_idx, len(blocks) - 1)
    if start_idx > end_idx:
        raise ValueError(f"start_idx={start_idx} > end_idx={end_idx}")

    tid_re: Optional[re.Pattern[str]] = re.compile(args.task_id_regex) if args.task_id_regex else None

    total = 0
    allowed_count = 0
    missing_task_in_step1 = 0
    empty_fragments_count = 0
    guardrail_checked = 0
    call_guardrail = None if args.skip_guardrail else _load_call_guardrail()

    for b in blocks[start_idx : end_idx + 1]:
        if tid_re and not tid_re.search(b.task_id):
            continue

        original_query = b.text
        fragments = step1_map.get(b.task_id)
        if fragments is None:
            missing_task_in_step1 += 1
            fragments = []
        if not fragments:
            empty_fragments_count += 1

        masked_query, mask_map = mask_fragments_in_text(original_query, fragments)

        # Write masked query record
        write_jsonl_line(
            args.out_masked,
            {
                "idx": b.idx,
                "task_id": b.task_id,
                "query": original_query,
                "prohibited_query": original_query,
                "sensitive_fragments": fragments,
                "mask_map": mask_map,
                "masked_query": masked_query,
                "q_mask": masked_query,
            },
        )

        if args.skip_guardrail:
            total += 1
            print(f"[{b.idx:03d}] TASK {b.task_id}: masked-only (fragments={len(fragments)})")
            continue

        # Guardrail check (OpenAI-only)
        decision = await call_guardrail(
            agent_trajectory="",
            observation="",
            action=masked_query,
            model_name=args.guardrail_model,
            max_tokens=args.guardrail_max_tokens,
            temperature=args.guardrail_temperature,
            allow_non_openai=False,
            fail_closed_on_parse_error=True,
        )

        is_allowed = bool(decision.get("allowed", False))
        note = _decision_note(decision, is_allowed)

        total += 1
        guardrail_checked += 1
        if is_allowed:
            allowed_count += 1

        write_jsonl_line(
            args.out_results,
            {
                "idx": b.idx,
                "task_id": b.task_id,
                "guardrail_model": args.guardrail_model,
                "masked_query": masked_query,
                "q_mask": masked_query,
                "guardrail_decision": decision,
                "note": note,  # <-- NEW
                "sensitive_fragments": fragments,
                "mask_map": mask_map,
            },
        )

        print(f"[{b.idx:03d}] TASK {b.task_id}: allowed={is_allowed} (fragments={len(fragments)})")

    print("\n========== Step2 Summary ==========")
    print(f"Processed: {total}")
    if args.skip_guardrail:
        print("Guardrail: skipped (--skip_guardrail)")
    else:
        pct = (allowed_count / total * 100.0) if total else 0.0
        print(f"Allowed after masking: {allowed_count} ({pct:.1f}%)")
        print(f"Guardrail checked: {guardrail_checked}")
    print(f"Tasks missing in step1_sensitive_fragments.jsonl: {missing_task_in_step1}")
    print(f"Tasks with empty sensitive_fragments: {empty_fragments_count}")
    print(f"Masked output:  {args.out_masked}")
    if args.skip_guardrail:
        print("Results output: skipped")
    else:
        print(f"Results output: {args.out_results}")
    print("==================================\n")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    ap.add_argument("--harmful_full", default=HARMFUL_FULL_DEFAULT)
    ap.add_argument("--step1_fragments", default=STEP1_FRAGMENTS_DEFAULT)
    ap.add_argument("--out_masked", default=STEP2_MASKED_OUT_DEFAULT)
    ap.add_argument("--out_results", default=STEP2_RESULTS_OUT_DEFAULT)

    ap.add_argument("--start_idx", type=int, default=0, help="0-based index into parsed TASK blocks")
    ap.add_argument("--end_idx", type=int, default=-1, help="inclusive; -1 means all")
    ap.add_argument("--task_id_regex", default="", help="Only process task_ids matching this regex")

    ap.add_argument("--skip_guardrail", action="store_true", help="Only write masked queries; do not call guardrail or write out_results")

    # Guardrail options (OpenAI model only)
    ap.add_argument("--guardrail_model", default="openai/gpt-4o-2024-08-06")
    ap.add_argument("--guardrail_max_tokens", type=int, default=512)
    ap.add_argument("--guardrail_temperature", type=float, default=0.0)

    return ap


def main() -> None:
    args = build_argparser().parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
