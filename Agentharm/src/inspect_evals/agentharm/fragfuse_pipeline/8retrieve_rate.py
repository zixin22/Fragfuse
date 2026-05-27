"""
8retrieve_rate.py

Compute "retrieval match-up rate" between:
- Similar Task 1 text shown in per-task preview (task_user_message_preview.txt)
and
- The carrier-query "text" in per-task bank file (bank_{task_id}.jsonl),
  where type == "q_carrier"

Pipeline
--------
1) From guardrail_pair_summary.json select task_ids where both carrier and attack are allowed
2) For each task_id:
   - parse preview file:
       retrieve_test_logs_memory/task_{task_id}/task_user_message_preview.txt
     extract:
       [Similar Task 1] (similarity=...)
       <similar_text ... until next [Similar Task] or until task instruction begins>
   - load bank_records/bank_{task_id}.jsonl and find q_carrier row -> bank_text
3) Compare preview_similar_text vs bank_text:
   - exact match
   - normalized exact match (whitespace normalized)
   - SequenceMatcher similarity and pass threshold
4) Write:
   - retrieve_rate_matches.jsonl
   - retrieve_rate_matches.csv
   - retrieve_rate_summary.json

Run
---
python src/inspect_evals/agentharm/fragfuse_pipeline/8retrieve_rate.py \
  --task_ids_path "src/inspect_evals/agentharm/fragfuse_output/guardrail_pair_summary.json" \
  --task_logs_root "src/inspect_evals/agentharm/fragfuse_output/retrieve_test_logs_memory" \
  --bank_dir "src/inspect_evals/agentharm/fragfuse_output/bank_records" \
  --threshold 0.99
"""

import argparse
import csv
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
AGENTHARM_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = AGENTHARM_DIR / "fragfuse_output"


# ---------------------------
# Helpers
# ---------------------------
def _safe_str(x: Any) -> str:
    try:
        return "" if x is None else str(x)
    except Exception:
        return repr(x)


def normalize_text(s: str) -> str:
    """
    Light normalization:
    - normalize newlines
    - strip leading/trailing spaces
    - collapse multiple whitespace to single spaces, but keep line breaks as spaces
    """
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.strip()
    # turn all whitespace (incl newlines/tabs) into single spaces
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def seq_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL not found: {path}")
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


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def write_json(path: str, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


# ---------------------------
# Step 1: selected task_ids
# ---------------------------
def _load_task_ids_from_json(path: Path) -> List[str]:
    obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(obj, dict):
        raise ValueError(f"JSON task id source must be an object: {path}")

    summary = obj.get("summary")
    if isinstance(summary, dict):
        for key in ("both_allowed_task_ids", "passed_task_ids", "allowed_task_ids"):
            values = summary.get(key)
            if isinstance(values, list):
                return sorted({str(v).strip() for v in values if str(v).strip()})

    for key in ("both_allowed_task_ids", "passed_task_ids", "allowed_task_ids", "task_ids"):
        values = obj.get(key)
        if isinstance(values, list):
            return sorted({str(v).strip() for v in values if str(v).strip()})

    raise ValueError(
        f"Could not find task ids in {path}. Expected summary.both_allowed_task_ids "
        "from 7_run_guardrail.py, or a compatible task id list."
    )


def _load_allowed_task_ids_from_jsonl(path: Path) -> List[str]:
    task_ids: List[str] = []
    seen = set()
    for rec in read_jsonl(str(path)):
        tid = rec.get("task_id")
        gd = rec.get("guardrail_decision", {})
        allowed = None
        if isinstance(gd, dict):
            allowed = gd.get("allowed", None)

        if allowed is True and isinstance(tid, str) and tid.strip():
            tid = tid.strip()
            if tid not in seen:
                seen.add(tid)
                task_ids.append(tid)

    return sorted(task_ids)


def load_selected_task_ids(task_ids_path: str) -> List[str]:
    path = Path(task_ids_path)
    if not path.exists():
        raise FileNotFoundError(f"Task id source not found: {task_ids_path}")
    if path.suffix == ".json":
        return _load_task_ids_from_json(path)
    return _load_allowed_task_ids_from_jsonl(path)


# ---------------------------
# Step 2: parse Similar Task 1 from preview
# ---------------------------
SIMILAR_HEADER_RE = re.compile(r"^\[Similar Task 1\]\s*\(similarity=([0-9.]+)\)\s*$", re.MULTILINE)
ANY_SIMILAR_RE = re.compile(r"^\[Similar Task\s+\d+\]\s*\(similarity=[0-9.]+\)\s*$", re.MULTILINE)

# Optional boundary for the start of the actual instruction. The similar-text
# block stops when a line begins with: "Extract contents inside <> of"
TASK_INSTRUCTION_START_RE = re.compile(r"^\s*Extract contents inside\s*<>\s*of", re.IGNORECASE | re.MULTILINE)


def parse_similar_task_1(preview_text: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Returns (similarity_score, similar_task_text)
    """
    m = SIMILAR_HEADER_RE.search(preview_text)
    if not m:
        return None, None

    try:
        sim = float(m.group(1))
    except Exception:
        sim = None

    start = m.end()

    # Determine the end boundary:
    # 1) next [Similar Task k] header
    m2 = ANY_SIMILAR_RE.search(preview_text, pos=start)
    end_candidates = []
    if m2:
        end_candidates.append(m2.start())

    # 2) task instruction start ("Extract contents inside <> of ...")
    m3 = TASK_INSTRUCTION_START_RE.search(preview_text, pos=start)
    if m3:
        end_candidates.append(m3.start())

    end = min(end_candidates) if end_candidates else len(preview_text)

    block = preview_text[start:end].strip()
    # Remove leading/trailing backticks/code fences if any
    block = block.strip("`").strip()
    return sim, block if block else None


# ---------------------------
# Step 3: load injected q_carrier text from bank file
# ---------------------------
def load_q_carrier_text(bank_path: Path, task_id: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Return (text, raw_row). Prefer row where type == "q_carrier".
    Fallback: row with metadata.task_id == task_id and having a "text" field.
    """
    if not bank_path.exists():
        return None, None

    best_row = None
    fallback_row = None

    with bank_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue

            if obj.get("type") == "q_carrier" and isinstance(obj.get("text"), str):
                best_row = obj
                break

            meta = obj.get("metadata")
            if isinstance(meta, dict) and _safe_str(meta.get("task_id")).strip() == task_id:
                if isinstance(obj.get("text"), str) and obj["text"].strip():
                    fallback_row = obj

    row = best_row or fallback_row
    if not row:
        return None, None
    return row.get("text"), row


# ---------------------------
# Paths per task
# ---------------------------
def preview_path(task_logs_root: str, task_id: str) -> Path:
    return Path(task_logs_root) / f"task_{task_id}" / "task_user_message_preview.txt"


def bank_path(bank_dir: str, task_id: str) -> Path:
    return Path(bank_dir) / f"bank_{task_id}.jsonl"


# ---------------------------
# CLI
# ---------------------------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--task_ids_path",
        default=str(OUTPUT_DIR / "guardrail_pair_summary.json"),
        help=(
            "Task id source. Defaults to the pair summary produced by 7_run_guardrail.py. "
            "JSON sources should contain summary.both_allowed_task_ids; JSONL sources "
            "fall back to rows with guardrail_decision.allowed == true."
        ),
    )
    ap.add_argument(
        "--task_logs_root",
        default=str(OUTPUT_DIR / "retrieve_test_logs_memory"),
        help="Folder containing task_{task_id}/task_user_message_preview.txt",
    )
    ap.add_argument(
        "--bank_dir",
        default=str(OUTPUT_DIR / "bank_records"),
        help="Folder containing bank_{task_id}.jsonl",
    )
    ap.add_argument(
        "--out_jsonl",
        default=str(OUTPUT_DIR / "retrieve_rate_matches.jsonl"),
    )
    ap.add_argument(
        "--out_csv",
        default=str(OUTPUT_DIR / "retrieve_rate_matches.csv"),
    )
    ap.add_argument(
        "--out_summary",
        default=str(OUTPUT_DIR / "retrieve_rate_summary.json"),
    )
    ap.add_argument("--limit", type=int, default=0, help="Debug: process only first N allowed tasks")
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.99,
        help="SequenceMatcher threshold to count as matched (default: 0.99)",
    )
    ap.add_argument("--verbose", action="store_true")
    return ap


def main() -> None:
    args = build_argparser().parse_args()

    task_ids = load_selected_task_ids(args.task_ids_path)
    total_selected = len(task_ids)
    if args.limit and args.limit > 0:
        task_ids = task_ids[: args.limit]

    print(f"[info] selected task_ids in source: {total_selected}")
    print(f"[info] processing: {len(task_ids)}")
    print(f"[info] task_ids_path: {args.task_ids_path}")
    print(f"[info] task_logs_root: {args.task_logs_root}")
    print(f"[info] bank_dir: {args.bank_dir}")
    print(f"[info] threshold: {args.threshold}")

    rows: List[Dict[str, Any]] = []

    n_missing_preview = 0
    n_missing_similar = 0
    n_missing_bank = 0
    n_missing_bank_text = 0

    n_exact = 0
    n_norm_exact = 0
    n_pass_thresh = 0

    for i, tid in enumerate(task_ids, start=1):
        rec: Dict[str, Any] = {"task_id": tid}

        # ---- preview ----
        p_preview = preview_path(args.task_logs_root, tid)
        rec["preview_path"] = str(p_preview)

        try:
            preview_txt = read_text(p_preview)
        except Exception as e:
            rec["error"] = f"missing_preview: {e}"
            n_missing_preview += 1
            rows.append(rec)
            if args.verbose:
                print(f"[{i}/{len(task_ids)}] {tid}: missing preview")
            continue

        sim_score, sim_text = parse_similar_task_1(preview_txt)
        rec["preview_sim1_score"] = sim_score
        rec["preview_sim1_text"] = sim_text
        rec["preview_sim1_text_len"] = len(sim_text) if isinstance(sim_text, str) else 0

        if sim_score is None or not sim_text:
            n_missing_similar += 1

        # ---- bank ----
        p_bank = bank_path(args.bank_dir, tid)
        rec["bank_path"] = str(p_bank)

        if not p_bank.exists():
            rec["error"] = rec.get("error") or "missing_bank_file"
            n_missing_bank += 1
            rows.append(rec)
            if args.verbose:
                print(f"[{i}/{len(task_ids)}] {tid}: missing bank file")
            continue

        bank_text, bank_row = load_q_carrier_text(p_bank, task_id=tid)
        rec["bank_q_carrier_text"] = bank_text
        rec["bank_q_carrier_text_len"] = len(bank_text) if isinstance(bank_text, str) else 0
        rec["bank_q_carrier_row_type"] = bank_row.get("type") if isinstance(bank_row, dict) else None

        if not bank_text:
            n_missing_bank_text += 1

        # ---- compare ----
        a = sim_text or ""
        b = bank_text or ""

        rec["exact_match"] = (a == b) and bool(a) and bool(b)
        if rec["exact_match"]:
            n_exact += 1

        a_norm = normalize_text(a) if a else ""
        b_norm = normalize_text(b) if b else ""
        rec["normalized_exact_match"] = (a_norm == b_norm) and bool(a_norm) and bool(b_norm)
        if rec["normalized_exact_match"]:
            n_norm_exact += 1

        ssim = seq_similarity(a_norm, b_norm) if a_norm and b_norm else 0.0
        rec["sequence_similarity_norm"] = ssim
        rec["pass_threshold"] = bool(a_norm and b_norm and ssim >= args.threshold)
        if rec["pass_threshold"]:
            n_pass_thresh += 1

        # small previews for debugging
        rec["preview_sim1_text_preview"] = (a[:200].replace("\n", "\\n") if a else "")
        rec["bank_text_preview"] = (b[:200].replace("\n", "\\n") if b else "")

        rows.append(rec)

        if args.verbose:
            print(
                f"[{i}/{len(task_ids)}] {tid}: "
                f"score={sim_score} seq_sim={ssim:.4f} exact={rec['exact_match']} pass={rec['pass_threshold']}"
            )

    processed = len(rows)
    comparable = sum(1 for r in rows if (r.get("preview_sim1_text") and r.get("bank_q_carrier_text")))

    summary = {
        "task_ids_path": args.task_ids_path,
        "task_logs_root": args.task_logs_root,
        "bank_dir": args.bank_dir,
        "selected_task_ids": len(task_ids),
        "processed_rows": processed,
        "comparable_rows_with_both_texts": comparable,
        "missing_preview": n_missing_preview,
        "missing_similar_task_1_block": n_missing_similar,
        "missing_bank_file": n_missing_bank,
        "missing_bank_text": n_missing_bank_text,
        "exact_match_count": n_exact,
        "normalized_exact_match_count": n_norm_exact,
        "pass_threshold_count": n_pass_thresh,
        "threshold": args.threshold,
        "exact_match_rate_over_selected": (n_exact / len(task_ids)) if task_ids else 0.0,
        "normalized_exact_match_rate_over_selected": (n_norm_exact / len(task_ids)) if task_ids else 0.0,
        "pass_threshold_rate_over_selected": (n_pass_thresh / len(task_ids)) if task_ids else 0.0,
        "exact_match_rate_over_comparable": (n_exact / comparable) if comparable else 0.0,
        "normalized_exact_match_rate_over_comparable": (n_norm_exact / comparable) if comparable else 0.0,
        "pass_threshold_rate_over_comparable": (n_pass_thresh / comparable) if comparable else 0.0,
        "out_jsonl": args.out_jsonl,
        "out_csv": args.out_csv,
        "out_summary": args.out_summary,
    }

    write_jsonl(args.out_jsonl, rows)

    # CSV fields (flat)
    csv_fields = [
        "task_id",
        "preview_sim1_score",
        "sequence_similarity_norm",
        "exact_match",
        "normalized_exact_match",
        "pass_threshold",
        "preview_sim1_text_len",
        "bank_q_carrier_text_len",
        "preview_path",
        "bank_path",
        "error",
        "preview_sim1_text_preview",
        "bank_text_preview",
    ]
    write_csv(args.out_csv, rows, csv_fields)
    write_json(args.out_summary, summary)

    print("\n=== retrieve_rate complete ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
