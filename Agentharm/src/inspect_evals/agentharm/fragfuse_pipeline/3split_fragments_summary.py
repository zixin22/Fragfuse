#!/usr/bin/env python3
"""
split_sensitive_fragments_char_halves.py

Reads JSONL lines like:
  {"task_id": "...", "sensitive_fragments": ["fragment a", "fragment b", ...]}

For EACH sensitive fragment:
  - split into words
  - for each word take first half of characters: word[:len(word)//2]
  - join the fragments back into a single string

Writes JSONL where each line is exactly:
  {"task_id": "...", "split_sensitive_fragments": [ ... ]}

Default input:
  src/inspect_evals/agentharm/fragfuse_output/all_sensitive_fragments_union.jsonl

Default output:
  src/inspect_evals/agentharm/fragfuse_output/all_sensitive_fragments_union.split_charhalves.jsonl
"""

#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_WS = re.compile(r"\s+")
SCRIPT_DIR = Path(__file__).resolve().parent
AGENTHARM_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = AGENTHARM_DIR / "fragfuse_output"


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    outp = Path(path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def tokenize_words(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    return _WS.split(text)


def split_fragment_char_halves(fragment: str) -> Tuple[str, str]:
    """
    Per-word char split:
      word -> (word[:len(word)//2], word[len(word)//2:])
    Returns two strings (half1, half2), each joined by spaces.
    """
    words = tokenize_words(fragment)
    if not words:
        return ("", "")

    half1_parts: List[str] = []
    half2_parts: List[str] = []

    for w in words:
        w = w.strip()
        if not w:
            continue
        l = len(w) // 2
        half1_parts.append(w[:l] if l > 0 else "")
        half2_parts.append(w[l:] if l < len(w) else "")

    half1 = " ".join([x for x in half1_parts if x]).strip()
    half2 = " ".join([x for x in half2_parts if x]).strip()
    return (half1, half2)


def get_task_id(obj: Dict[str, Any]) -> Optional[str]:
    tid = obj.get("task_id")
    if isinstance(tid, str) and tid.strip():
        return tid.strip()
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_path",
        default=str(OUTPUT_DIR / "all_sensitive_fragments_union.jsonl"),
        help="Input JSONL path containing task_id + sensitive_fragments.",
    )
    ap.add_argument(
        "--out_path",
        default=str(OUTPUT_DIR / "all_sensitive_fragments_union.split_charhalves.jsonl"),
        help="Output JSONL path (task_id + flattened split_sensitive_fragments).",
    )
    ap.add_argument(
        "--keep_empty",
        action="store_true",
        help="If set, keep empty halves as empty strings; otherwise skip empties (default).",
    )
    args = ap.parse_args()

    out_rows: List[Dict[str, Any]] = []
    kept = 0
    skipped = 0

    for obj in read_jsonl(args.in_path):
        tid = get_task_id(obj)
        if not tid:
            skipped += 1
            continue

        fragments = obj.get("sensitive_fragments", [])
        if not isinstance(fragments, list):
            fragments = []

        flattened: List[str] = []
        for t in fragments:
            if not isinstance(t, str):
                continue
            t = t.strip()
            if not t:
                continue

            h1, h2 = split_fragment_char_halves(t)

            if args.keep_empty:
                flattened.extend([h1, h2])
            else:
                if h1:
                    flattened.append(h1)
                if h2:
                    flattened.append(h2)

        out_rows.append(
            {
                "task_id": tid,
                "split_sensitive_fragments": flattened,
            }
        )
        kept += 1

    write_jsonl(args.out_path, out_rows)

    print("====================================")
    print("Split sensitive fragments (char-mode halves, flattened)")
    print(f"Input:  {args.in_path}")
    print(f"Output: {args.out_path}")
    print(f"Kept records:    {kept}")
    print(f"Skipped records: {skipped}")
    print("====================================")


if __name__ == "__main__":
    main()
