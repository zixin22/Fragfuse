#!/usr/bin/env python3
"""
Goal
- Write harmful_full.txt with RAW prompts (no segmentation / no fragment markers inside the prompt text).
- Write harmful_fragments.txt with task-by-task FRAGMENT 1..4, produced by:
  (A) segment by meaning (steps / hints / section markers / bullets / blank-line blocks),
      but DO NOT split if the boundary would cut through a protected span.
  (B) chunk into exactly 4 chunks using a greedy *length-aware* boundary picker
      (NER-span–aware; prefers higher-quality boundaries like segment ends / blank lines / sentence ends).

Protected spans
- spaCy NER spans (doc.ents)
- plus a few regex spans (URLs, emails, handles, paths, backticks, quotes, CamelCase)

Notes
- Requires: pip install spacy && python -m spacy download en_core_web_sm
"""

import argparse
import json
import os
import re
import sys
from typing import List, Tuple, Optional, Dict

# -----------------------------
# Regex protected spans (supplement spaCy)
# -----------------------------
URL_RE = re.compile(r"\bhttps?://[^\s)>\]]+|\bwww\.[^\s)>\]]+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
HANDLE_RE = re.compile(r"@[A-Za-z0-9_]{1,30}\b")
PATH_RE = re.compile(r"(?<!\w)(?:/[^ \n\t\r\f\v:;,'\"<>]+)+")
BACKTICK_RE = re.compile(r"`[^`]{1,500}`")
QUOTED_RE = re.compile(r"(['\"])(?:(?=(\\?))\2.){1,500}?\1")
CAMEL_RE = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z0-9]+){1,}\b")

PROTECTED_REGEXES = [URL_RE, EMAIL_RE, HANDLE_RE, PATH_RE, BACKTICK_RE, QUOTED_RE, CAMEL_RE]

# -----------------------------
# Meaning-boundary markers
# -----------------------------
SECTION_HDR_RE = re.compile(r"^\s*(hint|note|important)\s*:\s*", re.IGNORECASE)
STEP_HDR_RE = re.compile(r"^\s*(step\s*\d+)\s*[:.\-]\s*", re.IGNORECASE)
# Discourse step markers (start-of-line)
DISCOURSE_RE = re.compile(
    r"^\s*(first|then|after that|finally|once|before that)\b[,:]?\s*",
    re.IGNORECASE,
)
NUM_BULLET_RE = re.compile(r"^\s*\d+\s*[\.\)]\s+")
DASH_BULLET_RE = re.compile(r"^\s*[-*•]\s+")


def normalize_newlines(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


def read_input_text(in_path: Optional[str]) -> str:
    if in_path:
        with open(in_path, "r", encoding="utf-8") as f:
            return f.read()
    return sys.stdin.read()


def merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not spans:
        return []
    spans.sort()
    merged: List[Tuple[int, int]] = []
    cur_s, cur_e = spans[0]
    for s, e in spans[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def boundary_inside_span(pos: int, spans: List[Tuple[int, int]]) -> bool:
    # cutting exactly at span boundaries is OK; strictly inside is not
    for s, e in spans:
        if s < pos < e:
            return True
    return False


def collect_protected_spans_spacy(text: str, nlp) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []

    doc = nlp(text)
    for ent in doc.ents:
        spans.append((ent.start_char, ent.end_char))

    for rx in PROTECTED_REGEXES:
        for m in rx.finditer(text):
            spans.append((m.start(), m.end()))

    return merge_spans(spans)


def segment_by_meaning_ner_safe(text: str, protected_spans: List[Tuple[int, int]]) -> Tuple[List[str], List[int]]:
    """
    Returns:
      segments: list of segment strings (verbatim slices by line-accumulation)
      seg_end_positions: char positions in ORIGINAL text where each segment ends (boundary AFTER segment)
    """
    text = normalize_newlines(text)
    raw = text.strip("\n")
    if not raw.strip():
        return [], []

    lines = raw.split("\n")

    segments: List[str] = []
    seg_end_positions: List[int] = []
    buf_lines: List[str] = []

    offset = 0  # char offset at start of current line, in `raw`

    def flush(end_pos: int):
        nonlocal buf_lines
        if not buf_lines:
            return
        seg = "\n".join(buf_lines).strip("\n")
        if seg.strip():
            segments.append(seg)
            seg_end_positions.append(end_pos)
        buf_lines = []

    for i, line in enumerate(lines):
        boundary_before_line = offset
        is_blank = (line.strip() == "")

        starts_new = (
            SECTION_HDR_RE.match(line)
            or STEP_HDR_RE.match(line)
            or DISCOURSE_RE.match(line)
            or NUM_BULLET_RE.match(line)
            or DASH_BULLET_RE.match(line)
        )

        if starts_new and buf_lines:
            # try split BEFORE this line
            if not boundary_inside_span(boundary_before_line, protected_spans):
                flush(boundary_before_line)

        if is_blank:
            # blank line ends a block; boundary AFTER this line
            boundary_after_line = offset + len(line) + 1  # + '\n'
            flush(boundary_after_line)
        else:
            buf_lines.append(line)

        offset += len(line) + 1  # include '\n'

    # final flush at end-of-text (offset currently points past last '\n')
    flush(len(raw))

    return segments, seg_end_positions


def collect_candidate_boundaries(text: str, nlp, protected_spans: List[Tuple[int, int]], seg_end_positions: List[int]) -> Dict[int, str]:
    """
    boundary -> tier label ("A"/"B"/"C")

    Tier A: segment ends, double-newline boundaries, section headers beginnings (captured via seg ends), etc.
    Tier B: sentence ends (spaCy) and single newline boundaries
    Tier C: whitespace boundaries
    """
    cand: Dict[int, str] = {}

    n = len(text)

    # Tier A: segment boundaries
    for b in seg_end_positions:
        if 0 < b < n and not boundary_inside_span(b, protected_spans):
            cand[b] = "A"

    # Tier A: double-newline boundaries
    idx = 0
    while True:
        j = text.find("\n\n", idx)
        if j == -1:
            break
        b = j + 2
        if 0 < b < n and not boundary_inside_span(b, protected_spans):
            cand[b] = "A"
        idx = j + 2

    # Tier B: sentence ends (spaCy)
    doc = nlp(text)
    for sent in doc.sents:
        b = sent.end_char
        if 0 < b < n and not boundary_inside_span(b, protected_spans):
            cand.setdefault(b, "B")

    # Tier B: newline boundaries
    for m in re.finditer(r"\n", text):
        b = m.start() + 1
        if 0 < b < n and not boundary_inside_span(b, protected_spans):
            cand.setdefault(b, "B")

    # Tier C: whitespace boundaries
    for m in re.finditer(r"\s+", text):
        b = m.start()
        if 0 < b < n and not boundary_inside_span(b, protected_spans):
            cand.setdefault(b, "C")

    return cand


def tier_rank(t: str) -> int:
    return {"A": 3, "B": 2, "C": 1}.get(t, 0)


def greedy_chunk_into_4(
    text: str,
    cand: Dict[int, str],
    protected_spans: List[Tuple[int, int]],
    max_len: Optional[int] = None,
    min_len: Optional[int] = None,
    tolerance: int = 50,
) -> List[str]:
    """
    Exactly 4 chunks using a greedy boundary picker (NER-aware).
    If max_len not provided, we set a target L = ceil(len(text)/4) and use max_len = L.
    """
    T = text
    n = len(T)
    if n == 0:
        return ["", "", "", ""]

    # default length parameters (character-based)
    L = max_len if max_len and max_len > 0 else (n + 3) // 4
    m = min_len if min_len and min_len > 0 else max(1, L // 2)

    # prepare sorted candidate boundaries
    boundaries = sorted(cand.keys())

    cuts: List[int] = []
    i = 0
    for chunk_idx in range(1, 4):  # choose 3 cut points
        remaining_chunks = 4 - chunk_idx
        latest_cut = n - remaining_chunks * m
        if latest_cut <= i:
            # cannot satisfy min length; just cut evenly without violating protected spans
            latest_cut = n - remaining_chunks

        target = i + L
        upper = min(target, latest_cut)

        # candidate set within (i, upper]
        best_b = None
        best_score = None

        for b in boundaries:
            if b <= i:
                continue
            if b > upper:
                break
            if boundary_inside_span(b, protected_spans):
                continue
            clen = b - i
            if clen < m:
                continue

            t = cand[b]
            score = (tier_rank(t), -abs(b - target))  # prefer higher tier, then closeness
            if (best_score is None) or (score > best_score):
                best_score = score
                best_b = b

        # if none, try nearest valid boundary AFTER target (<= latest_cut)
        if best_b is None:
            after_upper = min(target + tolerance, latest_cut)
            for b in boundaries:
                if b <= i:
                    continue
                if b > after_upper:
                    break
                if boundary_inside_span(b, protected_spans):
                    continue
                clen = b - i
                if clen < m:
                    continue
                best_b = b
                break

        # fallback: find nearest whitespace after target that doesn't cut spans
        if best_b is None:
            b = min(target, latest_cut)
            while b < latest_cut and boundary_inside_span(b, protected_spans):
                b += 1
            if b <= i:
                b = min(i + m, latest_cut)
            best_b = b

        cuts.append(best_b)
        i = best_b

    # materialize chunks
    prev = 0
    chunks: List[str] = []
    for c in cuts:
        chunks.append(T[prev:c].strip("\n"))
        prev = c
    chunks.append(T[prev:].strip("\n"))

    # ensure exactly 4
    if len(chunks) < 4:
        chunks += [""] * (4 - len(chunks))
    return chunks[:4]


def load_tasks_from_jsonl(path: Optional[str]) -> List[Tuple[str, str]]:
    tasks: List[Tuple[str, str]] = []
    if path:
        src = open(path, "r", encoding="utf-8")
        close_src = True
    else:
        src = sys.stdin
        close_src = False

    try:
        for line_no, line in enumerate(src, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            task_id = str(obj.get("id", f"line_{line_no}"))
            prompt = obj.get("prompt", "")
            tasks.append((task_id, "" if prompt is None else str(prompt)))
    finally:
        if close_src:
            src.close()

    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", type=str, default=None, help="Input path (JSONL or plain text). If omitted, reads stdin.")
    ap.add_argument("--jsonl_prompts", action="store_true", help="Force treat input as JSONL with fields {id, prompt}.")
    ap.add_argument("--out_full", type=str, default="harmful_full.txt")
    ap.add_argument("--out_frags", type=str, default="harmful_fragments.txt")

    # chunking controls (character-based)
    ap.add_argument("--max_len", type=int, default=0, help="Max chunk length (chars). Default auto = ceil(len/4).")
    ap.add_argument("--min_len", type=int, default=0, help="Min chunk length (chars). Default auto = max_len/2.")
    ap.add_argument("--tolerance", type=int, default=50, help="Allowed overshoot window when searching after target.")
    ap.add_argument("--spacy_model", type=str, default="en_core_web_sm", help="spaCy model name.")

    args = ap.parse_args()

    # spaCy load
    try:
        import spacy
        nlp = spacy.load(args.spacy_model)
    except Exception as e:
        raise SystemExit(
            f"Failed to load spaCy model '{args.spacy_model}'. "
            f"Install with:\n  pip install spacy\n  python -m spacy download {args.spacy_model}\n\nError: {e}"
        )

    treat_as_jsonl = bool(args.jsonl_prompts) or (args.in_path and args.in_path.lower().endswith(".jsonl"))

    os.makedirs(os.path.dirname(args.out_full), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_frags), exist_ok=True)

    # read tasks
    if treat_as_jsonl:
        tasks = load_tasks_from_jsonl(args.in_path)
    else:
        raw = read_input_text(args.in_path) or ""
        tasks = [("task_1", raw)]

    # -----------------------------
    # harmful_full.txt (RAW)
    # -----------------------------
    with open(args.out_full, "w", encoding="utf-8") as f_full:
        for k, (task_id, prompt) in enumerate(tasks, 1):
            f_full.write(f"### TASK {task_id}\n")
            f_full.write(prompt.rstrip("\n") + "\n")
            if k != len(tasks):
                f_full.write("\n")

    # -----------------------------
    # harmful_fragments.txt (task-by-task FRAGMENT 1..4)
    # -----------------------------
    with open(args.out_frags, "w", encoding="utf-8") as f_frags:
        for t_idx, (task_id, prompt) in enumerate(tasks, 1):
            text = normalize_newlines(prompt).strip("\n")

            protected_spans = collect_protected_spans_spacy(text, nlp)

            # meaning segmentation (produces segment ends => Tier A boundaries)
            segments, seg_end_positions = segment_by_meaning_ner_safe(text, protected_spans)

            # candidate boundaries with tiers
            cand = collect_candidate_boundaries(text, nlp, protected_spans, seg_end_positions)

            # greedy chunking into 4
            chunks = greedy_chunk_into_4(
                text=text,
                cand=cand,
                protected_spans=protected_spans,
                max_len=args.max_len if args.max_len > 0 else None,
                min_len=args.min_len if args.min_len > 0 else None,
                tolerance=max(0, int(args.tolerance)),
            )

            f_frags.write(f"### TASK {task_id}\n")
            for i, ch in enumerate(chunks, 1):
                f_frags.write(f"### FRAGMENT {i}\n")
                f_frags.write(ch.rstrip("\n") + "\n")
                if i != 4:
                    f_frags.write("\n" + ("-" * 40) + "\n\n")

            if t_idx != len(tasks):
                f_frags.write("\n" + ("=" * 60) + "\n\n")

    print(f"Wrote:\n  {args.out_full}\n  {args.out_frags}")


if __name__ == "__main__":
    main()
