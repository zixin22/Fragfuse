"""
Parse and rebuild carrier-query f-string templates by swapping delimiters.
Supports both:
1) single-fragment form: ... L{sensitive_fragment}R ...
2) split-fragment form: ... L{sensitive_fragment_part1}R L{sensitive_fragment_part2}R ...
"""

from __future__ import annotations

from typing import Tuple

HI = "{q_host}"
FRAG = "{sensitive_fragment}"
FRAG1 = "{sensitive_fragment_part1}"
FRAG2 = "{sensitive_fragment_part2}"


def parse_carrier_query_template_structure(template: str) -> Tuple[str, str, str, str]:
    """
    Split template into static_prefix + ldelim + {sensitive_fragment} + rdelim + static_suffix.

    static_suffix is the closing quote segment (e.g. .' or .") when present; rdelim is the
    substring between {sensitive_fragment} and that suffix.
    """
    if HI not in template:
        raise ValueError("Template must contain {q_host}")

    # Prefer split-fragment form if both placeholders are present.
    if FRAG1 in template and FRAG2 in template:
        return _parse_split_fragment_template(template)
    if FRAG in template:
        return _parse_single_fragment_template(template)

    raise ValueError("Template must contain either {sensitive_fragment} or both {sensitive_fragment_part1}/{sensitive_fragment_part2}")


def _parse_single_fragment_template(template: str) -> Tuple[str, str, str, str]:
    """Split single-fragment form into static_prefix + ldelim + {sensitive_fragment} + rdelim + static_suffix."""

    i = template.index(FRAG)
    left_chunk = template[:i]
    right_chunk = template[i + len(FRAG) :]

    hi_pos = left_chunk.index(HI)
    static_base = left_chunk[: hi_pos + len(HI)]
    remainder = left_chunk[len(static_base) :]

    if right_chunk.endswith(".'"):
        static_suffix = ".'"
        rmid = right_chunk[:-2]
    elif right_chunk.endswith('."'):
        static_suffix = '."'
        rmid = right_chunk[:-2]
    else:
        static_suffix = ""
        rmid = right_chunk

    if not remainder:
        raise ValueError("Empty segment between {q_host} and {sensitive_fragment}")

    for l_len in range(1, min(8, len(remainder) + 1)):
        ldelim = remainder[-l_len:]
        mid_left = remainder[:-l_len]
        static_prefix_full = static_base + mid_left
        cand = static_prefix_full + ldelim + FRAG + rmid + static_suffix
        if cand == template:
            return static_prefix_full, ldelim, rmid, static_suffix

    raise ValueError(f"Could not parse single-fragment template delimiters: {template[:120]!r}")


def _parse_split_fragment_template(template: str) -> Tuple[str, str, str, str]:
    """
    Split split-fragment form into:
      static_prefix + ldelim + {sensitive_fragment_part1} + rdelim + ldelim + {sensitive_fragment_part2} + rdelim + static_suffix
    Returns (static_prefix, ldelim, rdelim, static_suffix), where static_suffix is tail after second rdelim.
    """
    if FRAG1 not in template or FRAG2 not in template:
        raise ValueError("Split-fragment template must contain both {sensitive_fragment_part1} and {sensitive_fragment_part2}")

    i1 = template.index(FRAG1)
    i2 = template.index(FRAG2)
    if i2 <= i1:
        raise ValueError("Invalid fragment placeholder order in split-fragment template")

    left_chunk = template[:i1]
    between = template[i1 + len(FRAG1):i2]
    tail = template[i2 + len(FRAG2):]

    hi_pos = left_chunk.index(HI)
    static_base = left_chunk[: hi_pos + len(HI)]
    remainder = left_chunk[len(static_base):]
    if not remainder:
        raise ValueError("Empty segment between {q_host} and {sensitive_fragment_part1}")

    # between should be rdelim + ldelim, and tail should start with rdelim
    max_len = min(8, len(remainder))
    for l_len in range(1, max_len + 1):
        ldelim = remainder[-l_len:]
        static_prefix_full = static_base + remainder[:-l_len]

        for r_len in range(1, min(8, len(between)) + 1):
            rdelim = between[:r_len]
            if between[r_len:] != ldelim:
                continue
            if not tail.startswith(rdelim):
                continue
            static_suffix = tail[r_len:]
            cand = (
                static_prefix_full
                + ldelim
                + FRAG1
                + rdelim
                + ldelim
                + FRAG2
                + rdelim
                + static_suffix
            )
            if cand == template:
                return static_prefix_full, ldelim, rdelim, static_suffix

    raise ValueError(f"Could not parse split-fragment template delimiters: {template[:120]!r}")


def apply_delimiters(
    static_prefix: str, ldelim: str, rdelim: str, static_suffix: str
) -> str:
    """Build a full template string from parsed pieces and new delimiters."""
    # If the canonical seed template is split-fragment, keep split-fragment form.
    if FRAG1 in static_prefix or FRAG2 in static_prefix:
        # Defensive fallback; static_prefix should not include fragment placeholders.
        pass
    return static_prefix + ldelim + FRAG + rdelim + static_suffix


def apply_delimiters_split(
    static_prefix: str, ldelim: str, rdelim: str, static_suffix: str
) -> str:
    """Build split-fragment form with one delimiter pair applied to both fragment parts."""
    return static_prefix + ldelim + FRAG1 + rdelim + ldelim + FRAG2 + rdelim + static_suffix


def validate_template_format(template: str) -> bool:
    """Ensure str.format works for typical host/fragment placeholders."""
    if "{" in template and HI not in template:
        return False
    try:
        template.format(
            q_host="__HOST__",
            fragment="__SENSITIVE_FRAGMENT__",
            sensitive_fragment_part1="__FRAG1__",
            sensitive_fragment_part2="__FRAG2__",
        )
    except (KeyError, ValueError, IndexError):
        return False
    return True


def load_carrier_query_template_lines(template_file: str) -> list:
    """Load the RHS of each carrier-query template assignment line."""
    out = []
    with open(template_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            rhs = line.split("=", 1)[1].strip()
            if rhs:
                out.append(rhs)
    return out

