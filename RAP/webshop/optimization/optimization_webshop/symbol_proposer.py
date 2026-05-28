"""
LLM proposer for delimiter pairs around {sensitive_fragment} in carrier-query templates.
Does not use user's seed file lines as few-shot examples (those are for initialization only).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, List, Optional

from carrier_query_template_utils import (
    apply_delimiters,
    apply_delimiters_split,
    load_carrier_query_template_lines,
    parse_carrier_query_template_structure,
    validate_template_format,
)
from proposer import LLMInterface


DELIMITER_SYSTEM_USER_PROMPT = """You help choose delimiter tokens that wrap {sensitive_fragment} inside a Python str.format template. The template already contains {q_host}.

Goal: propose diverse delimiter pairs. A balanced mix is preferred: some bracket-like pairs and some symbol/mixed-style pairs.

Hard requirements:
1) Return exactly one JSON object with two keys only: "left" and "right"
2) Keep each delimiter short (1-4 Unicode chars)
3) Do NOT include "{" or "}"
4) Output JSON only (no markdown, no extra text)

Examples (balanced style mix; format only):
Bracket-like:
{"left":"<","right":">"}
{"left":"(","right":")"}
{"left":"[","right":"]"}
{"left":"«","right":"»"}
{"left":"⟨","right":"⟩"}

Symbol / mixed:
{"left":"§","right":"§"}
{"left":"※","right":"※"}
{"left":"¦","right":"¦"}
{"left":"~|","right":"|~"}
{"left":"✦","right":"✦"}
{"left":"¶","right":"¶"}"""


class SymbolProposer:
    def __init__(self, config: Any, llm: Optional[LLMInterface] = None):
        self.config = config
        self.llm = llm or LLMInterface(config)
        self._canonical_template = self._load_canonical_seed_template()

    def _load_canonical_seed_template(self) -> Optional[str]:
        """
        Canonical carrier-query template skeleton: first seed line.
        This enforces "delimiter-only" proposing with fixed non-delimiter text.
        """
        try:
            template_file = os.path.join(
                self.config.base_dir, "data_webshop", "carrier_query_template_seed.txt"
            )
            templates = load_carrier_query_template_lines(template_file)
            if templates:
                return templates[0]
        except Exception as e:
            print(f"Symbol proposer canonical seed load failed: {e}")
        return None

    def _parse_delimiter_json(self, text: str) -> Optional[tuple]:
        raw = text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{[^{}]*\"left\"[^{}]*\"right\"[^{}]*\}", raw, re.DOTALL)
            if not m:
                return None
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        if not isinstance(obj, dict):
            return None
        left = obj.get("left")
        right = obj.get("right")
        if not isinstance(left, str) or not isinstance(right, str):
            return None
        left, right = left.strip(), right.strip()
        if not left or not right:
            return None
        if len(left) > 8 or len(right) > 8:
            return None
        if "{" in left or "}" in left or "{" in right or "}" in right:
            return None
        return left, right

    def propose_delimiter_pair(self) -> Optional[tuple]:
        try:
            text = self.llm.generate(DELIMITER_SYSTEM_USER_PROMPT)
            return self._parse_delimiter_json(text)
        except Exception as e:
            print(f"Symbol proposer LLM call failed: {e}")
            return None

    def template_from_delimiters(self, base_template: str, left: str, right: str) -> Optional[str]:
        try:
            prefix, _old_l, _old_r, suffix = parse_carrier_query_template_structure(base_template)
        except ValueError as e:
            print(f"Symbol proposer parse failed: {e}")
            return None
        if "{sensitive_fragment_part1}" in base_template and "{sensitive_fragment_part2}" in base_template:
            candidate = apply_delimiters_split(prefix, left, right, suffix)
        else:
            candidate = apply_delimiters(prefix, left, right, suffix)
        if candidate == base_template:
            return None
        if not validate_template_format(candidate):
            return None
        return candidate

    def generate_candidates(self, current_templates: List[str]) -> List[str]:
        """
        Delimiter-only proposer:
        - Keep non-delimiter text fixed to the canonical seed template.
        - Only propose new left/right delimiters around {sensitive_fragment}.
        """
        variants_per = getattr(self.config, "symbol_proposer_variants_per_template", 3)
        candidates: List[str] = []
        seen = set(current_templates)
        base_template = self._canonical_template or (current_templates[0] if current_templates else None)
        if not base_template:
            print("Symbol proposer: no base template available.")
            return []

        for _ in range(max(variants_per, 1)):
            pair = self.propose_delimiter_pair()
            if not pair:
                continue
            left, right = pair
            new_t = self.template_from_delimiters(base_template, left, right)
            if new_t and new_t not in seen:
                seen.add(new_t)
                candidates.append(new_t)

        print(
            f"Symbol proposer: generated {len(candidates)} candidate carrier-query templates "
            f"({variants_per} attempts on canonical base template)."
        )
        return candidates
