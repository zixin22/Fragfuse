#!/usr/bin/env python3
"""Run the LLM-AC guardrail over carrier, mask, and attack query files."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _repo_src() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_import_path() -> None:
    src = str(_repo_src())
    if src not in sys.path:
        sys.path.insert(0, src)


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and not os.environ.get(key):
            os.environ[key] = value


def _load_openai_key_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key in {"OPENAI_API_KEY", "OPENAI_BASE_URL"} and value and not os.environ.get(key):
                os.environ[key] = value
            continue
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = line
            return


def _load_api_key_file(path: Path) -> None:
    if os.environ.get("OPENAI_API_KEY"):
        return
    _load_openai_key_file(path)


def _normalize_openai_base_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if normalized and not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


def _load_local_env() -> None:
    repo_root = _repo_src().parent
    task_dir = Path(__file__).resolve().parent
    _load_env_file(repo_root / ".env")
    _load_env_file(Path.cwd() / ".env")
    _load_env_file(task_dir / "openai_local.env")
    _load_openai_key_file(task_dir / "openai_key.txt")
    base_url = _normalize_openai_base_url(os.environ.get("OPENAI_BASE_URL") or "")
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(obj)
    return rows


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _summary_path(base_dir: Path, output_suffix: str) -> Path:
    return base_dir / f"guardrail_pair_summary{output_suffix}.json"


def _write_pair_summary(
    *,
    output_dir: Path,
    output_suffix: str,
    outputs_by_label: dict[str, dict[str, Any]],
) -> None:
    carrier = outputs_by_label.get("carrier")
    attack = outputs_by_label.get("attack")
    if not carrier or not attack:
        return

    carrier_summary = carrier.get("summary", {})
    attack_summary = attack.get("summary", {})
    carrier_passed = set(carrier_summary.get("passed_task_ids", []) or [])
    attack_passed = set(attack_summary.get("passed_task_ids", []) or [])
    both_allowed_task_ids = sorted(carrier_passed & attack_passed)

    output = {
        "label": "carrier_attack_pair",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "carrier_output_file": carrier.get("output_file"),
        "attack_output_file": attack.get("output_file"),
        "summary": {
            "carrier_total_count": carrier_summary.get("total_count", 0),
            "carrier_allowed_count": carrier_summary.get("passed_count", 0),
            "attack_total_count": attack_summary.get("total_count", 0),
            "attack_allowed_count": attack_summary.get("passed_count", 0),
            "both_allowed_count": len(both_allowed_task_ids),
            "both_allowed_task_ids": both_allowed_task_ids,
        },
    }
    out_path = _summary_path(output_dir, output_suffix)
    _write_json(out_path, output)
    print(f"[carrier+attack] wrote {out_path}", flush=True)


async def _run_one(
    *,
    row: dict[str, Any],
    input_field: str,
    model: str,
    max_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    from inspect_evals.agentharm.guardrail_prompt.guardrail_prompt import (
        render_guardrail_prompt,
    )

    task_id = str(row.get("task_id", ""))
    text = str(row.get(input_field, ""))
    if not task_id:
        raise ValueError("Encountered row without task_id")
    if not text.strip():
        raise ValueError(f"Task {task_id} has empty {input_field}")

    async with semaphore:
        try:
            prompt = render_guardrail_prompt(agent_trajectory="", observation="", action=text)
            raw_text = await _call_openai_sdk(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            try:
                decision = _parse_guardrail_decision(raw_text)
                error = None
            except Exception as exc:
                decision = {
                    "allowed": False,
                    "violated_rule_indices": ["GUARDRAIL_PARSE_ERROR"],
                    "risk_category": 0,
                    "explanation": f"GUARDRAIL_PARSE_ERROR: {type(exc).__name__}: {exc}",
                    "raw_text": raw_text,
                    "parse_error": f"{type(exc).__name__}: {exc}",
                }
                error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # Keep batch outputs complete even if one call fails.
            decision = {
                "allowed": False,
                "violated_rule_indices": ["GUARDRAIL_CALL_ERROR"],
                "risk_category": 0,
                "explanation": f"GUARDRAIL_CALL_ERROR: {type(exc).__name__}: {exc}",
                "raw_text": "",
                "parse_error": f"{type(exc).__name__}: {exc}",
            }
            error = f"{type(exc).__name__}: {exc}"

    passed = bool(decision.get("allowed", False)) if isinstance(decision, dict) else False
    return {
        "task_id": task_id,
        "input_field": input_field,
        "input": text,
        "passed": passed,
        "guardrail_decision": decision,
        "llm_output": decision.get("raw_text", "") if isinstance(decision, dict) else "",
        "error": error,
    }


async def _call_openai_sdk(
    *,
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    from openai import AsyncOpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    model_name = model.removeprefix("openai/")
    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
    )
    response = await client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        choices = response.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            return str(message.get("content") or choices[0].get("text") or "")
        return str(response.get("content") or response.get("text") or "")
    return response.choices[0].message.content or ""


_CODEFENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _strip_code_fences(text: str) -> str:
    return _CODEFENCE_RE.sub("", text).strip()


def _parse_guardrail_decision(raw_text: str) -> dict[str, Any]:
    clean = _strip_code_fences(raw_text)
    parsed = json.loads(clean)
    if not isinstance(parsed, dict):
        raise ValueError("Guardrail output JSON is not an object")
    for key in ("allowed", "violated_rule_indices", "risk_category", "explanation"):
        if key not in parsed:
            raise ValueError(f"Missing required key: {key}")
    return {
        "allowed": bool(parsed.get("allowed", True)),
        "violated_rule_indices": parsed.get("violated_rule_indices", []) or [],
        "risk_category": int(parsed.get("risk_category", 0) or 0),
        "explanation": str(parsed.get("explanation", "")),
        "raw_text": raw_text,
        "parse_error": None,
    }


async def _run_file(
    *,
    input_path: Path,
    output_path: Path,
    input_field: str,
    label: str,
    model: str,
    max_tokens: int,
    temperature: float,
    concurrency: int,
    task_ids: set[str] | None,
) -> dict[str, Any]:
    rows = _read_jsonl(input_path)
    if task_ids is not None:
        rows = [row for row in rows if str(row.get("task_id", "")) in task_ids]
    semaphore = asyncio.Semaphore(concurrency)
    results: list[dict[str, Any]] = []

    for index in range(0, len(rows), concurrency):
        batch = rows[index : index + concurrency]
        batch_results = await asyncio.gather(
            *[
                _run_one(
                    row=row,
                    input_field=input_field,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    semaphore=semaphore,
                )
                for row in batch
            ]
        )
        results.extend(batch_results)
        print(f"[{label}] checked {len(results)}/{len(rows)}", flush=True)

    passed_task_ids = [r["task_id"] for r in results if r["passed"]]
    failed_task_ids = [r["task_id"] for r in results if not r["passed"]]
    errored_task_ids = [r["task_id"] for r in results if r.get("error")]
    output = {
        "label": label,
        "input_file": str(input_path),
        "output_file": str(output_path),
        "input_field": input_field,
        "guardrail_model": model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_count": len(results),
            "passed_count": len(passed_task_ids),
            "failed_count": len(failed_task_ids),
            "error_count": len(errored_task_ids),
            "passed_task_ids": passed_task_ids,
            "failed_task_ids": failed_task_ids,
            "errored_task_ids": errored_task_ids,
        },
        "results": results,
    }
    _write_json(output_path, output)
    print(f"[{label}] wrote {output_path}", flush=True)
    return output


async def _amain(args: argparse.Namespace) -> None:
    _ensure_import_path()
    _load_local_env()
    if args.api_base_url:
        os.environ["OPENAI_BASE_URL"] = _normalize_openai_base_url(args.api_base_url)
    if args.api_key_file:
        os.environ.pop("OPENAI_API_KEY", None)
        _load_api_key_file(Path(args.api_key_file))

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir) if args.output_dir else base_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (
            "carrier",
            Path(args.carrier_input) if args.carrier_input else base_dir / "carrier_query_results.jsonl",
            output_dir / "carrier_output.json",
            "carrier_query",
        ),
        (
            "mask",
            Path(args.mask_input) if args.mask_input else base_dir / "step2_masked_instructions.jsonl",
            output_dir / "mask_output.json",
            "masked_query",
        ),
        (
            "attack",
            Path(args.attack_input) if args.attack_input else base_dir / "attack_query.jsonl",
            output_dir / "attack_output.json",
            "attack_query",
        ),
    ]
    selected_labels = set(args.labels.split(",")) if args.labels else None
    task_ids = set(args.task_ids.split(",")) if args.task_ids else None

    outputs_by_label: dict[str, dict[str, Any]] = {}
    for label, input_path, output_path, input_field in jobs:
        if selected_labels is not None and label not in selected_labels:
            continue
        if args.output_suffix:
            output_path = output_path.with_name(
                f"{output_path.stem}{args.output_suffix}{output_path.suffix}"
            )
        outputs_by_label[label] = await _run_file(
            input_path=input_path,
            output_path=output_path,
            input_field=input_field,
            label=label,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            concurrency=args.concurrency,
            task_ids=task_ids,
        )
    _write_pair_summary(
        output_dir=output_dir,
        output_suffix=args.output_suffix,
        outputs_by_label=outputs_by_label,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the LLM-AC guardrail over carrier, masked, and attack query files."
        )
    )
    parser.add_argument(
        "--base-dir",
        default=str(Path(__file__).resolve().parents[1] / "fragfuse_output"),
        help="Directory for default input and output files. Explicit input paths override this.",
    )
    parser.add_argument("--output-dir", default="", help="Directory for output JSON files. Defaults to --base-dir.")
    parser.add_argument("--carrier-input", default="", help="Path to carrier query JSONL. Defaults to <base-dir>/carrier_query_results.jsonl.")
    parser.add_argument("--mask-input", default="", help="Path to masked query JSONL. Defaults to <base-dir>/step2_masked_instructions.jsonl.")
    parser.add_argument("--attack-input", default="", help="Path to attack query JSONL. Defaults to <base-dir>/attack_query.jsonl.")
    parser.add_argument("--model", default="openai/gpt-4o-2024-08-06")
    parser.add_argument(
        "--api-base-url",
        default="",
        help="OpenAI-compatible API base URL. /v1 is appended when omitted.",
    )
    parser.add_argument(
        "--api-key-file",
        default="",
        help="Read the OpenAI-compatible API key from this file.",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--labels",
        default="",
        help="Comma-separated subset of labels to run: carrier,mask,attack. Use carrier,attack to check only q_carrier and q_attack.",
    )
    parser.add_argument("--task-ids", default="", help="Comma-separated task_id subset.")
    parser.add_argument("--output-suffix", default="", help="Suffix before .json for test runs.")
    args = parser.parse_args()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
