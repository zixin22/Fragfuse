# Tasks10 Minimal Repo Guide

This repository is a minimized variant focused on:

- `src/inspect_evals/agentharm/tasks10`
- `src/inspect_evals/agentharm/tasks10_outputs`
- dependencies required by the tasks10 pipeline

It is **not** a full `inspect_evals` repository anymore.  
The root `README.md` describes the full project and may reference files/directories not present in this minimized version.

## 1) Environment

Use the conda env you prepared:

```bash
conda activate Agentharm
cd "/Users/zixinrao/Desktop/Agentharm-revised--main"
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

If needed:

```bash
pip install inspect-ai requests anyio
```

## 2) What is kept

- Core pipeline scripts:
  - `src/inspect_evals/agentharm/tasks10/1extract_terms.py`
  - `src/inspect_evals/agentharm/tasks10/2_1iterative_mask.py`
  - `src/inspect_evals/agentharm/tasks10/3sensitive_terms_summary.py`
  - `src/inspect_evals/agentharm/tasks10/3seperate_terms_summary.py`
  - `src/inspect_evals/agentharm/tasks10/4host_check.py`
  - `src/inspect_evals/agentharm/tasks10/5injection_store.py`
  - `src/inspect_evals/agentharm/tasks10/6retrieve_test.py`
  - `src/inspect_evals/agentharm/tasks10/7retrieve_guardrail_decision.py`
  - `src/inspect_evals/agentharm/tasks10/8retrieve_rate.py`
- Related dependency modules under `src/inspect_evals/agentharm` (e.g. `tasks3`, `agents`, `benchmark`)
- Data and outputs under `src/inspect_evals/agentharm/tasks10_outputs`
- `harmful_full.txt` and `harmful_fragments.txt`

## 3) Quick smoke checks

```bash
python src/inspect_evals/agentharm/tasks10/1extract_terms.py --help
python src/inspect_evals/agentharm/tasks10/2_1iterative_mask.py --help
python src/inspect_evals/agentharm/tasks10/4host_check.py --help
python src/inspect_evals/agentharm/tasks10/7retrieve_guardrail_decision.py --help
```

## 4) Common run pattern

### Step 1

```bash
python src/inspect_evals/agentharm/tasks10/1extract_terms.py \
  --env src/inspect_evals/agentharm/tasks10/openai_local.env \
  --input src/inspect_evals/agentharm/tasks10_outputs/Agentharm/iteration_3/masked_query_iteration2.json \
  --output src/inspect_evals/agentharm/tasks10_outputs/Agentharm/iteration_3/masked_query_step1_output.jsonl \
  --model gpt-4o \
  --overwrite
```

### Step 2 (skip guardrail)

```bash
python src/inspect_evals/agentharm/tasks10/2_1iterative_mask.py \
  --harmful_full src/inspect_evals/agentharm/tasks10_outputs/Agentharm/iteration_3/masked_query_iteration2.json \
  --step1_terms src/inspect_evals/agentharm/tasks10_outputs/Agentharm/iteration_3/masked_query_step1_output.jsonl \
  --out_masked src/inspect_evals/agentharm/tasks10_outputs/Agentharm/iteration_3/masked_query_step2_output.json \
  --skip_guardrail \
  --overwrite
```

## 5) Notes

- Some scripts accept `.json` paths that actually contain JSONL-style lines. This is expected in this workflow.
- Guardrail/model steps require valid API credentials and provider availability.
- If imports fail for `inspect_evals.*`, check `PYTHONPATH` first.

