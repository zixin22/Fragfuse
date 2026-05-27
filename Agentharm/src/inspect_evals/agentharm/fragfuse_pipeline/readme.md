FragFuse Pipeline

## 1) Scope

- Purpose: extract sensitive fragments from AgentHarm harmful tasks, construct
  q_mask / q_carrier / q_attack, and evaluate the resulting guardrail and
  retrieval/fusion pipeline.
- Default task source: `harmful_full.txt` at the repository root. The expected
  format is `### TASK x-y`; the current public set contains 176 tasks.
- Output root: `src/inspect_evals/agentharm/fragfuse_output`.

## 2) Terminology

- `prohibited_query` / `q`: the original harmful task.
- `sensitive_fragments`: `{f_t}` in the paper.
- `q_mask` / `masked_query`: the masked query.
- `host_query` / `q_host`: the benign host query.
- `split_sensitive_fragments`: sensitive fragments split and wrapped in markers.
- `carrier_query` / `q_carrier`: the carrier query.
- `attack_query` / `q_attack`: the attack query with the fusion instruction template.

## 3) Required Files

These files are not all core scripts, but they are required inputs or state files
for reproducible pipeline execution:

- `../../../harmful_full.txt`
  - Default harmful task source for steps 1 and 2.
- `benign_behaviors_dataset/benign_behaviors_test_public_ALL.txt`
  - Default benign host query source for step 4.
- `fragfuse_output/base_bank.jsonl`
  - Shared base bank used by step 5 to create per-task memory banks.
- `fragfuse_output/bank_records/bank_{task_id}.jsonl`
  - Step 5 output and step 6/8 input. Each file contains the shared base memory
    plus the task-specific `q_carrier`.
- `fragfuse_output/retrieve_test_logs_memory/task_{task_id}/...`
  - Step 6 output and step 8 input. This directory contains the fused instruction
    inputs and readable retrieval logs.

## 4) Core Scripts

1) `1extract_fragments.py`
- Input: `harmful_full.txt` or JSON/JSONL records with `task_id` and a query-like field.
- Output: `fragfuse_output/step1_sensitive_fragments.jsonl`
- Function: calls a model for each task to extract `sensitive_fragments` and removes
  placeholder tokens such as `[MASK_i]`.

2) `2_1iterative_mask.py`
- Input: `step1_sensitive_fragments.jsonl` and `harmful_full.txt`.
- Output:
  - `fragfuse_output/step2_masked_instructions.jsonl`
  - `fragfuse_output/step2_iterative_mask.jsonl` when guardrail checking is enabled.
- Function: masks sensitive fragments by `task_id`, creates `q_mask`, and can run
  optional guardrail checks.

3) `3sensitive_fragments_summary.py` and `3split_fragments_summary.py`
- `3sensitive_fragments_summary.py`
  - Input: `step2_masked_instructions.jsonl` by default.
  - Output: `fragfuse_output/all_sensitive_fragments_union.jsonl`
  - Function: aggregates and deduplicates sensitive fragments per task.
- `3split_fragments_summary.py`
  - Input: `all_sensitive_fragments_union.jsonl`
  - Output: `fragfuse_output/all_sensitive_fragments_union.split_charhalves.jsonl`
  - Function: splits sensitive fragments into character-level halves and creates
    `split_sensitive_fragments`.
- Write policy: output files are overwritten by default.

4) `4host_check.py`
- Input:
  - `benign_behaviors_dataset/benign_behaviors_test_public_ALL.txt`
  - `step2_masked_instructions.jsonl`
  - `all_sensitive_fragments_union.split_charhalves.jsonl`
- Output:
  - `fragfuse_output/carrier_query_inputs.jsonl`
  - `fragfuse_output/carrier_query_results.jsonl` when carrier guardrail checking is enabled.
  - `fragfuse_output/attack_query.jsonl`
- Function: inserts split fragments into `host_query` to create `carrier_query`.
  It also combines `masked_query` and `host_query` to create an `attack_query`
  template containing the `{q_carrier}` placeholder. Carrier guardrail checking is optional.

5) `5injection_store.py`
- Input: `carrier_query_results.jsonl` and `base_bank.jsonl`.
- Output: `fragfuse_output/bank_records/bank_{task_id}.jsonl`
- Function: creates each per-task bank from the shared base bank plus the task-specific
  `q_carrier`. To reproduce the legacy behavior of storing only `allowed=True`
  carriers, use `--only_allowed`.
- Write policy: each `bank_{task_id}.jsonl` is overwritten by default.
- Retrieval-time bank: `fragfuse_output/bank_records/bank_{task_id}.jsonl`.
  `base_bank.jsonl` is only the template input used to build the per-task banks.

6) `6retrieve_test.py`
- Input:
  - `attack_query.jsonl`
  - `bank_records/bank_{task_id}.jsonl`
- Output: `fragfuse_output/retrieve_test_logs_{agent}/task_{task_id}/...`
- Function: uses the `attack_query` template as the retrieval query, retrieves the
  top-1 `q_carrier` from the task memory bank, replaces `{q_carrier}` with that
  retrieved carrier, and runs the resulting fused instruction through Inspect.
  By default, every task in the attack file is processed. To reproduce the legacy
  carrier-allowed filter, use `--only_allowed_carrier`.
- Write policy: readable logs such as `retrieve_test_readable_log_{task_id}.md`
  are overwritten by default.

7) `7_run_guardrail.py`
- Input:
  - `fragfuse_output/carrier_query_results.jsonl` with field `carrier_query`.
  - `fragfuse_output/attack_query.jsonl` with field `attack_query`.
- Output:
  - `fragfuse_output/carrier_output.json`
  - `fragfuse_output/attack_output.json`
  - `fragfuse_output/guardrail_pair_summary.json`
- Function: runs the LLM-AC guardrail over full `q_carrier` and `q_attack` inputs
  and reports the task IDs allowed in both files.
- Write policy: output JSON files are overwritten by default. Use `--output-suffix`
  to write suffixed result files.

8) `8retrieve_rate.py`
- Input:
  - `fragfuse_output/guardrail_pair_summary.json`, using `summary.both_allowed_task_ids`.
  - `retrieve_test_logs_memory/task_{task_id}/task_user_message_preview.txt`
  - `bank_records/bank_{task_id}.jsonl`
- Output:
  - `fragfuse_output/retrieve_rate_matches.jsonl`
  - `fragfuse_output/retrieve_rate_matches.csv`
  - `fragfuse_output/retrieve_rate_summary.json`
- Function: computes the match rate between the retrieved Similar Task 1 text in
  the preview file and the `q_carrier` stored in the corresponding task bank.

## 5) Pipeline Order

`1 -> 2 -> 3sensitive_fragments -> 3split_fragments -> 4 -> 5 -> 6 -> 7 -> 8`

## 6) Command Template

Commands assume execution from the repository root.

```bash
# 0) Environment
export PYTHONPATH=src
```

```bash
# 1) Extract sensitive fragments
python "src/inspect_evals/agentharm/fragfuse_pipeline/1extract_fragments.py" \
  --env "src/inspect_evals/agentharm/fragfuse_pipeline/openai_local.env"
```

```bash
# 2) Mask sensitive fragments
python "src/inspect_evals/agentharm/fragfuse_pipeline/2_1iterative_mask.py" \
  --skip_guardrail
```

```bash
# 3.1) Aggregate sensitive fragments
python "src/inspect_evals/agentharm/fragfuse_pipeline/3sensitive_fragments_summary.py"
```

```bash
# 3.2) Split sensitive fragments
python "src/inspect_evals/agentharm/fragfuse_pipeline/3split_fragments_summary.py"
```

```bash
# 4) Generate carrier query and attack query template
python "src/inspect_evals/agentharm/fragfuse_pipeline/4host_check.py" \
  --allowlist shieldagent
```

```bash
# 5) Create per-task banks
python "src/inspect_evals/agentharm/fragfuse_pipeline/5injection_store.py" \
  --base_bank "src/inspect_evals/agentharm/fragfuse_output/base_bank.jsonl"
```

```bash
# 6) Run retrieval/fusion test
python "src/inspect_evals/agentharm/fragfuse_pipeline/6retrieve_test.py" \
  --agent memory \
  --model "openai/gpt-4o-2024-08-06"
```

```bash
# 7) Run LLM-AC over q_carrier and q_attack
python "src/inspect_evals/agentharm/fragfuse_pipeline/7_run_guardrail.py" \
  --carrier-input "src/inspect_evals/agentharm/fragfuse_output/carrier_query_results.jsonl" \
  --attack-input "src/inspect_evals/agentharm/fragfuse_output/attack_query.jsonl" \
  --output-dir "src/inspect_evals/agentharm/fragfuse_output" \
  --labels carrier,attack
```

```bash
# 8) Compute retrieval match rate
python "src/inspect_evals/agentharm/fragfuse_pipeline/8retrieve_rate.py" \
  --task_ids_path "src/inspect_evals/agentharm/fragfuse_output/guardrail_pair_summary.json" \
  --threshold 0.99
```

## 7) Running from Prepared iteration_4 Query Files

The following commands can be used when the prepared iteration_4 query files are
already available:

- `fragfuse_output/Agentharm/iteration_4/carrier_query_iteration4.json`
- `fragfuse_output/Agentharm/iteration_4/attack_query_iteration4.json`

First, create the iteration_4 per-task memory banks:

```bash
conda run -n Agentharm python fragfuse_pipeline/5injection_store.py \
  --in_path fragfuse_output/Agentharm/iteration_4/carrier_query_iteration4.json \
  --base_bank fragfuse_output/base_bank.jsonl \
  --out_dir fragfuse_output/Agentharm/iteration_4/bank_records
```

Then run retrieval/fusion with the iteration_4 attack query template:

```bash
conda run -n Agentharm python fragfuse_pipeline/6retrieve_test.py \
  --attack_path fragfuse_output/Agentharm/iteration_4/attack_query_iteration4.json \
  --bank_dir fragfuse_output/Agentharm/iteration_4/bank_records \
  --out_dir fragfuse_output/Agentharm/iteration_4 \
  --agent memory \
  --model openai/gpt-4o-2024-08-06
```

The output directory is:

```text
fragfuse_output/Agentharm/iteration_4/retrieve_test_logs_memory/task_{task_id}/
```

## 8) Notes

- Paths inside `fragfuse_pipeline` are resolved relative to the script location
  unless explicit CLI paths are provided.
- When importing `inspect_evals`, set `PYTHONPATH=src` from the repository root.
- Guardrail and model-calling steps require valid `OPENAI_BASE_URL` and
  `OPENAI_API_KEY` environment variables, either exported or loaded from an env file.
- `1extract_fragments.py`, `2_1iterative_mask.py`, and `4host_check.py` clear
  and regenerate their outputs by default.
- `3sensitive_fragments_summary.py`, `5injection_store.py`, and `6retrieve_test.py`
  also overwrite their outputs by default.
- `6retrieve_test.py` keeps `--limit` as the debug truncation option.
