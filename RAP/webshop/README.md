# WebShop Experiments (Simple Guide)

This directory runs three common experiment modes through `main.py`:

- Attack + RuleChecker (LLM-AC)
- Attack + GuardAgent
- Detection only (no attack)

## Main Commands

### 1) Attack with RuleChecker detection (LLM-AC)

```bash
python main.py --model gpt-4o --attack --dataset dataset_attack.json --defense_mode rule_checker --defense_check_target instruction --retrieve_mode rap --limit 100 --output output/rulechecker_test
```

### 2) Attack with GuardAgent detection

```bash
python main.py --model gpt-4o --attack --dataset dataset_attack.json --defense_mode guard_agent --defense_check_target instruction --retrieve_mode rap --limit 100 --output output/guardagent_test
```

### 3) Detection only (RuleChecker, no attack)

```bash
python main.py \
  --model gpt-4o \
  --dataset dataset_benign.json \
  --defense_mode rule_checker \
  --defense_check_target instruction \
  --output output/rulechecker_without_attack
```

You can switch modes with the same pattern:

- `--defense_mode rule_checker` ↔ `--defense_mode guard_agent`
- `--dataset dataset_attack.json` ↔ `--dataset dataset_benign.json`
- Change `--output` per run for clean experiment folders

## Key Parameters

- `--model`
  - Main model name, for example `gpt-4o`.

- `--attack`
  - Enables attack mode.
  - With this flag, the pipeline runs carrier/attack-query attack stages.
  - Without this flag, it runs normal task flow (can still run defense checks).

- `--dataset`
  - Input dataset JSON.
  - Attack experiments usually use `dataset_attack.json`.
  - Detection-only experiments usually use `dataset_benign.json`.

- `--defense_mode`
  - Defense type:
    - `rule_checker`
    - `guard_agent`
    - `none`

- `--defense_check_target`
  - Defense check target, usually `instruction`.

- `--retrieve_mode`
  - Retrieval mode:
    - `rap`: retrieval-enabled flow
    - `none`: no retrieval

- `--limit`
  - Runs only the first N cases, useful for quick checks (for example `--limit 100`).

- `--output`
  - Output directory for logs, summaries, memory, and evaluation artifacts.
  - Use a unique directory per run for easy comparison.

## Common Output Files

Output files vary by mode, but commonly include:

- `attack_summary.txt`: high-level run summary
- `rulechecker_log.txt` or corresponding GuardAgent logs
- `memory_1.json`: session memory and retrieval records
- `rule_violation.txt`: rule violation statistics

