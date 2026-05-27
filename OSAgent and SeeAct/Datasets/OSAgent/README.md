# OSAgent FragFuse Data

## Datasets

| File | Records | Role |
| --- | ---: | --- |
| `benign.json` | 27 | Benign OSAgent tasks for utility evaluation. |
| `environment.json` | 20 | Original prohibited-query dataset for the environment setting. |
| `system_sabotage.json` | 30 | Original prohibited-query dataset for the system sabotage setting. |

## Pipeline

FragFuse builds two independent query inputs from each prohibited-query dataset:

```text
prohibited_query -> sensitive_fragments
host_query + sensitive_fragments -> carrier_query
host_query + fragment-recovery instruction -> attack_query
```

`carrier_query` and `attack_query` are run separately. The current code does not automatically run a multi-turn session where the carrier query is executed first and the attack query consumes that runtime context.

## Environment

| File | Records | Field |
| --- | ---: | --- |
| `environment/host_instructions.json` | 20 | `host_query` |
| `environment/environment_sensitive_fragments-0.json` | 20 | `prohibited_query`, `sensitive_fragments` |
| `environment/environment_sensitive_fragments-1.json` | 20 | `prohibited_query`, `sensitive_fragments` |
| `environment/environment_sensitive_fragments-2.json` | 20 | `prohibited_query`, `sensitive_fragments` |
| `environment/environment_sensitive_fragments-3.json` | 20 | `prohibited_query`, `sensitive_fragments` |
| `environment/carrier_queries.json` | 540 | `carrier_query` |
| `environment/attack_queries.json` | 180 | `attack_query`, `sensitive_fragments` |

## System Sabotage

| File | Records | Field |
| --- | ---: | --- |
| `system_sabotage/host_instructions.json` | 30 | `host_query` |
| `system_sabotage/system_sabotage_sensitive_fragments-0.json` | 30 | `prohibited_query`, `sensitive_fragments` |
| `system_sabotage/system_sabotage_sensitive_fragments-1.json` | 30 | `prohibited_query`, `sensitive_fragments` |
| `system_sabotage/system_sabotage_sensitive_fragments-2.json` | 30 | `prohibited_query`, `sensitive_fragments` |
| `system_sabotage/system_sabotage_sensitive_fragments-3.json` | 30 | `prohibited_query`, `sensitive_fragments` |
| `system_sabotage/system_sabotage_sensitive_fragments-4.json` | 30 | `prohibited_query`, `sensitive_fragments` |
| `system_sabotage/system_sabotage_sensitive_fragments-5.json` | 30 | `prohibited_query`, `sensitive_fragments` |
| `system_sabotage/carrier_queries.json` | 810 | `carrier_query` |
| `system_sabotage/attack_queries.json` | 270 | `attack_query`, `sensitive_fragments` |

## Run

From `AGrail4Agent/`:

Benign utility:

```bash
python DAS/exp_OS.py \
  --dataset ../Datasets/OSAgent/benign.json \
  --memory memory_os_benign_claude.json \
  --agent_model gpt-4o \
  --use_memory \
  --memory_bank memory_bank-32.csv
```

Environment carrier query:

```bash
python DAS/exp_OS.py \
  --dataset ../Datasets/OSAgent/environment/carrier_queries.json \
  --memory memory_os_environment_gpt.json \
  --agent_model gpt-4o \
  --use_guardrail \
  --guardrail_model gpt-4o \
  --only_guard
```

Environment attack query:

```bash
python DAS/exp_OS.py \
  --dataset ../Datasets/OSAgent/environment/attack_queries.json \
  --memory memory_os_environment_gpt.json \
  --agent_model gpt-4o \
  --use_guardrail \
  --guardrail_model gpt-4o \
  --only_guard
```

System sabotage carrier query:

```bash
python DAS/exp_OS.py \
  --dataset ../Datasets/OSAgent/system_sabotage/carrier_queries.json \
  --memory memory_os_system_sabotage_gpt.json \
  --agent_model gpt-4o \
  --use_guardrail \
  --guardrail_model gpt-4o \
  --only_guard
```

System sabotage attack query:

```bash
python DAS/exp_OS.py \
  --dataset ../Datasets/OSAgent/system_sabotage/attack_queries.json \
  --memory memory_os_system_sabotage_gpt.json \
  --agent_model gpt-4o \
  --use_guardrail \
  --guardrail_model gpt-4o \
  --only_guard
```

Default memory bank:

```text
AGrail4Agent/memory_bank-32.csv
```
