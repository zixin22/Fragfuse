# SeeAct FragFuse Data

## Datasets

| File | Records | Role |
| --- | ---: | --- |
| `sample_labeled_benign.json` | 100 | Benign SeeAct tasks for utility evaluation. |
| `sample_labeled_harmful.json` | 100 | Original harmful SeeAct tasks. |
| `host_instructions.json` | 100 | Host instructions used to build carrier queries. |

## Pipeline

FragFuse builds two independent query inputs from the harmful SeeAct tasks:

```text
harmful query -> sensitive_fragments
host instruction + sensitive_fragments -> carrier_query
host instruction + fragment-recovery instruction -> attack_query
```

`carrier_query` and `attack_query` are run separately. The current code does not automatically run a multi-turn session where the carrier query is executed first and the attack query consumes that runtime context.

## Sensitive Fragments

`sample_labeled_harmful_sensitive_fragments-0.json` through `sample_labeled_harmful_sensitive_fragments-3.json` are fragment-extraction outputs from different iterations. The pipeline uses the merged result:

```text
sample_labeled_harmful_sensitive_fragments_merged.json
```

| File | Records | Field |
| --- | ---: | --- |
| `sample_labeled_harmful_sensitive_fragments-0.json` | 100 | Iteration 0 extraction output. |
| `sample_labeled_harmful_sensitive_fragments-1.json` | 100 | Iteration 1 extraction output. |
| `sample_labeled_harmful_sensitive_fragments-2.json` | 100 | Iteration 2 extraction output. |
| `sample_labeled_harmful_sensitive_fragments-3.json` | 100 | Iteration 3 extraction output. |
| `sample_labeled_harmful_sensitive_fragments_merged.json` | 100 | Final merged `sensitive_fragments` used by the pipeline. |

## FragFuse Queries

| File | Records | Field | Role |
| --- | ---: | --- | --- |
| `data/carrier_query.json` | 72 | `carrier_query` | Main carrier-query file. |
| `data/attack_query.json` | 72 | `attack_query`, `sensitive_fragments` | Main attack-query file. |

## Run

From `AGrail4Agent/`:

```bash
mkdir -p result/seeact
```

Carrier query:

```bash
python DAS/exp_SeeAct.py \
  --file ../Datasets/SeeAct/data/carrier_query.json \
  --guardrail_model gpt-4o
```

Attack query:

```bash
python DAS/exp_SeeAct.py \
  --file ../Datasets/SeeAct/data/attack_query.json \
  --guardrail_model gpt-4o
```

Benign utility:

```bash
python DAS/exp_SeeAct.py \
  --file ../Datasets/SeeAct/sample_labeled_benign.json \
  --guardrail_model gpt-4o
```
