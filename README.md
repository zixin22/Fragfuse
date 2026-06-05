# FragFuse

FragFuse is a research pipeline for studying memory-based access-control risks in
LLM agents. The pipeline converts harmful or sensitive intent into fragments,
stores or injects those fragments into agent memory, and then evaluates whether
retrieval, reasoning, and downstream actions can reconstruct unsafe behavior.
It also includes defensive checks such as rule checking, guard-agent decisions,
and retrieval-time mitigation.

This repository is organized around three experiment tracks. Each track has its
own README with setup instructions, datasets, and runnable commands.

## Experiment Tracks

### RAP

[`RAP/README.md`](RAP/README.md) documents the WebShop-based RAP experiments,
including attack runs, RuleChecker evaluation, and GuardAgent evaluation.

### AgentHarm

[`Agentharm/README.md`](Agentharm/README.md) documents the minimized AgentHarm
tasks10 workflow, including term extraction, iterative masking, injection,
retrieval tests, and guardrail decision scripts.

### OSAgent and SeeAct

[`OSAgent and SeeAct/README.md`](OSAgent%20and%20SeeAct/README.md) documents the
environment, Docker setup, prepared datasets, and links to the OSAgent and
SeeAct run instructions.

## Responsible Use Notice

This repository is released only for research reproducibility and defensive
evaluation. The code and prompts are intended to help study and mitigate
memory-based access-control risks in LLM agents, not to enable misuse.

Please use these artifacts only in controlled benchmark or synthetic research
settings. Do not apply them to real-world deployed systems or third-party agents
without authorization.

We also provide mitigation guidance, including memory admission control,
retrieval-time policy checking, and access-control enforcement after memory
augmentation.
