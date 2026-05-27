# AgentHarm FragFuse Environment

This directory contains the AgentHarm evaluation components used by the FragFuse
pipeline. This README documents the environment and configuration requirements
for running the local FragFuse workflow.

Detailed pipeline steps, inputs, outputs, and execution order are documented in
[fragfuse_pipeline/readme.md](fragfuse_pipeline/readme.md).

## Environment Installation

The recommended setup is to use the Inspect Evals repository environment from
the repository root. Install the project dependencies with `uv sync`, then run
FragFuse scripts from an environment where the repository `src` directory is on
`PYTHONPATH`.

An equivalent conda environment can also be used. The environment should provide
Python and the dependencies required by Inspect Evals and AgentHarm, including:

- `inspect_ai`
- `inspect_evals`
- `openai`
- `numpy`
- `scipy`
- `pandas`

The local AgentHarm package must also be importable as `inspect_evals.agentharm`.
When commands are executed from the repository root, this is normally handled by
adding `src` to `PYTHONPATH`.

The FragFuse scripts have been validated with a conda environment named
`Agentharm`. The environment name is not required; any environment with the same
dependencies and import paths is sufficient.

## Model API Configuration

FragFuse stages that call a model or the LLM-AC guardrail read API credentials
from `fragfuse_pipeline/openai_key.txt`.

The file may contain only the API key:

- first non-comment line: API key for the selected OpenAI-compatible endpoint.

The file may also use `KEY=VALUE` lines when a custom endpoint is required:

- `OPENAI_API_KEY`: API key for the selected model endpoint.
- `OPENAI_BASE_URL`: base URL for the OpenAI-compatible endpoint.

`openai_key.txt` is a local secret file and must not be committed.

## Local Data and State

Important project-local paths:

- `fragfuse_pipeline/`: FragFuse pipeline scripts and method documentation.
- `fragfuse_output/`: pipeline state, generated query files, guardrail outputs,
  memory banks, and retrieval/fusion logs.
- `fragfuse_output/base_bank.jsonl`: shared base memory bank used to create
  per-task memory banks.
- `fragfuse_output/bank_records/bank_{task_id}.jsonl`: per-task retrieval-time
  memory banks.
- `benign_behaviors_dataset/`: benign host-query source used when constructing
  carrier queries.
- `benchmark/` and `agents/`: AgentHarm task tools, grading functions, and agent
  implementations used by the pipeline.

Paths such as `/home/...` inside benchmark prompts, tool outputs, and stored
AgentHarm trajectories are sandbox paths from the benchmark environment. They
are part of the benchmark semantics and should not be rewritten as repository
paths.

## FragFuse Documentation

The FragFuse method and current 1-to-8 pipeline are documented here:

- [fragfuse_pipeline/readme.md](fragfuse_pipeline/readme.md)

That document is the source of truth for command order, generated files,
prepared `iteration_4` inputs, guardrail evaluation, retrieval/fusion outputs,
and retrieval match-rate computation.

## License

The AgentHarm benchmark is licensed under the MIT License with the additional
terms in `LICENSE`.
